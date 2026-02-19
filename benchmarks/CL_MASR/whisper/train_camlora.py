#!/usr/bin/env python3
"""
LoRA continual fine-tuning recipe for a Whisper-based ASR system on Common Voice.

This is a LoRA-adapted version of `train_ft.py` from SpeechBrain CL_MASR:
https://github.com/speechbrain/benchmarks/tree/main/benchmarks/CL_MASR

Usage (same CLI style as SpeechBrain recipes):
> python train_lora_ft.py hparams/train_ft.yaml

Notes
-----
- Requires `peft` (Hugging Face PEFT): `pip install peft`
- Freezes the base Whisper weights and trains LoRA params (+ optionally token embeddings).
- Continual learning loop over locales is preserved (base locales evaluation, then sequentially add new locales).

Authors
- Adapted by ChatGPT (2026)
- Original recipe: Luca Della Libera 2023
"""

import logging
import os
import pathlib
import sys
import time
import copy

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

import math
from collections import defaultdict, deque

import torch.nn.functional as F
from torch.utils.data import DataLoader

def lora_vector_from_adapter(base_template, adapter_dir, device="cpu"):
    """Return a 1D vector containing all LoRA params (A/B) concatenated."""
    base = copy.deepcopy(base_template).to(device)
    m = PeftModel.from_pretrained(base, adapter_dir).to(device)
    sd = m.state_dict()

    vecs = []
    for k, v in sd.items():
        if "lora_" in k.lower():
            vecs.append(v.detach().float().flatten().cpu())
    if not vecs:
        raise RuntimeError(f"No LoRA params found in {adapter_dir}")
    return torch.cat(vecs, dim=0)  # [D]

def cosine_sim_matrix(vectors):
    """vectors: list of 1D cpu tensors"""
    X = torch.stack(vectors, dim=0)  # [N,D]
    X = F.normalize(X, p=2, dim=1)
    return X @ X.t()  # [N,N]

def select_most_similar_group(locales, adapter_dir_by_locale, base_template, group_size=4, device="cpu"):
    """
    Return a list of locales of length group_size that are most similar
    (greedy: pick seed with highest average similarity, then add most similar).
    """
    # precompute vectors
    vecs = []
    for loc in locales:
        vecs.append(lora_vector_from_adapter(base_template, adapter_dir_by_locale[loc], device=device))
    S = cosine_sim_matrix(vecs)  # [N,N]

    n = len(locales)
    if n <= group_size:
        return locales

    # pick seed with max average similarity
    avg = (S.sum(dim=1) - 1) / (n - 1)
    seed = int(torch.argmax(avg).item())

    chosen = [seed]
    remaining = set(range(n)) - {seed}

    # greedy add
    while len(chosen) < group_size:
        best_j, best_score = None, -1e9
        for j in remaining:
            # similarity to current set = mean sim
            score = S[j, chosen].mean().item()
            if score > best_score:
                best_score, best_j = score, j
        chosen.append(best_j)
        remaining.remove(best_j)

    return [locales[i] for i in chosen]

def _peft_state_dict_from_dir(adapter_dir, device="cpu"):
    """Load adapter-only state dict from a saved PEFT adapter directory."""
    # PEFT saves adapter weights in adapter_model.bin / safetensors depending on version
    # We'll rely on HF/PEFT to load into a temp model then read state_dict for safety.
    return adapter_dir  # placeholder (we'll load via PeftModel and then state_dict)


def _average_lora_weights_into(base_hf, adapter_dirs, lora_cfg, device="cpu"):
    """
    Create a fresh PEFT LoRA model from base_hf and load averaged LoRA weights
    from multiple adapter_dirs as initialization (warm-start).
    We average LoRA A/B tensors with matching keys.
    """
    # fresh LoRA wrapper
    merged = get_peft_model(base_hf, lora_cfg)

    # collect per-adapter LoRA tensors
    avg = {}
    count = 0

    for ad in adapter_dirs:
        teacher = PeftModel.from_pretrained(copy.deepcopy(base_hf), ad).to(device)
        sd = teacher.state_dict()
        # only LoRA params (and optionally embeddings if you want)
        lora_items = {k: v.detach().clone() for k, v in sd.items() if "lora_" in k.lower()}
        if not lora_items:
            raise RuntimeError(f"No LoRA params found in adapter: {ad}")

        if count == 0:
            for k, v in lora_items.items():
                avg[k] = v.to(device)
        else:
            for k, v in lora_items.items():
                if k in avg:
                    avg[k] += v.to(device)
                else:
                    # if key missing in some adapters, just start accumulating
                    avg[k] = v.to(device)
        count += 1

        # free teacher asap
        del teacher
        torch.cuda.empty_cache()

    for k in avg:
        avg[k] /= float(count)

    # load averaged weights into merged
    missing, unexpected = merged.load_state_dict(avg, strict=False)
    # missing is expected for base weights etc.
    return merged


@torch.no_grad()
def _teacher_logits_for_batch(hparams, base_template, adapter_dir, batch, device):
    """Forward pass teacher(base + adapter) to get logits for KL preservation."""
    # build teacher model
    sb_whisper = hparams["whisper"]
    sb_whisper.model = copy.deepcopy(base_template).to(device)
    sb_whisper.model = PeftModel.from_pretrained(sb_whisper.model, adapter_dir).to(device)
    sb_whisper.model.eval()

    batch = batch.to(device)
    wavs, _ = batch.sig
    bos_tokens, _ = batch.tokens_bos

    enc_out, logits, _ = sb_whisper(wavs, bos_tokens)
    return logits


def _masked_kl_div(teacher_logits, student_logits, target_ids, ignore_index, temperature=1.0, topk=0):
    """
    KL( p_teacher || p_student ) with masking on target_ids != ignore_index.
    teacher_logits/student_logits: [B, T, V]
    target_ids: [B, T]
    """
    T = float(temperature)
    B, L, V = student_logits.shape

    # mask (where we have valid target tokens)
    mask = (target_ids != ignore_index).float()  # [B, T]

    if topk and topk > 0:
        # restrict to teacher top-k to speed up
        with torch.no_grad():
            topv, topi = torch.topk(teacher_logits, k=topk, dim=-1)  # [B,T,K]
        # gather student logits at teacher top-k indices
        student_sel = torch.gather(student_logits, dim=-1, index=topi)
        teacher_sel = topv

        t_prob = F.softmax(teacher_sel / T, dim=-1)
        s_logprob = F.log_softmax(student_sel / T, dim=-1)
        kl = torch.sum(t_prob * (torch.log(t_prob + 1e-12) - s_logprob), dim=-1)  # [B,T]
    else:
        t_prob = F.softmax(teacher_logits / T, dim=-1)
        s_logprob = F.log_softmax(student_logits / T, dim=-1)
        kl = torch.sum(t_prob * (torch.log(t_prob + 1e-12) - s_logprob), dim=-1)  # [B,T]

    kl = kl * mask
    denom = mask.sum().clamp_min(1.0)
    return kl.sum() / denom


def cam_consolidate_merge(
    hparams,
    run_opts,
    base_template,
    locales_to_merge,
    adapter_dirs_by_locale,
    out_merged_dir,
):
    """
    Constraint-aware merge (CAM):
    - warm-start merged LoRA by averaging LoRA weights across locales_to_merge
    - optimize merged LoRA on small dev subsets:
        L = CE + lambda * KL(teacher || merged)
      where teacher is the per-locale expert adapter.
    - save merged adapter to out_merged_dir
    """

    device = run_opts.get("device", "cuda")
    os.makedirs(out_merged_dir, exist_ok=True)

    # 1) Prepare per-locale dev loaders (small subset)
    tokenizer = hparams["whisper"].tokenizer

    dev_loaders = {}
    for loc in locales_to_merge:
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [loc],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )
        hparams["forced_decoder_locale"] = loc
        _, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # limit dev batches for CAM
        cam_dev_max_batches = int(hparams.get("cam_dev_max_batches", 20))
        loader = DataLoader(
            valid_data,
            batch_size=hparams["valid_dataloader_kwargs"].get("batch_size", 1),
            shuffle=False,
            collate_fn=hparams["valid_dataloader_kwargs"]["collate_fn"],
            num_workers=hparams["valid_dataloader_kwargs"].get("num_workers", 0),
            pin_memory=True,
        )
        dev_loaders[loc] = (loader, cam_dev_max_batches)

    # 2) Build warm-start merged model: base + averaged LoRA
    sb_whisper = hparams["whisper"]
    base_hf = copy.deepcopy(base_template).to(device)

    # match LoRA config used in training
    freeze_encoder = bool(hparams.get("freeze_encoder", False))
    target_modules = _collect_lora_linear_module_names(base_hf, freeze_encoder=freeze_encoder)

    lora_cfg = LoraConfig(
        r=int(hparams.get("lora_r", 32)),
        lora_alpha=int(hparams.get("lora_alpha", 64)),
        lora_dropout=float(hparams.get("lora_dropout", 0.05)),
        bias=str(hparams.get("lora_bias", "none")),
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    adapter_dirs = [adapter_dirs_by_locale[loc] for loc in locales_to_merge]
    merged_hf = _average_lora_weights_into(
        base_hf=base_hf, adapter_dirs=adapter_dirs, lora_cfg=lora_cfg, device=device
    )

    # freeze base, train LoRA only (and optionally embeddings)
    for n, p in merged_hf.named_parameters():
        p.requires_grad = False
        if "lora_" in n.lower():
            p.requires_grad = True
    if bool(hparams.get("train_embeddings", True)):
        for n, p in merged_hf.named_parameters():
            if any(k in n for k in ["embed_tokens", "decoder.embed_tokens"]):
                p.requires_grad = True

    merged_hf.train()
    sb_whisper.model = merged_hf  # write back into SB wrapper

    # optimizer
    cam_lr = float(hparams.get("cam_lr", 5e-4))
    opt = torch.optim.AdamW([p for p in merged_hf.parameters() if p.requires_grad], lr=cam_lr)

    # losses / params
    ce = torch.nn.CrossEntropyLoss(ignore_index=int(hparams["ignore_index"]))
    cam_lambda = float(hparams.get("cam_lambda", 1.0))
    cam_steps = int(hparams.get("cam_steps", 800))
    temp = float(hparams.get("cam_temperature", 1.0))
    topk = int(hparams.get("cam_topk_kl", 0))
    ignore_index = int(hparams["ignore_index"])

    # 3) Round-robin optimization across locales
    locale_cycle = deque(locales_to_merge)
    iters = {loc: iter(dev_loaders[loc][0]) for loc in locales_to_merge}
    used_batches = defaultdict(int)

    step = 0
    while step < cam_steps:
        loc = locale_cycle[0]
        locale_cycle.rotate(-1)

        loader, max_batches = dev_loaders[loc]
        if used_batches[loc] >= max_batches:
            # skip locale if exhausted
            continue

        try:
            batch = next(iters[loc])
        except StopIteration:
            iters[loc] = iter(loader)
            batch = next(iters[loc])

        used_batches[loc] += 1
        step += 1

        batch = batch.to(device)
        wavs, _ = batch.sig
        bos_tokens, _ = batch.tokens_bos
        tokens_eos, _ = batch.tokens_eos

        # student forward
        enc_out, student_logits, _ = sb_whisper(wavs, bos_tokens)

        # fit loss
        fit_loss = ce(student_logits.flatten(end_dim=-2), tokens_eos.flatten())

        # preserve loss: teacher is (base + locale expert)
        with torch.no_grad():
            teacher_logits = _teacher_logits_for_batch(
                hparams=hparams,
                base_template=base_template,
                adapter_dir=adapter_dirs_by_locale[loc],
                batch=batch,
                device=device,
            )

        pres_loss = _masked_kl_div(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=tokens_eos,
            ignore_index=ignore_index,
            temperature=temp,
            topk=topk,
        )

        loss = fit_loss + cam_lambda * pres_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in merged_hf.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 50 == 0:
            logging.info(
                f"[CAM] step={step}/{cam_steps} loc={loc} "
                f"fit={fit_loss.item():.4f} pres={pres_loss.item():.4f} total={loss.item():.4f}"
            )

    # 4) Save merged adapter
    merged_hf.eval()
    merged_hf.save_pretrained(out_merged_dir)
    tokenizer.save_pretrained(out_merged_dir)
    logging.info(f"[CAM] Saved merged adapter: {out_merged_dir}")

    return out_merged_dir

def _get_hf_model_from_sb_whisper(sb_whisper):
    """
    SpeechBrain CL_MASR uses hparams['whisper'] which wraps a HF Whisper model.
    Common patterns:
      - sb_whisper.model is the HF model
      - sb_whisper.tokenizer is the HF tokenizer

    If your wrapper differs, adjust this function.
    """
    if hasattr(sb_whisper, "model"):
        return sb_whisper.model
    raise AttributeError(
        "Cannot locate HF model inside hparams['whisper']. "
        "Expected hparams['whisper'].model to exist."
    )


def _set_hf_model_into_sb_whisper(sb_whisper, hf_model):
    """Write-back HF model into the SpeechBrain whisper wrapper."""
    if hasattr(sb_whisper, "model"):
        sb_whisper.model = hf_model
        return
    raise AttributeError(
        "Cannot set HF model into hparams['whisper']. "
        "Expected hparams['whisper'].model to exist."
    )

def _collect_lora_linear_module_names(base_model, freeze_encoder: bool, allowlist=None):
    """
    base_model.named_modules()를 훑어서 LoRA target_modules 이름 리스트를 구성.
    freeze_encoder=True면 encoder 경로에 있는 모듈은 제외.
    allowlist는 최종적으로 이름에 포함되어야 하는 suffix(예: q_proj, fc1 등)들.
    """
    if allowlist is None:
        allowlist = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj_out"}

    target_set = set()

    for name, module in base_model.named_modules():
        # LoRA는 보통 Linear류에 적용
        if not isinstance(module, torch.nn.Linear):
            continue

        # encoder 제외 옵션
        # WhisperModel이면 name이 "encoder.layers.0.self_attn.q_proj" 형태,
        # WhisperForConditionalGeneration이면 "model.encoder.layers..." 형태 가능
        is_encoder = (
            name.startswith("encoder.")
            or name.startswith("model.encoder.")
            or ".encoder." in name
        )
        is_decoder = (
            name.startswith("decoder.")
            or name.startswith("model.decoder.")
            or ".decoder." in name
        )

        if freeze_encoder and is_encoder and not is_decoder:
            continue

        # suffix 기반 필터 (q_proj, fc1 등만)
        last = name.split(".")[-1]
        if last in allowlist:
            target_set.add(name)

    # PEFT의 target_modules는 "모듈 이름 suffix" 리스트를 받는 경우가 많아서
    # 여기서는 suffix들만 반환 (q_proj, k_proj, ...)
    # 단, 커스텀하게 full path로 주고 싶으면 여기 로직을 바꾸면 됨.
    return list(target_set)


def make_fresh_lora_whisper(hparams, locale, train_embeddings=True):
    sb_whisper = hparams["whisper"]
    base_hf = sb_whisper.model

    if isinstance(base_hf, PeftModel):
        base_hf = base_hf.get_base_model()

    freeze_encoder = bool(hparams.get("freeze_encoder", False))

    target_modules = _collect_lora_linear_module_names(
        base_hf, freeze_encoder=freeze_encoder
    )

    logging.info(f"[LoRA] freeze_encoder={freeze_encoder} target_modules={target_modules}")

    lora_cfg = LoraConfig(
        r=int(hparams.get("lora_r", 32)),
        lora_alpha=int(hparams.get("lora_alpha", 64)),
        lora_dropout=float(hparams.get("lora_dropout", 0.05)),
        bias=str(hparams.get("lora_bias", "none")),
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    hf = get_peft_model(base_hf, lora_cfg)

    # freeze base params
    for _, p in hf.named_parameters():
        p.requires_grad = False

    # unfreeze LoRA params
    for name, p in hf.named_parameters():
        if "lora_" in name.lower():
            p.requires_grad = True

    # embedding 학습 옵션 (새 언어 토큰 임베딩 학습하려면 True 추천)
    if train_embeddings:
        for name, p in hf.named_parameters():
            if any(k in name for k in ["embed_tokens", "decoder.embed_tokens"]):
                p.requires_grad = True

    sb_whisper.model = hf
    return hf

def save_lora_adapter(hparams, adapter_dir):
    os.makedirs(adapter_dir, exist_ok=True)
    hf = hparams["whisper"].model
    if not isinstance(hf, PeftModel):
        raise RuntimeError("Expected PeftModel to save adapter.")
    hf.save_pretrained(adapter_dir)
    # tokenizer도 같이 저장해두면 재현/배포 편함
    hparams["whisper"].tokenizer.save_pretrained(adapter_dir)


def load_lora_adapter_into_whisper(hparams, adapter_dir):
    """
    base whisper + adapter 로드해서 hparams['whisper'].model 에 넣기
    """
    sb_whisper = hparams["whisper"]
    base_hf = sb_whisper.model
    if isinstance(base_hf, PeftModel):
        base_hf = base_hf.get_base_model()
    sb_whisper.model = PeftModel.from_pretrained(base_hf, adapter_dir)
    return sb_whisper.model

def ensure_new_token_embedding_trainable(hparams):
    """
    After adding a new language token and resizing embeddings, PEFT/base freezing might
    leave embeddings frozen. If train_embeddings=True, force them trainable again.
    """
    if not bool(hparams.get("train_embeddings", True)):
        return
    sb_whisper = hparams["whisper"]
    hf_model = _get_hf_model_from_sb_whisper(sb_whisper)
    for name, p in hf_model.named_parameters():
        if any(k in name for k in ["embed_tokens", "decoder.embed_tokens"]):
            p.requires_grad = True

# -----------------------------
# SpeechBrain Brain
# -----------------------------
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos

        # Forward encoder + decoder
        if self.hparams.gradient_checkpointing:
            wavs.requires_grad_()
            enc_out, logits, _ = torch.utils.checkpoint.checkpoint(
                self.modules.whisper, wavs, bos_tokens
            )
        else:
            enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        hyps = None
        if stage != sb.Stage.TRAIN:
            hyps, _ = self.modules.whisper.generate(
                audio_features=enc_out,
                forced_decoder_locale=self.hparams.forced_decoder_locale,
                max_gen_tokens=self.hparams.max_gen_tokens,
            )

        return logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        logits, hyps = predictions
        ids = batch.id
        tokens_eos, _ = batch.tokens_eos

        loss = self.hparams.ce_loss(
            logits.flatten(end_dim=-2), tokens_eos.flatten()
        )

        if stage != sb.Stage.TRAIN:
            target_words = batch.target_wrd

            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )

            if self.hparams.normalize_transcripts:
                predicted_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in predicted_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            stats_meta_data = {"epoch": epoch, "lr": old_lr}
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta_data,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)


# -----------------------------
# Data pipelines (unchanged)
# -----------------------------
def dataio_prepare(hparams, tokenizer):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_folder"], "train.csv"),
        replacements={"data_root": hparams["data_folder"]},
    )

    if hparams["sorting"] in ["descending", "ascending"]:
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=hparams["sorting"] == "descending",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        hparams["train_dataloader_kwargs"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise ValueError(
            f"`sorting` ({hparams['sorting']}) must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_folder"], "dev.csv"),
        replacements={"data_root": hparams["data_folder"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_folder"], "test.csv"),
        replacements={"data_root": hparams["data_folder"]},
    ).filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    datasets = [train_data, valid_data, test_data]

    @sb.utils.data_pipeline.takes("mp3")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(mp3):
        info = torchaudio.info(mp3)
        sig = sb.dataio.dataio.read_audio(mp3)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"]
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wrd", "locale")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "target_wrd")
    def text_pipeline(wrd, locale):
        if locale.startswith("zh"):
            locale = "zh"
        locale = locale.lower()
        language = tokenizer.supported_languages.get(locale, "english")
        tokenizer.set_prefix_tokens(language=language)

        tokens_list = tokenizer.encode(wrd)
        assert sum(i == tokenizer.unk_token_id for i in tokens_list) == 1

        bos_index, tokens_list, eos_index = (
            tokens_list[0],
            tokens_list[1:-1],
            tokens_list[-1],
        )
        tokens_list = tokens_list[: hparams["max_target_length"] - 1]
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos

        if hparams["normalize_transcripts"]:
            wrd = tokenizer._normalize(wrd)
        wrd = wrd.split(" ")
        for i, char in enumerate(wrd):
            if len(char) == 0:
                wrd[i] = " "
        yield wrd

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "target_wrd"]
    )
    return train_data, valid_data, test_data


# -----------------------------
# Test / Train (mostly unchanged)
# -----------------------------
def test(hparams, run_opts, locales, wer_file="wer_test.txt"):
    for locale in locales:
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        if locale in ["zh-CN", "ja"]:
            hparams["wer_computer"] = lambda *args, **kwargs: sb.utils.metric_stats.ErrorRateStats(
                split_tokens=True
            )
        else:
            hparams["wer_computer"] = sb.utils.metric_stats.ErrorRateStats

        hparams["forced_decoder_locale"] = locale
        tokenizer = hparams["whisper"].tokenizer
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        asr_brain = ASR(modules=hparams["modules"], hparams=hparams, run_opts=run_opts)
        asr_brain.tokenizer = tokenizer

        locale_folder = os.path.join(hparams["output_folder"], locale)
        os.makedirs(locale_folder, exist_ok=True)
        asr_brain.hparams.wer_file = os.path.join(locale_folder, wer_file)

        if hparams["skip_test"]:
            train_log_backup = asr_brain.hparams.train_logger.save_file
            asr_brain.hparams.train_logger.save_file = (
                asr_brain.hparams.wer_file
            ) = os.path.join(locale_folder, "tmp.txt")
            test_data.data_ids = list(test_data.data.keys())[:1]
            test_data.data = {k: test_data.data[k] for k in test_data.data_ids}
            asr_brain.evaluate(
                test_data,
                min_key="WER",
                test_loader_kwargs=hparams["valid_dataloader_kwargs"],
            )
            os.remove(asr_brain.hparams.wer_file)
            asr_brain.hparams.train_logger.save_file = train_log_backup
            asr_brain.hparams.wer_file = os.path.join(locale_folder, wer_file)
        else:
            asr_brain.evaluate(
                test_data,
                min_key="WER",
                test_loader_kwargs=hparams["valid_dataloader_kwargs"],
            )

    if not hparams["skip_test"]:
        try:
            profile(hparams, run_opts)
        except Exception:
            logging.warning(
                "Install ptflops and torchinfo to profile the model "
                "(e.g. `pip install ptflops torchinfo`)"
            )


def train(hparams, run_opts):
    """
    1) locale별 LoRA expert 학습/저장
    2) 매 K개마다 CAM consolidation(Constraint-aware merge)
    3) merged adapter로 해당 언어들 테스트
    """
    adapters_root = os.path.join(hparams["output_folder"], "lora_adapters")
    merged_root = os.path.join(hparams["output_folder"], "lora_merged")
    os.makedirs(adapters_root, exist_ok=True)
    os.makedirs(merged_root, exist_ok=True)

    # base template (HF whisper base) - keep on CPU, clone to GPU as needed
    base_template = copy.deepcopy(hparams["whisper"].model).cpu()

    # mapping: locale -> current adapter dir (initially its own expert)
    adapter_dir_by_locale = {}
    # list of locales trained so far (for consolidation)
    trained_locales = []

    K = int(hparams.get("consolidate_every", 4))

    for i, locale in enumerate(hparams["new_locales"]):
        logging.info(f"[CL] Training LoRA expert for locale={locale} ({i+1}/{len(hparams['new_locales'])})")

        # reset model to base for each expert (language expert library)
        hparams["whisper"].model = copy.deepcopy(base_template).to("cuda")

        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        # tokenizer update: add language token (keep consistent across experts)
        tokenizer = hparams["whisper"].tokenizer
        new_tokens = [f"<|{locale.lower()}|>"]
        tokenizer._additional_special_tokens += new_tokens
        tokenizer.supported_languages.update({locale.lower(): locale.lower()})
        tokenizer.to_language_codes.update({locale.lower(): locale.lower()})

        new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
        tokenizer.add_tokens(new_tokens)

        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))
        
        # attach fresh LoRA
        make_fresh_lora_whisper(
            hparams,
            locale,
            train_embeddings=bool(hparams.get("train_embeddings", True)),
        )
        ensure_new_token_embedding_trainable(hparams)

        hparams["forced_decoder_locale"] = locale

        # dataset
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # locale별 checkpoint 폴더
        checkpoint_folder = os.path.join(hparams["save_folder"], f"lora_{locale}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_folder)

        # scheduler reset
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0

        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
        )
        asr_brain.tokenizer = tokenizer

        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0

        # train expert
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # save adapter
        adapter_dir = os.path.join(adapters_root, locale)
        save_lora_adapter(hparams, adapter_dir)
        adapter_dir_by_locale[locale] = adapter_dir
        trained_locales.append(locale)

        # evaluate expert
        if isinstance(hparams["whisper"].model, PeftModel):
            hparams["whisper"].model = hparams["whisper"].model.get_base_model()
        load_lora_adapter_into_whisper(hparams, adapter_dir)
        test(hparams, run_opts, [locale], f"wer_test_after_{locale}.txt")

        # ----------------------------
        # CAM consolidation every K locales
        # ----------------------------
        if (len(trained_locales) % K) == 0:
            # merge candidates: you can merge among all trained, or recent window
            candidates = trained_locales  # or trained_locales[-(2*K):] for speed
            locales_to_merge = select_most_similar_group(
                locales=candidates,
                adapter_dir_by_locale=adapter_dir_by_locale,
                base_template=base_template,
                group_size=K,
                device="cpu",
            )
            
            merged_name = f"merge_{len(trained_locales)//K:04d}"
            out_merged_dir = os.path.join(merged_root, merged_name)

            logging.info(f"[CAM] Consolidating locales={locales_to_merge} -> {merged_name}")

            cam_consolidate_merge(
                hparams=hparams,
                run_opts=run_opts,
                base_template=base_template,
                locales_to_merge=locales_to_merge,
                adapter_dirs_by_locale=adapter_dir_by_locale,
                out_merged_dir=out_merged_dir,
            )

            # After consolidation: map those locales to the merged adapter
            for loc in locales_to_merge:
                adapter_dir_by_locale[loc] = out_merged_dir

            # (선택) 개별 어댑터 폴더 삭제하고 싶으면 여기에 처리
            # for loc in locales_to_merge:
            #     old_dir = os.path.join(adapters_root, loc)
            #     if os.path.isdir(old_dir):
            #         shutil.rmtree(old_dir)

            # quick evaluation of merged adapter on merged locales
            # load base + merged
            hparams["whisper"].model = copy.deepcopy(base_template).to("cuda")
            load_lora_adapter_into_whisper(hparams, out_merged_dir)

            # test each locale under merged adapter
            for loc in locales_to_merge:
                hparams["forced_decoder_locale"] = loc
                test(hparams, run_opts, [loc], f"wer_test_after_{merged_name}_{loc}.txt")


def profile(hparams, run_opts):
    import ptflops
    import torchinfo

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.whisper = hparams["whisper"]
            self.wavs = torch.randn(
                1, hparams["sample_rate"], device="cuda"
            )
            self.bos_tokens = torch.ones(
                1,
                self.whisper.model.config.max_length,
                dtype=torch.int,
                device="cuda",
            )

        @torch.no_grad()
        def forward(self, _=None):
            enc_out, logits, _ = self.whisper(self.wavs, self.bos_tokens)
            return logits

    model = Model().eval().to("cuda")
    macs, params = ptflops.get_model_complexity_info(
        model, (1,), as_strings=True, print_per_layer_stat=False
    )
    time_start = time.time()
    model()
    torch.cuda.synchronize()
    time_stop = time.time() - time_start
    max_mem = torch.cuda.max_memory_allocated("cuda") / 10**9
    result = {"MACs": macs, "memory": max_mem, "time": time_stop}
    logging.info(torchinfo.summary(model, verbose=0))
    logging.info(result)


# -----------------------------
# Main (mostly unchanged)
# -----------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["train_logger"].save_file = hparams["train_logger"].save_file.replace(
        ".txt",
        f"_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}_lora.txt",
    )

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    class CustomPaddedBatch(PaddedBatch):
        """PaddedBatch with custom padding values."""

        def __init__(self, examples, *args, **kwargs):
            for k in ["tokens_bos", "tokens_eos"]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0.0
                if k == "tokens_bos":
                    pad_value = hparams["whisper"].tokenizer.pad_token_id
                elif k == "tokens_eos":
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    example[k] = torch.nn.functional.pad(
                        x, [0, max_len - len(x)], value=pad_value
                    )
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch

    start_time = time.time()
    train(hparams, run_opts)
    duration = time.time() - start_time
    logging.info(f"Time elapsed: {duration} seconds")
