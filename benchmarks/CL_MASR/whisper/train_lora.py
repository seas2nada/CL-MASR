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
    # 0) base 평가 (LoRA 없이)
    # test(hparams, run_opts, hparams["base_locales"], "wer_test_before.txt")

    adapters_root = os.path.join(hparams["output_folder"], "lora_adapters")
    os.makedirs(adapters_root, exist_ok=True)

    base_template = copy.deepcopy(hparams["whisper"].model).cpu()

    # 1) locale별 독립 adapter 학습
    for i, locale in enumerate(hparams["new_locales"]):
        # locale loop 시작마다
        hparams["whisper"].model = copy.deepcopy(base_template).to("cuda")

        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        # ---- tokenizer에 새 language token 추가 (원본 유지) ----
        new_tokens = [f"<|{locale.lower()}|>"]
        tokenizer = hparams["whisper"].tokenizer
        tokenizer._additional_special_tokens += new_tokens
        tokenizer.supported_languages.update({locale.lower(): locale.lower()})
        tokenizer.to_language_codes.update({locale.lower(): locale.lower()})

        new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
        tokenizer.add_tokens(new_tokens)

        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))

        # ---- locale별 “새 LoRA attach” ----
        make_fresh_lora_whisper(
            hparams,
            locale,
            train_embeddings=bool(hparams.get("train_embeddings", True)),
        )

        # forced decoder locale
        hparams["forced_decoder_locale"] = locale

        # dataset
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # checkpoint 폴더 (locale별)
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

        # ---- FT ----
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # ---- adapter 저장 ----
        adapter_dir = os.path.join(adapters_root, locale)
        save_lora_adapter(hparams, adapter_dir)

        # ---- 테스트: base + 해당 adapter ----
        # (혹시 fit 동안 내부적으로 상태가 바뀌었을 수 있으니, base+adapter 로드해서 평가 권장)
        if isinstance(hparams["whisper"].model, PeftModel):
            hparams["whisper"].model = hparams["whisper"].model.get_base_model()
        load_lora_adapter_into_whisper(hparams, adapter_dir)

        test(
            hparams,
            run_opts,
            [locale],  # 요청대로 FT된 language에 대해 test
            f"wer_test_after_{locale}.txt",
        )


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
