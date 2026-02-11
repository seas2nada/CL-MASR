#!/usr/bin/env python3
"""
train_adapter.py

Bottleneck Adapter continual fine-tuning recipe for Whisper ASR on Common Voice
(SpeechBrain CL_MASR style).

- Per-locale independent adapter:
  for each locale:
    1) start from pristine base HF Whisper model (deepcopy)
    2) (optional) add language token + resize embeddings
    3) attach bottleneck adapters (encoder optional)
    4) freeze base, train adapters (+ optional embeddings)
    5) save adapter weights
    6) restore pristine base
    7) load adapter weights
    8) test that locale

Extra hparams (optional):
  freeze_encoder: true|false (default: false)
  adapter_dim: int (default: 64)          # bottleneck rank
  adapter_dropout: float (default: 0.0)
  train_embeddings: true|false (default: false)
  per_locale_add_lang_token: true|false (default: true)
  eval_base_before: true|false (default: false)

Usage:
> python train_adapter.py hparams/train_ft.yaml
"""

import copy
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice


# -----------------------------
# SB Whisper wrapper helpers
# -----------------------------
def _get_hf_model_from_sb_whisper(sb_whisper):
    if hasattr(sb_whisper, "model"):
        return sb_whisper.model
    raise AttributeError("Expected hparams['whisper'].model to exist.")


def _set_hf_model_into_sb_whisper(sb_whisper, hf_model):
    if hasattr(sb_whisper, "model"):
        sb_whisper.model = hf_model
        return
    raise AttributeError("Expected hparams['whisper'].model to exist.")


def _find_whisper_enc_dec_layers(hf_model) -> Tuple[List[torch.nn.Module], List[torch.nn.Module]]:
    """
    Return encoder layers list, decoder layers list.
    Works for common HF WhisperModel / WhisperForConditionalGeneration structures.
    """
    # Case A: WhisperModel has .encoder.layers / .decoder.layers
    if hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layers"):
        enc_layers = list(hf_model.encoder.layers)
        dec_layers = list(hf_model.decoder.layers) if hasattr(hf_model, "decoder") else []
        return enc_layers, dec_layers

    # Case B: WhisperForConditionalGeneration has .model.encoder.layers
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "encoder") and hasattr(hf_model.model.encoder, "layers"):
        enc_layers = list(hf_model.model.encoder.layers)
        dec_layers = list(hf_model.model.decoder.layers) if hasattr(hf_model.model, "decoder") else []
        return enc_layers, dec_layers

    raise AttributeError("Cannot locate encoder/decoder layers in the HF Whisper model.")


def _get_d_model(hf_model) -> int:
    cfg = getattr(hf_model, "config", None)
    if cfg is not None:
        for k in ["d_model", "hidden_size"]:
            if hasattr(cfg, k):
                return int(getattr(cfg, k))
    # fallback: infer from first attention projection
    for _, p in hf_model.named_parameters():
        if p.dim() == 2:
            return int(p.shape[1])
    raise RuntimeError("Cannot infer d_model.")


# -----------------------------
# Bottleneck Adapter module
# -----------------------------
class BottleneckAdapter(torch.nn.Module):
    def __init__(self, d_model: int, adapter_dim: int, dropout: float = 0.0):
        super().__init__()
        self.down = torch.nn.Linear(d_model, adapter_dim, bias=True)
        self.act = torch.nn.GELU()
        self.up = torch.nn.Linear(adapter_dim, d_model, bias=True)
        self.drop = torch.nn.Dropout(dropout)

        # 안정적 초기화: up를 0에 가깝게 두면 초기엔 거의 identity
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self.drop(self.up(self.act(self.down(x))))


# -----------------------------
# Adapter injection via hooks
# -----------------------------
class AdapterHookManager:
    """
    Attaches adapters to encoder/decoder layers via forward hooks.
    """
    def __init__(self):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def attach_to_layers(
        self,
        layers: List[torch.nn.Module],
        adapters: torch.nn.ModuleList,
    ):
        assert len(layers) == len(adapters), "layers and adapters length mismatch"

        for layer, adapter in zip(layers, adapters):
            def _hook(module, inp, out, adapter=adapter):
                # HF transformer layers often return tuple: (hidden_states, ...)
                if isinstance(out, tuple):
                    hs = out[0]
                    hs2 = hs + adapter(hs)
                    return (hs2,) + out[1:]
                # else out is hidden_states
                return out + adapter(out)

            self.handles.append(layer.register_forward_hook(_hook))


def make_fresh_adapters(hparams, locale: str, hook_mgr: AdapterHookManager):
    """
    Create and attach adapters, freeze base model, unfreeze adapters (+ optional embeddings).
    """
    hf = _get_hf_model_from_sb_whisper(hparams["whisper"])
    hook_mgr.clear()

    freeze_encoder = bool(hparams.get("freeze_encoder", False))
    adapter_dim = int(hparams.get("adapter_dim", 64))
    adapter_dropout = float(hparams.get("adapter_dropout", 0.0))
    train_embeddings = bool(hparams.get("train_embeddings", False))

    d_model = _get_d_model(hf)
    enc_layers, dec_layers = _find_whisper_enc_dec_layers(hf)

    # Build adapters
    enc_adapters = torch.nn.ModuleList()
    dec_adapters = torch.nn.ModuleList()

    if not freeze_encoder:
        for _ in enc_layers:
            enc_adapters.append(BottleneckAdapter(d_model, adapter_dim, adapter_dropout))

    for _ in dec_layers:
        dec_adapters.append(BottleneckAdapter(d_model, adapter_dim, adapter_dropout))

    # register adapters as modules so optimizer can see them
    # (put into hparams["modules"] so SpeechBrain moves them to device)
    hparams["modules"]["enc_adapters"] = enc_adapters
    hparams["modules"]["dec_adapters"] = dec_adapters

    # Freeze base HF parameters
    for _, p in hf.named_parameters():
        p.requires_grad = False

    # Unfreeze adapters
    for p in enc_adapters.parameters():
        p.requires_grad = True
    for p in dec_adapters.parameters():
        p.requires_grad = True

    # Optionally train embeddings (주의: 전체 embedding 열림)
    if train_embeddings:
        for name, p in hf.named_parameters():
            if any(k in name for k in ["embed_tokens", "decoder.embed_tokens"]):
                p.requires_grad = True

    # Attach hooks (adapters applied on forward)
    if not freeze_encoder:
        hook_mgr.attach_to_layers(enc_layers, enc_adapters)
    hook_mgr.attach_to_layers(dec_layers, dec_adapters)

    n_train = sum(p.numel() for p in hf.parameters() if p.requires_grad) \
              + sum(p.numel() for p in enc_adapters.parameters() if p.requires_grad) \
              + sum(p.numel() for p in dec_adapters.parameters() if p.requires_grad)

    logging.info(
        f"[Adapter] locale={locale} freeze_encoder={freeze_encoder} "
        f"adapter_dim={adapter_dim} train_embeddings={train_embeddings} "
        f"trainable_params≈{n_train:,}"
    )


def extract_adapter_state(hparams) -> Dict[str, torch.Tensor]:
    """
    Save only adapter weights (enc_adapters/dec_adapters).
    """
    state = {}
    if "enc_adapters" in hparams["modules"]:
        for k, v in hparams["modules"]["enc_adapters"].state_dict().items():
            state["enc_adapters." + k] = v.detach().cpu().clone()
    if "dec_adapters" in hparams["modules"]:
        for k, v in hparams["modules"]["dec_adapters"].state_dict().items():
            state["dec_adapters." + k] = v.detach().cpu().clone()
    return state


def load_adapter_state(hparams, state: Dict[str, torch.Tensor]):
    if "enc_adapters" in hparams["modules"]:
        enc_sd = {k.replace("enc_adapters.", ""): v for k, v in state.items() if k.startswith("enc_adapters.")}
        hparams["modules"]["enc_adapters"].load_state_dict(enc_sd, strict=True)
    if "dec_adapters" in hparams["modules"]:
        dec_sd = {k.replace("dec_adapters.", ""): v for k, v in state.items() if k.startswith("dec_adapters.")}
        hparams["modules"]["dec_adapters"].load_state_dict(dec_sd, strict=True)


def save_adapter(hparams, adapter_dir: str):
    os.makedirs(adapter_dir, exist_ok=True)
    payload = {
        "state": extract_adapter_state(hparams),
        "meta": {
            "adapter_dim": int(hparams.get("adapter_dim", 64)),
            "adapter_dropout": float(hparams.get("adapter_dropout", 0.0)),
            "freeze_encoder": bool(hparams.get("freeze_encoder", False)),
        },
    }
    torch.save(payload, os.path.join(adapter_dir, "adapter.pt"))
    hparams["whisper"].tokenizer.save_pretrained(adapter_dir)


def load_adapter(hparams, adapter_dir: str):
    payload = torch.load(os.path.join(adapter_dir, "adapter.pt"), map_location="cpu")
    load_adapter_state(hparams, payload["state"])
    return payload.get("meta", {})


# -----------------------------
# SpeechBrain Brain
# -----------------------------
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos

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

        loss = self.hparams.ce_loss(logits.flatten(end_dim=-2), tokens_eos.flatten())

        if stage != sb.Stage.TRAIN:
            target_words = batch.target_wrd
            predicted_words = self.tokenizer.batch_decode(hyps, skip_special_tokens=True)
            if self.hparams.normalize_transcripts:
                predicted_words = [self.tokenizer._normalize(t).split(" ") for t in predicted_words]
            else:
                predicted_words = [t.split(" ") for t in predicted_words]
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
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(meta={"WER": stage_stats["WER"]}, min_keys=["WER"])
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)


# -----------------------------
# Data pipelines (same as train_ft)
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
        raise ValueError(f"`sorting` must be random/ascending/descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_folder"], "dev.csv"),
        replacements={"data_root": hparams["data_folder"]},
    ).filtered_sorted(sort_key="duration", reverse=True, key_max_value={"duration": hparams["avoid_if_longer_than"]})

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["data_folder"], "test.csv"),
        replacements={"data_root": hparams["data_folder"]},
    ).filtered_sorted(sort_key="duration", reverse=True, key_max_value={"duration": hparams["avoid_if_longer_than"]})

    datasets = [train_data, valid_data, test_data]

    @sb.utils.data_pipeline.takes("mp3")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(mp3):
        info = torchaudio.info(mp3)
        sig = sb.dataio.dataio.read_audio(mp3)
        resampled = torchaudio.transforms.Resample(info.sample_rate, hparams["sample_rate"])(sig)
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

        bos_index, tokens_list, eos_index = tokens_list[0], tokens_list[1:-1], tokens_list[-1]
        tokens_list = tokens_list[: hparams["max_target_length"] - 1]
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos

        if hparams["normalize_transcripts"]:
            wrd = tokenizer._normalize(wrd)
        wrd = wrd.split(" ")
        for i, ch in enumerate(wrd):
            if len(ch) == 0:
                wrd[i] = " "
        yield wrd

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "tokens_bos", "tokens_eos", "target_wrd"])
    return train_data, valid_data, test_data


# -----------------------------
# Test / Train
# -----------------------------
def test(hparams, run_opts, locales, wer_file="wer_test.txt"):
    for locale in locales:
        run_on_main(
            prepare_common_voice,
            kwargs={"locales": [locale], "data_folder": hparams["data_folder"], "max_durations": hparams["max_durations"]},
        )

        if locale in ["zh-CN", "ja"]:
            hparams["wer_computer"] = lambda *args, **kwargs: sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
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

        asr_brain.evaluate(test_data, min_key="WER", test_loader_kwargs=hparams["valid_dataloader_kwargs"])


def train(hparams, run_opts):
    adapters_root = os.path.join(hparams["output_folder"], "adapters")
    os.makedirs(adapters_root, exist_ok=True)

    per_locale_add_lang_token = bool(hparams.get("per_locale_add_lang_token", True))
    eval_base_before = bool(hparams.get("eval_base_before", False))

    # --- keep pristine base HF model template (no adapters, no peft) ---
    base_hf_template = copy.deepcopy(_get_hf_model_from_sb_whisper(hparams["whisper"])).cpu()

    if eval_base_before:
        test(hparams, run_opts, hparams["base_locales"], "wer_test_before.txt")

    hook_mgr = AdapterHookManager()

    for locale in hparams["new_locales"]:
        run_on_main(
            prepare_common_voice,
            kwargs={"locales": [locale], "data_folder": hparams["data_folder"], "max_durations": hparams["max_durations"]},
        )

        # 1) reset to pristine base model for this locale
        hf = copy.deepcopy(base_hf_template).to(run_opts["device"])
        _set_hf_model_into_sb_whisper(hparams["whisper"], hf)

        tokenizer = hparams["whisper"].tokenizer

        # 2) optional add language token + resize
        if per_locale_add_lang_token:
            new_tokens = [f"<|{locale.lower()}|>"]
            tokenizer._additional_special_tokens += new_tokens
            tokenizer.supported_languages.update({locale.lower(): locale.lower()})
            tokenizer.to_language_codes.update({locale.lower(): locale.lower()})
            new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
            tokenizer.add_tokens(new_tokens)
            hf.resize_token_embeddings(len(tokenizer))

        # 3) attach adapters + freeze/unfreeze
        make_fresh_adapters(hparams, locale, hook_mgr)

        # 4) datasets
        hparams["forced_decoder_locale"] = locale
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # 5) checkpoint folder
        checkpoint_folder = os.path.join(hparams["save_folder"], f"adapter_{locale}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_folder)

        # reset scheduler
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0
        hparams["epoch_counter"].current = 0
        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)

        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
        )
        asr_brain.tokenizer = tokenizer

        # 6) train
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # 7) save adapter weights
        adapter_dir = os.path.join(adapters_root, locale)
        save_adapter(hparams, adapter_dir)

        # 8) evaluate: rebuild pristine base + reattach adapters + load state
        hf = copy.deepcopy(base_hf_template).to(run_opts["device"])
        _set_hf_model_into_sb_whisper(hparams["whisper"], hf)

        if per_locale_add_lang_token:
            # tokenizer already includes token; just resize to current vocab
            hf.resize_token_embeddings(len(tokenizer))

        make_fresh_adapters(hparams, locale, hook_mgr)
        load_adapter(hparams, adapter_dir)

        test(hparams, run_opts, [locale], f"wer_test_after_{locale}.txt")

    hook_mgr.clear()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["train_logger"].save_file = hparams["train_logger"].save_file.replace(
        ".txt",
        f"_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}_adapter.txt",
    )

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    class CustomPaddedBatch(PaddedBatch):
        def __init__(self, examples, *args, **kwargs):
            for k in ["tokens_bos", "tokens_eos"]:
                max_len = max(len(x[k]) for x in examples)
                pad_value = 0.0
                if k == "tokens_bos":
                    pad_value = hparams["whisper"].tokenizer.pad_token_id
                elif k == "tokens_eos":
                    pad_value = hparams["ignore_index"]
                for ex in examples:
                    x = ex[k]
                    ex[k] = torch.nn.functional.pad(x, [0, max_len - len(x)], value=pad_value)
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch

    start = time.time()
    train(hparams, run_opts)
    logging.info(f"Time elapsed: {time.time() - start:.2f} seconds")
