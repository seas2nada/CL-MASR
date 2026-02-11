#!/usr/bin/env python3
"""
train_ia3.py

PEFT-IA3 continual fine-tuning recipe for a Whisper-based ASR system on Common Voice
(SpeechBrain CL_MASR style), with per-locale (task) IA3 adapters saved independently.

For each locale in hparams["new_locales"]:
  1) Prepare locale data
  2) Add language token (<|xx|>) + resize embeddings (optional)
  3) Snapshot BASE weights (after resize) for per-locale independence
  4) Attach a fresh IA3 adapter (PEFT) with freeze_encoder respected
  5) Train (only IA3 params trainable; optionally embeddings trainable)
  6) Save IA3 adapter to output_folder/ia3_adapters/<locale>/
  7) Restore BASE weights, load IA3 adapter, test on that locale

Usage:
> python train_ia3.py hparams/train_ft.yaml

Extra hparams you can add (optional):
  freeze_encoder: true|false                (default: false)
  train_embeddings: true|false              (default: true)
  per_locale_add_lang_token: true|false     (default: true)
  eval_base_before: true|false              (default: false)
  ia3_target_suffixes: ["k_proj","v_proj","fc1","fc2"]  (optional override)
  ia3_feedforward_suffixes: ["fc1","fc2"]              (optional override)

Requires:
  pip install peft

Notes:
- We use TaskType.FEATURE_EXTRACTION to avoid generation-method requirements on WhisperModel.
- target_modules and feedforward_modules are passed as FULL module names to guarantee
  "no adapter in encoder" when freeze_encoder=True.
"""

import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice

from peft import IA3Config, PeftModel, TaskType, get_peft_model

import copy

# -----------------------------
# SB Whisper wrapper helpers
# -----------------------------
def _get_hf_model_from_sb_whisper(sb_whisper):
    if hasattr(sb_whisper, "model"):
        return sb_whisper.model
    raise AttributeError(
        "Cannot locate HF model inside hparams['whisper']. Expected hparams['whisper'].model."
    )


def _set_hf_model_into_sb_whisper(sb_whisper, hf_model):
    if hasattr(sb_whisper, "model"):
        sb_whisper.model = hf_model
        return
    raise AttributeError(
        "Cannot set HF model into hparams['whisper']. Expected hparams['whisper'].model."
    )


def _unwrap_to_base(hf_model):
    if isinstance(hf_model, PeftModel):
        return hf_model.get_base_model()
    return hf_model


# -----------------------------
# Encoder/decoder name helpers
# -----------------------------
def _is_encoder_name(name: str) -> bool:
    return (
        name.startswith("encoder.")
        or name.startswith("model.encoder.")
        or ".encoder." in name
    )


def _is_decoder_name(name: str) -> bool:
    return (
        name.startswith("decoder.")
        or name.startswith("model.decoder.")
        or ".decoder." in name
    )


# -----------------------------
# Snapshot / restore (per-locale independence)
# -----------------------------
@torch.no_grad()
def snapshot_base_state(hparams) -> Dict[str, torch.Tensor]:
    """Snapshot current BASE HF model state_dict into CPU tensors."""
    hf = _unwrap_to_base(_get_hf_model_from_sb_whisper(hparams["whisper"]))
    return {k: v.detach().cpu().clone() for k, v in hf.state_dict().items()}


@torch.no_grad()
def restore_base_state(hparams, state: Dict[str, torch.Tensor]):
    """Restore BASE HF model weights from snapshot."""
    hf = _unwrap_to_base(_get_hf_model_from_sb_whisper(hparams["whisper"]))
    missing, unexpected = hf.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.warning(f"[Restore] missing={len(missing)} unexpected={len(unexpected)}")
    _set_hf_model_into_sb_whisper(hparams["whisper"], hf)


# -----------------------------
# IA3 target collection
# -----------------------------
def _collect_ia3_targets(
    base_model: torch.nn.Module,
    freeze_encoder: bool,
    target_suffixes: List[str],
    feedforward_suffixes: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Collect FULL module names for IA3:
      - target_modules: full names of Linear layers matching target_suffixes
      - feedforward_modules: subset of target_modules treated as FF layers (input scaling)
    """
    target_modules: List[str] = []
    feedforward_modules: List[str] = []

    for name, module in base_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if freeze_encoder and _is_encoder_name(name) and not _is_decoder_name(name):
            continue

        last = name.split(".")[-1]
        if last in target_suffixes:
            target_modules.append(name)
            if last in feedforward_suffixes:
                feedforward_modules.append(name)

    # feedforward_modules must be subset of target_modules
    ff_set = set(feedforward_modules)
    tgt_set = set(target_modules)
    feedforward_modules = [n for n in target_modules if n in ff_set]
    target_modules = list(dict.fromkeys(target_modules))  # stable unique
    return target_modules, feedforward_modules


# -----------------------------
# IA3 attach / save / load
# -----------------------------
def make_fresh_ia3_whisper(hparams, locale: str):
    """
    Attach a fresh IA3 adapter to BASE model.
    Trains only IA3 params (+ optionally embeddings).
    """
    sb_whisper = hparams["whisper"]
    hf = _get_hf_model_from_sb_whisper(sb_whisper)
    base = _unwrap_to_base(hf)
    _set_hf_model_into_sb_whisper(sb_whisper, base)

    freeze_encoder = bool(hparams.get("freeze_encoder", False))
    train_embeddings = bool(hparams.get("train_embeddings", True))

    # Default IA3 targets for Transformer blocks:
    # - attention: typically k_proj, v_proj (IA3 paper focuses on K/V scaling)
    # - feedforward: fc1, fc2
    target_suffixes = hparams.get("ia3_target_suffixes", None) or ["k_proj", "v_proj", "fc1", "fc2"]
    feedforward_suffixes = hparams.get("ia3_feedforward_suffixes", None) or ["fc1", "fc2"]

    target_modules, feedforward_modules = _collect_ia3_targets(
        base_model=base,
        freeze_encoder=freeze_encoder,
        target_suffixes=list(target_suffixes),
        feedforward_suffixes=list(feedforward_suffixes),
    )

    if len(target_modules) == 0:
        raise RuntimeError(
            "No IA3 target_modules were found. "
            "Check module names in your Whisper model, or set ia3_target_suffixes."
        )

    logging.info(
        f"[IA3] locale={locale} freeze_encoder={freeze_encoder} "
        f"#target={len(target_modules)} #ff={len(feedforward_modules)}"
    )

    ia3_cfg = IA3Config(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,             # FULL names
        feedforward_modules=feedforward_modules,   # FULL names subset
    )

    peft_model = get_peft_model(base, ia3_cfg)

    # Freeze everything first (PEFT usually does this, but we enforce)
    for _, p in peft_model.named_parameters():
        p.requires_grad = False

    # Unfreeze IA3 params
    for name, p in peft_model.named_parameters():
        # IA3 parameters typically have "ia3_" in their names in PEFT
        if "ia3" in name.lower():
            p.requires_grad = True

    # Optionally train embeddings (new language token)
    if train_embeddings:
        for name, p in peft_model.named_parameters():
            if any(k in name for k in ["embed_tokens", "decoder.embed_tokens"]):
                p.requires_grad = True

    # Log trainable params
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logging.info(f"[IA3] trainable={trainable:,}/{total:,} ({100*trainable/total:.6f}%)")

    _set_hf_model_into_sb_whisper(sb_whisper, peft_model)
    return peft_model


def save_ia3_adapter(hparams, adapter_dir: str):
    os.makedirs(adapter_dir, exist_ok=True)
    hf = _get_hf_model_from_sb_whisper(hparams["whisper"])
    if not isinstance(hf, PeftModel):
        raise RuntimeError("Expected PeftModel to save IA3 adapter.")
    hf.save_pretrained(adapter_dir)
    hparams["whisper"].tokenizer.save_pretrained(adapter_dir)


def load_ia3_adapter_into_whisper(hparams, adapter_dir: str):
    sb_whisper = hparams["whisper"]
    hf = _get_hf_model_from_sb_whisper(sb_whisper)
    base = _unwrap_to_base(hf)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    _set_hf_model_into_sb_whisper(sb_whisper, peft_model)
    return peft_model


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
# Test / Train
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

        if bool(hparams.get("skip_test", False)):
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


def train(hparams, run_opts):
    adapters_root = os.path.join(hparams["output_folder"], "ia3_adapters")
    os.makedirs(adapters_root, exist_ok=True)

    base_template = copy.deepcopy(hparams["whisper"].model).cpu()

    if bool(hparams.get("eval_base_before", False)):
        test(hparams, run_opts, hparams["base_locales"], "wer_test_before.txt")

    per_locale_add_lang_token = bool(hparams.get("per_locale_add_lang_token", True))

    for i, locale in enumerate(hparams["new_locales"]):

        hparams["whisper"].model = copy.deepcopy(base_template).to("cuda")

        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        tokenizer = hparams["whisper"].tokenizer

        # ---- tokenizer에 새 language token 추가 (원본 유지) ----
        new_tokens = [f"<|{locale.lower()}|>"]
        tokenizer = hparams["whisper"].tokenizer
        tokenizer._additional_special_tokens += new_tokens
        tokenizer.supported_languages.update({locale.lower(): locale.lower()})
        tokenizer.to_language_codes.update({locale.lower(): locale.lower()})

        new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
        tokenizer.add_tokens(new_tokens)

        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))

        # ---- snapshot base (after resize) for this locale ----
        # base_snapshot = snapshot_base_state(hparams)

        # ---- attach fresh IA3 adapter ----
        make_fresh_ia3_whisper(hparams, locale)

        # forced decoder locale
        hparams["forced_decoder_locale"] = locale

        # datasets
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # checkpoint folder (locale별)
        checkpoint_folder = os.path.join(hparams["save_folder"], f"ia3_{locale}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_folder)

        # reset scheduler/epoch counter per locale
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

        # ---- train ----
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # ---- save IA3 adapter ----
        adapter_dir = os.path.join(adapters_root, locale)
        save_ia3_adapter(hparams, adapter_dir)

        # ---- restore base, load adapter, test ----
        # restore_base_state(hparams, base_snapshot)
        # hparams["whisper"].model = copy.deepcopy(base_template).to("cuda")
        if isinstance(hparams["whisper"].model, PeftModel):
            hparams["whisper"].model = hparams["whisper"].model.get_base_model()
        load_ia3_adapter_into_whisper(hparams, adapter_dir)

        test(hparams, run_opts, [locale], f"wer_test_after_{locale}.txt")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # logger file suffix
    hparams["train_logger"].save_file = hparams["train_logger"].save_file.replace(
        ".txt",
        f"_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}_ia3.txt",
    )

    # experiment dir
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
