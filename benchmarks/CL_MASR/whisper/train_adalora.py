#!/usr/bin/env python3
"""
train_adalora.py

AdaLoRA continual fine-tuning recipe for Whisper ASR on Common Voice
(SpeechBrain CL_MASR style).

- Requires `peft`: pip install peft
- Per-locale independent AdaLoRA adapters are saved/loaded.
- Uses pristine HF model deepcopy per locale to avoid in-place PEFT residue.

Extra hparams (optional):
  freeze_encoder: true|false (default: false)
  train_embeddings: true|false (default: false)   # WARNING: opens full embedding matrix
  per_locale_add_lang_token: true|false (default: true)
  eval_base_before: true|false (default: false)

AdaLoRA hparams (optional):
  adalora_r: 32
  adalora_init_r: 32
  adalora_tinit: 200
  adalora_tfinal: 1000
  adalora_deltaT: 10
  adalora_beta1: 0.85
  adalora_beta2: 0.85
  adalora_lora_alpha: 64
  adalora_lora_dropout: 0.05
  adalora_target_modules: [q_proj, k_proj, v_proj, out_proj, fc1, fc2]

Usage:
> python train_adalora.py hparams/train_ft.yaml
"""

import math
import copy
import logging
import os
import pathlib
import sys
import time
import types
from typing import List

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice

from peft import AdaLoraConfig, get_peft_model, PeftModel, TaskType


def _get_hf_model_from_sb_whisper(sb_whisper):
    if hasattr(sb_whisper, "model"):
        return sb_whisper.model
    raise AttributeError("Expected hparams['whisper'].model to exist.")


def _set_hf_model_into_sb_whisper(sb_whisper, hf_model):
    if hasattr(sb_whisper, "model"):
        sb_whisper.model = hf_model
        return
    raise AttributeError("Expected hparams['whisper'].model to exist.")


def _ensure_prepare_inputs_for_generation(hf_model):
    """
    Some PEFT versions expect base_model.prepare_inputs_for_generation to exist.
    WhisperModel may not have it; provide a minimal stub.
    """
    if hasattr(hf_model, "prepare_inputs_for_generation"):
        return

    def _pifg(self, input_ids=None, **kwargs):
        # minimal: just pass through
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        return kwargs

    hf_model.prepare_inputs_for_generation = types.MethodType(_pifg, hf_model)


def _freeze_all_then_unfreeze_adalora(hf, train_embeddings: bool, freeze_encoder: bool):
    # Freeze everything
    for _, p in hf.named_parameters():
        p.requires_grad = False

    # Unfreeze PEFT/AdaLoRA params
    for name, p in hf.named_parameters():
        if "lora_" in name.lower() or "adalora" in name.lower():
            p.requires_grad = True

    # Optionally block encoder LoRA params
    if freeze_encoder:
        for name, p in hf.named_parameters():
            if (".encoder." in name or name.startswith("encoder.") or name.startswith("model.encoder.")) and ("lora_" in name.lower()):
                p.requires_grad = False

    # Optionally train embeddings (주의: 전체 embedding)
    if train_embeddings:
        for name, p in hf.named_parameters():
            if any(k in name for k in ["embed_tokens", "decoder.embed_tokens"]):
                p.requires_grad = True


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
        raise ValueError("`sorting` must be random/ascending/descending")

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
    adapters_root = os.path.join(hparams["output_folder"], "adalora_adapters")
    os.makedirs(adapters_root, exist_ok=True)

    per_locale_add_lang_token = bool(hparams.get("per_locale_add_lang_token", True))
    eval_base_before = bool(hparams.get("eval_base_before", False))

    freeze_encoder = bool(hparams.get("freeze_encoder", False))
    train_embeddings = bool(hparams.get("train_embeddings", False))

    # AdaLoRA params
    r = int(hparams.get("adalora_r", 32))
    init_r = int(hparams.get("adalora_init_r", r))
    tinit = int(hparams.get("adalora_tinit", 200))
    tfinal = int(hparams.get("adalora_tfinal", 1000))
    deltaT = int(hparams.get("adalora_deltaT", 10))
    beta1 = float(hparams.get("adalora_beta1", 0.85))
    beta2 = float(hparams.get("adalora_beta2", 0.85))
    lora_alpha = int(hparams.get("adalora_lora_alpha", 64))
    lora_dropout = float(hparams.get("adalora_lora_dropout", 0.05))

    target_modules = hparams.get(
        "adalora_target_modules",
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )

    # pristine base HF template (no PEFT)
    base_hf_template = copy.deepcopy(_get_hf_model_from_sb_whisper(hparams["whisper"])).cpu()

    if eval_base_before:
        test(hparams, run_opts, hparams["base_locales"], "wer_test_before.txt")

    for locale in hparams["new_locales"]:
        run_on_main(
            prepare_common_voice,
            kwargs={"locales": [locale], "data_folder": hparams["data_folder"], "max_durations": hparams["max_durations"]},
        )

        # 1) reset to pristine base model for this locale
        base_hf = copy.deepcopy(base_hf_template).to(run_opts["device"])
        _ensure_prepare_inputs_for_generation(base_hf)
        _set_hf_model_into_sb_whisper(hparams["whisper"], base_hf)

        tokenizer = hparams["whisper"].tokenizer

        # 2) optional add language token + resize
        if per_locale_add_lang_token:
            new_tokens = [f"<|{locale.lower()}|>"]
            tokenizer._additional_special_tokens += new_tokens
            tokenizer.supported_languages.update({locale.lower(): locale.lower()})
            tokenizer.to_language_codes.update({locale.lower(): locale.lower()})
            new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
            tokenizer.add_tokens(new_tokens)
            base_hf.resize_token_embeddings(len(tokenizer))

        # 5) datasets
        hparams["forced_decoder_locale"] = locale
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # ---- compute total_step for AdaLoRA ----
        train_loader = sb.dataio.dataloader.make_dataloader(
            train_data,
            **hparams["train_dataloader_kwargs"],
        )
        steps_per_epoch = len(train_loader)

        # epoch 수 (SpeechBrain epoch_counter 기준)
        # 보통 hparams["epoch_counter"]는 sb.utils.epoch_loop.EpochCounter
        if hasattr(hparams["epoch_counter"], "limit") and hparams["epoch_counter"].limit is not None:
            num_epochs = int(hparams["epoch_counter"].limit)
        else:
            # fallback (너 yaml에 맞는 키로 하나 잡아두면 됨)
            num_epochs = int(hparams.get("num_epochs", 1))

        # grad accumulation (SB에서 흔한 키들)
        grad_accum = int(
            hparams.get("gradient_accumulation_factor",
            hparams.get("grad_accumulation_factor",
            hparams.get("grad_accum", 1)))
        )
        total_step = math.ceil(steps_per_epoch * num_epochs / max(1, grad_accum))

        # 3) attach AdaLoRA
        tinit_eff = min(tinit, max(1, total_step // 10))
        tfinal_eff = min(tfinal, max(tinit_eff + 1, total_step - 1))
        cfg = AdaLoraConfig(
            r=r,
            init_r=init_r,
            tinit=tinit_eff,
            tfinal=tfinal_eff,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            total_step=total_step,
        )
        peft_hf = get_peft_model(base_hf, cfg)

        # 4) freeze/unfreeze
        _freeze_all_then_unfreeze_adalora(peft_hf, train_embeddings=train_embeddings, freeze_encoder=freeze_encoder)

        _set_hf_model_into_sb_whisper(hparams["whisper"], peft_hf)

        # 6) checkpoint folder
        checkpoint_folder = os.path.join(hparams["save_folder"], f"adalora_{locale}")
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

        # 7) train
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # 8) save adapter
        adapter_dir = os.path.join(adapters_root, locale)
        os.makedirs(adapter_dir, exist_ok=True)
        peft_hf.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # 9) evaluate: rebuild pristine base + load adapter
        base_hf2 = copy.deepcopy(base_hf_template).to(run_opts["device"])
        _ensure_prepare_inputs_for_generation(base_hf2)
        _set_hf_model_into_sb_whisper(hparams["whisper"], base_hf2)

        if per_locale_add_lang_token:
            base_hf2.resize_token_embeddings(len(tokenizer))

        peft_loaded = PeftModel.from_pretrained(base_hf2, adapter_dir)
        _set_hf_model_into_sb_whisper(hparams["whisper"], peft_loaded)

        test(hparams, run_opts, [locale], f"wer_test_after_{locale}.txt")


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["train_logger"].save_file = hparams["train_logger"].save_file.replace(
        ".txt",
        f"_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}_adalora.txt",
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
