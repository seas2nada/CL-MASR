#!/usr/bin/env python3
"""
SLoRA-CSR: Shared + Language-Expert LoRA with Continual Stable Routing
---------------------------------------------------------------------
A continual language expansion recipe for Whisper (SpeechBrain CL_MASR style),
but implementing the *proposed* method:

  (1) Shared LoRA  (global, always-on)
  (2) Expert LoRA  (per-language, routed)
  (3) Router-stability regularizer: KL(r_{t-1}(x) || r_t(x))
  (4) Shared-drift regularizer: ||ΔW_s^t - ΔW_s^{t-1}||^2

Key design choices in this reference implementation:
- We do NOT rely on PEFT multi-adapter composition per-sample (PEFT isn't designed
  for per-sample dynamic weighted mixtures inside the same forward).
- Instead, we patch selected nn.Linear layers to MultiLoRALinear that supports:
    y = xW^T + LoRA_shared(x) + LoRA_expert(x, r(x))
- Routing is computed from encoder representations with a small LID router head.
- Rehearsal-free: we keep only *snapshots* of router + shared LoRA params from
  previous step for regularization, no previous audio is stored.

This is a "drop-in" recipe-level code. You still need the CL_MASR hparams yaml
(optimizer, dataloaders, whisper wrapper, etc).

Usage:
> python train_slora_csr.py hparams/train_ft.yaml

Authors
- Adapted by ChatGPT (2026)
- Based on SpeechBrain CL_MASR recipe structure
"""

import logging
import os
import pathlib
import sys
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice


# ============================================================
# 1) MultiLoRA modules (Shared + Experts, routed)
# ============================================================

@dataclass
class LoRAParams:
    A: nn.Parameter  # [r, in]
    B: nn.Parameter  # [out, r]


class MultiLoRALinear(nn.Module):
    """
    Wraps a frozen base Linear and adds:
      - shared LoRA: (A_s, B_s)
      - expert LoRA bank: {lang_key: (A_k, B_k)}
    Forward supports soft routing weights r(x) over experts:
      y = xW^T + x A_s^T B_s^T + sum_k r_k(x) * x A_k^T B_k^T
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        init_scale: float = 0.01,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base = base

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.drop = nn.Dropout(p=float(dropout))

        # shared LoRA
        self.shared = self._make_lora(init_scale=init_scale)

        # expert bank: lang -> LoRAParams
        self.experts: nn.ModuleDict = nn.ModuleDict()

    def _make_lora(self, init_scale: float) -> nn.Module:
        mod = nn.Module()
        A = nn.Parameter(init_scale * torch.randn(self.r, self.in_features))
        B = nn.Parameter(init_scale * torch.randn(self.out_features, self.r))
        mod.register_parameter("A", A)
        mod.register_parameter("B", B)
        return mod

    @torch.no_grad()
    def add_expert(self, lang_key: str, init_from_shared: bool = False):
        if lang_key in self.experts:
            return
        expert = self._make_lora(init_scale=0.01)
        if init_from_shared:
            expert.A.copy_(self.shared.A)
            expert.B.copy_(self.shared.B)
        self.experts[lang_key] = expert

    def set_trainable(self, train_shared: bool, train_experts: List[str]):
        # shared params
        self.shared.A.requires_grad = bool(train_shared)
        self.shared.B.requires_grad = bool(train_shared)
        # experts
        for k, expert in self.experts.items():
            req = (k in set(train_experts))
            expert.A.requires_grad = req
            expert.B.requires_grad = req

    def forward(
        self,
        x: torch.Tensor,
        route_w: Optional[torch.Tensor] = None,
        expert_keys: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, in] or [N, in]
        route_w: [B, K] routing weights over expert_keys (soft)
        expert_keys: list of expert names length K (must align with route_w)
        """
        y = self.base(x)

        # shared LoRA
        xs = self.drop(x)
        # (x @ A^T) -> [..., r], then @ B^T -> [..., out]
        y = y + (xs @ self.shared.A.t() @ self.shared.B.t()) * self.scaling

        # expert LoRA mixture
        if route_w is not None and expert_keys is not None and len(expert_keys) > 0:
            # We'll compute per-expert update and weight by route_w (soft mixture).
            # route_w is per-utterance (B, K). We broadcast across time.
            # x shape can be [B, T, in] or [B, in]. We handle both.
            if x.dim() == 3:
                B, T, _ = x.shape
                w = route_w[:, :, None, None]  # [B, K, 1, 1]
                # stack expert outputs: [B, K, T, out]
                outs = []
                for k in expert_keys:
                    expert = self.experts[k]
                    outs.append((xs @ expert.A.t() @ expert.B.t()) * self.scaling)
                e = torch.stack(outs, dim=1)  # [B, K, T, out]
                y = y + (w * e).sum(dim=1)
            elif x.dim() == 2:
                B, _ = x.shape
                w = route_w[:, :, None]  # [B, K, 1]
                outs = []
                for k in expert_keys:
                    expert = self.experts[k]
                    outs.append((xs @ expert.A.t() @ expert.B.t()) * self.scaling)
                e = torch.stack(outs, dim=1)  # [B, K, out]
                y = y + (w * e).sum(dim=1)
            else:
                raise ValueError(f"Unsupported x.dim={x.dim()} for MultiLoRALinear")

        return y


def patch_whisper_with_multilora(
    hf_model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    freeze_encoder: bool,
    allow_suffix: Optional[set] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Replace selected nn.Linear modules with MultiLoRALinear in-place.

    Returns:
      - patched model
      - list of patched module fullnames
    """
    if allow_suffix is None:
        allow_suffix = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj_out"}

    patched = []

    def _is_encoder(name: str) -> bool:
        return name.startswith("encoder.") or name.startswith("model.encoder.") or ".encoder." in name

    def _is_decoder(name: str) -> bool:
        return name.startswith("decoder.") or name.startswith("model.decoder.") or ".decoder." in name

    # We'll traverse parent modules so we can setattr on them
    for full_name, module in list(hf_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        last = full_name.split(".")[-1]
        if last not in allow_suffix:
            continue

        if freeze_encoder and _is_encoder(full_name) and not _is_decoder(full_name):
            continue

        # locate parent
        parent = hf_model
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        # replace
        base_linear = getattr(parent, child_name)
        if isinstance(base_linear, MultiLoRALinear):
            continue

        setattr(parent, child_name, MultiLoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout))
        patched.append(full_name)

    return hf_model, patched


# ============================================================
# 2) Router (LID head) + continual stability regularization
# ============================================================

class SimpleLIDRouter(nn.Module):
    """
    Given encoder hidden states, output routing logits over current expert set.
    We'll do:
      pooled = mean over time (masked)
      logits = MLP(pooled)
    """
    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # placeholder; will be reset per step
        )

    def reset_out(self, num_langs: int):
        # Replace last layer to match current #experts
        in_dim = self.net[-1].in_features
        self.net[-1] = nn.Linear(in_dim, num_langs)

    def forward(self, enc: torch.Tensor, enc_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        enc: [B, T, D]
        enc_lens: [B] in [0,1] or absolute lengths. We'll assume relative lens in [0,1]
                  as SpeechBrain usually provides.
        """
        B, T, D = enc.shape
        if enc_lens is None:
            pooled = enc.mean(dim=1)
        else:
            # enc_lens is relative (0..1). convert to lengths
            lengths = torch.clamp((enc_lens * T).long(), min=1, max=T)  # [B]
            mask = torch.arange(T, device=enc.device)[None, :] < lengths[:, None]  # [B,T]
            mask = mask.unsqueeze(-1).float()
            pooled = (enc * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.net(pooled)  # [B, K]


@torch.no_grad()
def snapshot_shared_params(hf_model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Store a CPU snapshot of shared LoRA params only (for shared drift penalty).
    """
    snap = {}
    for name, m in hf_model.named_modules():
        if isinstance(m, MultiLoRALinear):
            snap[f"{name}.shared.A"] = m.shared.A.detach().cpu().clone()
            snap[f"{name}.shared.B"] = m.shared.B.detach().cpu().clone()
    return snap


def shared_drift_penalty(hf_model: nn.Module, prev_snap: Dict[str, torch.Tensor], device) -> torch.Tensor:
    if prev_snap is None or len(prev_snap) == 0:
        return torch.zeros([], device=device)
    loss = torch.zeros([], device=device)
    for name, m in hf_model.named_modules():
        if isinstance(m, MultiLoRALinear):
            kA = f"{name}.shared.A"
            kB = f"{name}.shared.B"
            if kA in prev_snap:
                loss = loss + F.mse_loss(m.shared.A, prev_snap[kA].to(device))
            if kB in prev_snap:
                loss = loss + F.mse_loss(m.shared.B, prev_snap[kB].to(device))
    return loss


# ============================================================
# 3) SpeechBrain Brain (ASR + Router regularizers)
# ============================================================

class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos

        # forward whisper wrapper:
        # Expect whisper wrapper returns (enc_out, logits, ...)
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        # routing weights computed from encoder output
        route_logits = self.modules.router(enc_out, wav_lens)  # [B, K]
        route_w = F.softmax(route_logits / self.hparams.route_tau, dim=-1)  # [B, K]

        hyps = None
        if stage != sb.Stage.TRAIN:
            hyps, _ = self.modules.whisper.generate(
                audio_features=enc_out,
                forced_decoder_locale=self.hparams.forced_decoder_locale,
                max_gen_tokens=self.hparams.max_gen_tokens,
            )

        return logits, hyps, enc_out, wav_lens, route_w

    def compute_objectives(self, predictions, batch, stage):
        logits, hyps, enc_out, wav_lens, route_w = predictions
        ids = batch.id
        tokens_eos, _ = batch.tokens_eos

        # Base ASR CE loss
        loss_asr = self.hparams.ce_loss(logits.flatten(end_dim=-2), tokens_eos.flatten())
        loss = loss_asr

        # ===== Continual regularizers (TRAIN only) =====
        if stage == sb.Stage.TRAIN:
            # (1) Router stability KL(r_{t-1} || r_t)
            loss_route = torch.zeros([], device=self.device)
            if getattr(self.hparams, "router_prev", None) is not None and self.hparams.lambda_route > 0:
                with torch.no_grad():
                    prev_logits = self.hparams.router_prev(enc_out.detach(), wav_lens)  # [B,K]
                    prev_w = F.softmax(prev_logits / self.hparams.route_tau, dim=-1)
                # KL(prev || cur) = sum prev * (log prev - log cur)
                loss_route = torch.sum(prev_w * (torch.log(prev_w + 1e-8) - torch.log(route_w + 1e-8)), dim=-1).mean()
                loss = loss + self.hparams.lambda_route * loss_route

            # (2) Shared drift control ||ΔW_s^t - ΔW_s^{t-1}||^2
            loss_shared = torch.zeros([], device=self.device)
            if getattr(self.hparams, "shared_prev_snap", None) is not None and self.hparams.lambda_shared > 0:
                hf = self.modules.whisper.model  # HF model inside wrapper
                loss_shared = shared_drift_penalty(hf, self.hparams.shared_prev_snap, self.device)
                loss = loss + self.hparams.lambda_shared * loss_shared

            # log scalars (optional)
            if hasattr(self, "train_logger") and self.train_logger is not None:
                # SpeechBrain logger will read from stage_stats at epoch end; we store in self
                self.last_reg = {
                    "loss_asr": float(loss_asr.detach().cpu()),
                    "loss_route": float(loss_route.detach().cpu()),
                    "loss_shared": float(loss_shared.detach().cpu()),
                }

        # ===== Metrics for VALID/TEST =====
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
            # include reg breakdown if available
            if hasattr(self, "last_reg"):
                stage_stats.update({k: v for k, v in self.last_reg.items()})
            self.train_stats = stage_stats
            return

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
            self.checkpointer.save_and_keep_only(meta={"WER": stage_stats["WER"]}, min_keys=["WER"])

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)


# ============================================================
# 4) Data pipelines (same as your baseline)
# ============================================================

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
        raise ValueError(f"`sorting` ({hparams['sorting']}) must be random, ascending or descending")

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
        for i, char in enumerate(wrd):
            if len(char) == 0:
                wrd[i] = " "
        yield wrd

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "tokens_bos", "tokens_eos", "target_wrd"])
    return train_data, valid_data, test_data


# ============================================================
# 5) Train / Test
# ============================================================

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
    """
    Continual loop:
      - Initialize Whisper
      - Patch with MultiLoRA (shared + expert)
      - Maintain router and snapshots for stability regs
      - For each new locale:
          - add expert params for that locale (trainable)
          - optionally allow shared LoRA to update with drift penalty
          - train on that locale
          - snapshot router/shared for next step
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # (A) Prepare base model
    # ----------------------------
    sb_whisper = hparams["whisper"]
    hf = sb_whisper.model

    freeze_encoder = bool(hparams.get("freeze_encoder", False))
    lora_r = int(hparams.get("lora_r", 16))
    lora_alpha = float(hparams.get("lora_alpha", 32))
    lora_dropout = float(hparams.get("lora_dropout", 0.05))

    # Patch linear layers
    hf, patched = patch_whisper_with_multilora(
        hf,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        freeze_encoder=freeze_encoder,
    )
    logging.info(f"[SLoRA-CSR] patched {len(patched)} Linear modules")

    # Put back
    sb_whisper.model = hf.to(device)

    # Router init (need encoder hidden size)
    d_model = sb_whisper.model.config.d_model
    router = SimpleLIDRouter(d_model=d_model, hidden=int(hparams.get("router_hidden", 256))).to(device)

    # register router into modules so Brain can access
    hparams["modules"]["router"] = router

    # Continual regularization state
    hparams["router_prev"] = None
    hparams["shared_prev_snap"] = None

    # hyperparams
    hparams["lambda_route"] = float(hparams.get("lambda_route", 1.0))
    hparams["lambda_shared"] = float(hparams.get("lambda_shared", 1.0))
    hparams["route_tau"] = float(hparams.get("route_tau", 1.0))
    hparams["train_shared"] = bool(hparams.get("train_shared", True))

    # ----------------------------
    # (B) Continual language expansion
    # ----------------------------
    for step, locale in enumerate(hparams["new_locales"]):
        locale_l = locale.lower()
        logging.info(f"\n========== [CLE step {step+1}/{len(hparams['new_locales'])}] locale={locale} ==========")

        # Prepare data for this locale
        run_on_main(
            prepare_common_voice,
            kwargs={"locales": [locale], "data_folder": hparams["data_folder"], "max_durations": hparams["max_durations"]},
        )

        # Tokenizer: add locale token (same as your baseline logic)
        tokenizer = hparams["whisper"].tokenizer
        new_tokens = [f"<|{locale_l}|>"]
        tokenizer._additional_special_tokens += new_tokens
        tokenizer.supported_languages.update({locale_l: locale_l})
        tokenizer.to_language_codes.update({locale_l: locale_l})
        new_tokens = sorted(list(set(new_tokens) - set(tokenizer.get_vocab().keys())))
        tokenizer.add_tokens(new_tokens)
        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))

        # Add expert to every MultiLoRALinear
        for _, m in hparams["whisper"].model.named_modules():
            if isinstance(m, MultiLoRALinear):
                m.add_expert(locale_l, init_from_shared=bool(hparams.get("init_expert_from_shared", True)))

        # Router output dimension = #experts so far
        expert_keys = sorted({k for _, m in hparams["whisper"].model.named_modules() if isinstance(m, MultiLoRALinear) for k in m.experts.keys()})
        router.reset_out(num_langs=len(expert_keys))
        hparams["expert_keys"] = expert_keys  # for debugging/logging if needed

        # Decide trainable params:
        # - Train new expert only (plasticity)
        # - Optionally train shared LoRA (with drift penalty)
        # - Train router (so it learns to map enc reps to the correct expert)
        train_shared = bool(hparams["train_shared"])
        train_experts = [locale_l]  # only new
        for _, m in hparams["whisper"].model.named_modules():
            if isinstance(m, MultiLoRALinear):
                m.set_trainable(train_shared=train_shared, train_experts=train_experts)

        # Freeze everything else (safety)
        for n, p in hparams["whisper"].model.named_parameters():
            # MultiLoRALinear handles its own flags; we just ensure base remains frozen
            if "base." in n:
                p.requires_grad = False

        # Router trainable
        for p in router.parameters():
            p.requires_grad = True

        # SpeechBrain expects optimizer built from asr_brain opt_class and parameters
        # Make sure only trainable params are seen
        def _trainable_params():
            for p in hparams["whisper"].model.parameters():
                if p.requires_grad:
                    yield p
            for p in router.parameters():
                if p.requires_grad:
                    yield p

        # Dataset
        hparams["forced_decoder_locale"] = locale
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # Checkpoint folder per locale
        checkpoint_folder = os.path.join(hparams["save_folder"], f"slora_csr_{locale}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_folder)

        # Reset scheduler & epoch
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0
        hparams["epoch_counter"].current = 0

        # Build brain
        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
        )
        asr_brain.tokenizer = tokenizer

        # IMPORTANT: override optimizer creation to use only trainable params
        asr_brain.optimizer = hparams["opt_class"](_trainable_params())

        # Train
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # Snapshot router + shared LoRA for next step (rehearsal-free anchors)
        hparams["router_prev"] = copy.deepcopy(router).eval().to(device)
        for p in hparams["router_prev"].parameters():
            p.requires_grad = False
        hparams["shared_prev_snap"] = snapshot_shared_params(hparams["whisper"].model)

        # Evaluate the newly learned locale
        test(hparams, run_opts, [locale], wer_file=f"wer_test_after_{locale}.txt")


# ============================================================
# 6) Main
# ============================================================

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["train_logger"].save_file = hparams["train_logger"].save_file.replace(
        ".txt",
        f"_slora_csr_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}.txt",
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
                    example[k] = torch.nn.functional.pad(x, [0, max_len - len(x)], value=pad_value)
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch

    start_time = time.time()
    train(hparams, run_opts)
    logging.info(f"Time elapsed: {time.time() - start_time:.1f} seconds")