#!/usr/bin/env python3

"""Recipe for fine-tuning a Whisper-based ASR system on Common Voice in a continual
learning fashion, with Language-Aware Gradient Projection (LAGP) that computes
language similarity from Whisper encoder representations.

To run this recipe, do the following:
> python train_lagp_whispervec.py hparams/train_ft.yaml

Authors
 * Luca Della Libera 2023 (original)
 * Modified with LAGP using Whisper features by ChatGPT 2026
"""

import logging
import os
import pathlib
import sys
import time
import math
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main

from common_voice_prepare import prepare_common_voice


# ----------------------------------------------------------------------
# Language-Aware Gradient Projector (LAGP) with Whisper-based language vectors
# ----------------------------------------------------------------------
class LanguageAwareGradientProjector:
    """
    Stores projected gradients per language and projects the current gradient
    away from the subspace spanned by previous languages' gradients,
    weighted by language similarity derived from Whisper encoder representations.
    """

    def __init__(
        self,
        model,
        proj_dim=256,
        buffer_size=50,
        subspace_rank=10,
        beta=0.5,
        device="cuda",
    ):
        """
        Args:
            model: Whisper model (used to get encoder and total parameter count)
            proj_dim: dimensionality for random projection
            buffer_size: number of gradient vectors to store per language
            subspace_rank: number of principal components to keep
            beta: projection strength (0 = no projection, 1 = hard orthogonal)
            device: device for computations
        """
        self.model = model
        self.proj_dim = proj_dim
        self.buffer_size = buffer_size
        self.subspace_rank = subspace_rank
        self.beta = beta
        self.device = device

        # Calculate total number of trainable parameters
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[LAGP] Total trainable parameters: {self.total_params}")

        # Random projection matrix (lazy initialization)
        self.R = None

        # Buffers: language -> list of projected gradient vectors (as CPU tensors)
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))

        # Language vectors: averaged encoder representations (from validation data)
        self.lang_vectors = {}  # language code -> tensor

        # Similarity cache
        self.sim_cache = {}

    def _get_projection_matrix(self):
        """Lazily create the random projection matrix (kept on CPU)."""
        if self.R is None:
            logging.info(f"[LAGP] Creating random projection matrix of size {self.total_params} x {self.proj_dim}")
            self.R = torch.randn(self.total_params, self.proj_dim, device="cpu") / math.sqrt(self.proj_dim)
        return self.R

    def _flatten_grads(self):
        """Return a flat vector of all gradients (for trainable parameters)."""
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            return None
        return torch.cat(grads)

    def _unflatten_grads(self, flat_grad):
        """Assign the flat gradient back to model parameters."""
        idx = 0
        for p in self.model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.grad = flat_grad[idx : idx + numel].view(p.shape).to(p.device)
                idx += numel

    def compute_language_vector(self, lang, dataloader, num_samples=100):
        """
        Compute an averaged encoder representation for a language using a dataloader.
        This vector will be used to compute language similarities.
        """
        self.model.eval()
        vecs = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                batch = batch.to(self.device)
                wavs, _ = batch.sig
                # Get encoder output (mean over time)
                encoder_outputs = self.model.get_encoder()(wavs)  # [batch, time, dim]
                pooled = encoder_outputs.mean(dim=1)  # [batch, dim]
                vecs.append(pooled.cpu())
        if not vecs:
            raise RuntimeError(f"No data to compute language vector for {lang}")
        avg_vec = torch.cat(vecs, dim=0).mean(dim=0)  # [dim]
        self.lang_vectors[lang] = avg_vec
        logging.info(f"[LAGP] Computed language vector for {lang}, shape {avg_vec.shape}")
        return avg_vec

    def get_similarity(self, lang1, lang2):
        """Return cosine similarity between two language vectors."""
        if lang1 == lang2:
            return 1.0
        key = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
        if key in self.sim_cache:
            return self.sim_cache[key]
        if lang1 not in self.lang_vectors or lang2 not in self.lang_vectors:
            return 0.0
        v1 = self.lang_vectors[lang1].to(self.device)
        v2 = self.lang_vectors[lang2].to(self.device)
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        self.sim_cache[key] = sim
        return sim

    def project_gradient(self, current_lang):
        """
        Project the current gradients (already computed in model) and modify them in-place.
        """
        # Flatten current gradients
        flat_grad = self._flatten_grads()
        if flat_grad is None:
            return

        # Project to low-dimensional space
        R = self._get_projection_matrix().to(flat_grad.device)
        grad_proj = torch.mv(R.t(), flat_grad)  # [proj_dim]

        # Collect all stored gradients from previous languages, weighted by similarity
        all_proj_grads = []
        weights = []
        for lang, buf in self.buffers.items():
            if lang == current_lang:
                continue
            sim = self.get_similarity(lang, current_lang)
            if sim <= 0:
                continue
            for g in buf:
                all_proj_grads.append(g.to(grad_proj.device))
                weights.append(sim)

        if not all_proj_grads:
            return  # no previous languages to project against

        # Stack into matrix G of shape (N, proj_dim)
        G = torch.stack(all_proj_grads)  # [N, proj_dim]
        W = torch.tensor(weights, device=grad_proj.device).unsqueeze(1)  # [N, 1]
        G_weighted = W * G

        # Compute top-k right singular vectors of G_weighted
        try:
            U, S, Vh = torch.linalg.svd(G_weighted, full_matrices=False)
            # Vh shape: [min(N, proj_dim), proj_dim]
            k = min(self.subspace_rank, G_weighted.shape[0])
            V = Vh[:k, :]  # [k, proj_dim]
        except:
            # If SVD fails (e.g., due to NaN), skip projection
            return

        # Project grad_proj onto the subspace spanned by V
        coeff = V @ grad_proj  # [k]
        proj_component = V.t() @ coeff  # [proj_dim]

        # Subtract a fraction beta of the projection
        grad_proj_clean = grad_proj - self.beta * proj_component

        # Map back to full gradient space
        flat_grad_clean = R @ grad_proj_clean

        # Unflatten and assign
        self._unflatten_grads(flat_grad_clean)

    def add_to_buffer(self, lang, grad_flat=None):
        """
        Store a projected gradient for language `lang`.
        If grad_flat is None, use current model gradients.
        """
        if grad_flat is None:
            grad_flat = self._flatten_grads()
            if grad_flat is None:
                return
        R = self._get_projection_matrix().to(grad_flat.device)
        proj_g = torch.mv(R.t(), grad_flat).detach().cpu()
        self.buffers[lang].append(proj_g)


# ----------------------------------------------------------------------
# ASR Brain (modified to use LAGP)
# ----------------------------------------------------------------------
class ASR(sb.Brain):
    def __init__(self, *args, lagp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lagp = lagp  # LAGP projector instance

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, _ = batch.tokens_bos

        # Forward encoder + decoder
        if self.hparams.gradient_checkpointing:
            wavs.requires_grad_()
            enc_out, logits, _ = torch.utils.checkpoint.checkpoint(
                self.modules.whisper, wavs, bos_tokens,
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
        """Computes the loss given predictions and targets."""
        logits, hyps = predictions
        ids = batch.id
        tokens_eos, _ = batch.tokens_eos

        loss = self.hparams.ce_loss(
            logits.flatten(end_dim=-2), tokens_eos.flatten()
        )

        if stage != sb.Stage.TRAIN:
            target_words = batch.target_wrd

            # Decode predicted tokens to words
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

    def fit_batch(self, batch):
        """Override fit_batch to apply LAGP after backward."""
        # Standard forward-backward
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        loss.backward()

        # Apply gradient projection if LAGP is enabled and we are training a new language
        if self.lagp is not None and hasattr(self, "current_lang"):
            self.lagp.project_gradient(self.current_lang)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            stats_meta_data = {
                "epoch": epoch,
                "lr": old_lr,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta_data,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)


# ----------------------------------------------------------------------
# Data preparation (unchanged)
# ----------------------------------------------------------------------
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
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
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wrd", "locale")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "target_wrd")
    def text_pipeline(wrd, locale):
        if locale.startswith("zh"):
            locale = "zh"
        locale = locale.lower()
        language = tokenizer.supported_languages.get(
            locale, "english"
        )  # Use English if unknown
        tokenizer.set_prefix_tokens(language=language)
        tokens_list = tokenizer.encode(wrd)
        assert sum(i == tokenizer.unk_token_id for i in tokens_list) == 1
        # Remove BOS and EOS tokens from tokens_list
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
        # When `ref_tokens` is an empty string add dummy space
        # to avoid division by 0 when computing WER/CER
        for i, char in enumerate(wrd):
            if len(char) == 0:
                wrd[i] = " "
        yield wrd

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "tokens_bos", "tokens_eos", "target_wrd"],
    )

    return train_data, valid_data, test_data


# ----------------------------------------------------------------------
# Testing function (unchanged)
# ----------------------------------------------------------------------
def test(hparams, run_opts, locales, wer_file="wer_test.txt"):
    """Test incrementally on the given locales."""
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
            hparams[
                "wer_computer"
            ] = lambda *args, **kwargs: sb.utils.metric_stats.ErrorRateStats(
                split_tokens=True
            )
        else:
            hparams["wer_computer"] = sb.utils.metric_stats.ErrorRateStats

        hparams["forced_decoder_locale"] = locale
        tokenizer = hparams["whisper"].tokenizer
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        asr_brain = ASR(
            modules=hparams["modules"], hparams=hparams, run_opts=run_opts,
        )
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
                "Install ptflops and torchinfo to profile the model (e.g. `pip install ptflops torchinfo`)"
            )


# ----------------------------------------------------------------------
# Main training function with LAGP using Whisper features
# ----------------------------------------------------------------------
def train(hparams, run_opts):
    """Train incrementally on the new locales, using LAGP with Whisper-based language vectors."""
    # Testing before any new language
    test(
        hparams, run_opts, hparams["base_locales"], "wer_test_before.txt",
    )

    # Initialize LAGP projector (no external language vectors needed)
    lagp = LanguageAwareGradientProjector(
        model=hparams["whisper"].model,
        proj_dim=hparams.get("lagp_proj_dim", 256),
        buffer_size=hparams.get("lagp_buffer_size", 50),
        subspace_rank=hparams.get("lagp_subspace_rank", 10),
        beta=hparams.get("lagp_beta", 0.5),
        device=run_opts.get("device", "cuda"),
    )

    # Pre-compute language vectors for base locales using validation data
    tokenizer = hparams["whisper"].tokenizer
    for base_loc in hparams["base_locales"]:
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [base_loc],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )
        _, valid_data, _ = dataio_prepare(hparams, tokenizer)
        valid_loader = sb.dataio.dataloader.make_dataloader(
            valid_data, **hparams["valid_dataloader_kwargs"]
        )
        lagp.compute_language_vector(base_loc, valid_loader, num_samples=100)

    # Train on new locales sequentially
    for i, locale in enumerate(hparams["new_locales"]):
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_common_voice,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        # Add new language token (if needed)
        # new_tokens = [f"<|{locale.lower()}|>"]
        tokenizer = hparams["whisper"].tokenizer
        # tokenizer._additional_special_tokens += new_tokens
        # tokenizer.supported_languages.update({locale.lower(): locale.lower()})
        # tokenizer.to_language_codes.update({locale.lower(): locale.lower()})

        # new_tokens = sorted(
        #     list(set(new_tokens) - set(tokenizer.get_vocab().keys()))
        # )
        # tokenizer.add_tokens(new_tokens)

        # logging.info(
        #     f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        # )
        # hparams["whisper"].model.resize_token_embeddings(len(tokenizer))
        # logging.info(
        #     f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        # )

        # hparams["forced_decoder_locale"] = locale
        train_data, valid_data, _ = dataio_prepare(hparams, tokenizer)

        # Trainer initialization
        checkpoint_folder = os.path.join(hparams["save_folder"], locale)
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(checkpoint_folder)
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0

        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
            lagp=lagp,
        )
        asr_brain.tokenizer = tokenizer
        asr_brain.current_lang = locale

        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0
        asr_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )

        # After training, compute language vector for the new language using its validation data
        _, valid_data, _ = dataio_prepare(hparams, tokenizer)
        valid_loader = sb.dataio.dataloader.make_dataloader(
            valid_data, **hparams["valid_dataloader_kwargs"]
        )
        lagp.compute_language_vector(locale, valid_loader, num_samples=100)

        # Optionally add some gradients to the buffer for this language
        # lagp.add_to_buffer(locale)

        # Test after adding this language
        test(
            hparams,
            run_opts,
            hparams["base_locales"] + hparams["new_locales"][: i + 1],
            f"wer_test_after_{locale}.txt",
        )


# ----------------------------------------------------------------------
# Profiling (unchanged)
# ----------------------------------------------------------------------
def profile(hparams, run_opts):
    import ptflops
    import torchinfo

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.whisper = hparams["whisper"]
            self.wavs = torch.randn(
                1, hparams["sample_rate"], device=run_opts["device"],
            )
            self.bos_tokens = torch.ones(
                1,
                self.whisper.model.config.max_length,
                dtype=torch.int,
                device=run_opts["device"],
            )

        @torch.no_grad()
        def forward(self, _=None):
            enc_out, logits, _ = self.whisper(self.wavs, self.bos_tokens)
            return logits

    model = Model().eval().to(run_opts["device"])
    macs, params = ptflops.get_model_complexity_info(
        model, (1,), as_strings=True, print_per_layer_stat=False,
    )
    time_start = time.time()
    model()
    torch.cuda.synchronize()
    time_stop = time.time() - time_start
    max_mem = torch.cuda.max_memory_allocated("cuda") / 10 ** 9
    result = {
        "MACs": macs,
        "memory": max_mem,
        "time": time_stop,
    }
    logging.info(torchinfo.summary(model, verbose=0))
    logging.info(result)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams["train_logger"].save_file = hparams[
        "train_logger"
    ].save_file.replace(
        ".txt",
        f"_base={','.join(hparams['base_locales'])}_new={','.join(hparams['new_locales'])}_lagp_whispervec.txt",
    )

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    class CustomPaddedBatch(PaddedBatch):
        def __init__(self, examples, *args, **kwargs):
            for k in ["tokens_bos", "tokens_eos"]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0.0
                if k in ["tokens_bos"]:
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