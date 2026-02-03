#!/usr/bin/env python3
import os
import sys
import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from train_ft import ASR, dataio_prepare
from common_voice_prepare import prepare_common_voice


def test_only(hparams, run_opts, locales, wer_file):
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
            hparams["wer_computer"] = lambda *args, **kwargs: sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
        else:
            hparams["wer_computer"] = sb.utils.metric_stats.ErrorRateStats

        hparams["forced_decoder_locale"] = locale

        tokenizer = hparams["whisper"].tokenizer
        _, _, test_data = dataio_prepare(hparams, tokenizer)

        asr_brain = ASR(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        asr_brain.tokenizer = tokenizer

        # ✅ 여기서 체크포인트 복구
        asr_brain.checkpointer.recover_if_possible()

        locale_folder = os.path.join(hparams["output_folder"], locale)
        os.makedirs(locale_folder, exist_ok=True)
        asr_brain.hparams.wer_file = os.path.join(locale_folder, wer_file)

        asr_brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )


if __name__ == "__main__":
    # CLI
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # infer에서 꼭 지정해야 할 것들:
    # 1) ckpt_dir: 어떤 checkpoint 폴더를 볼지
    # 2) infer_locales: 어떤 언어들 평가할지
    ckpt_dir = hparams.get("ckpt_dir", None)
    infer_locales = hparams.get("infer_locales", None)
    wer_file = hparams.get("infer_wer_file", "wer_infer.txt")

    if ckpt_dir is None:
        raise SystemExit("Pass --ckpt_dir <path-to-checkpoints>")
    if infer_locales is None:
        raise SystemExit('Pass --infer_locales "en,de,..."')

    # ✅ checkpointer가 보는 경로를 infer에서 덮어쓰기
    hparams["checkpointer"].checkpoints_dir = pathlib.Path(ckpt_dir)

    locales = [x.strip() for x in infer_locales]
    test_only(hparams, run_opts, locales, wer_file=wer_file)
