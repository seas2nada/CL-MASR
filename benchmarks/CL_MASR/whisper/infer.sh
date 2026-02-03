WHISPER_VARIANT=whisper-small
DATA_FOLDER=/DB/CL-MASR

python infer.py hparams/infer.yaml --whisper_variant "$WHISPER_VARIANT" --data_folder "$DATA_FOLDER"