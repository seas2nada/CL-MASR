#!/usr/bin/bash
# Configuration
WHISPER_VARIANT="whisper-base"
CONFIG_FILE="hparams/train_ft.yaml"
# Check for data folder arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-data-folder>"
    echo "Example: $0 /path/to/CL-MASR/data"
    exit 1
fi
DATA_FOLDER="$1"
GPU="$2"
# Navigate to whisper directory
if [ -d "whisper" ]; then
    cd whisper || exit
    echo "Changed directory to whisper."
elif [ $(basename "$PWD") == "whisper" ]; then
    echo "Already in whisper directory."
else
    echo "Error: 'whisper' directory not found. Please run this script from the CL_MASR root or inside the whisper directory."
    exit 1
fi
# Run the training script
echo "Running FT with $WHISPER_VARIANT using data from $DATA_FOLDER..."
CUDA_VISIBLE_DEVICES=$GPU python train_ft.py "$CONFIG_FILE" --whisper_variant "$WHISPER_VARIANT" --data_folder "$DATA_FOLDER"