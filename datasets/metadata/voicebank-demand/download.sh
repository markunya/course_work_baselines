#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/target_directory"
    exit 1
fi

OUT_DIR="$1"
mkdir -p "$OUT_DIR"

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: 'kaggle' CLI not found. Install it with 'pip install kaggle'"
    exit 1
fi

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API token not found. Place kaggle.json in ~/.kaggle/"
    exit 1
fi

# Ensure correct permissions
chmod 600 ~/.kaggle/kaggle.json

# Download dataset and unzip
echo "Downloading VoiceBank-Demand dataset to: $OUT_DIR"
kaggle datasets download -d jiangwq666/voicebank-demand -p "$OUT_DIR" --unzip

echo "Download complete. Dataset extracted to: $OUT_DIR"
