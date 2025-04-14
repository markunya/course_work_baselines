#!/usr/bin/env bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/target_directory"
    exit 1
fi

OUT_DIR="$1"
mkdir -p "$OUT_DIR"

ZIP_NAME="DAPSv1.0.zip"
ZIP_URL="https://zenodo.org/records/4660670/files/$ZIP_NAME"

echo "Downloading DAPS dataset to: $OUT_DIR"

# Download the zip file
wget -O "$OUT_DIR/$ZIP_NAME" "$ZIP_URL"

# Extract the zip file
echo "Extracting..."
unzip -q "$OUT_DIR/$ZIP_NAME" -d "$OUT_DIR"

# Remove the zip file
rm "$OUT_DIR/$ZIP_NAME"

echo "Download and extraction complete. Dataset available at: $OUT_DIR"
