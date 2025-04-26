#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Check for target directory argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/target_directory"
    exit 1
fi

OUT_DIR="$1"
mkdir -p "$OUT_DIR"

# List of archive URLs
URLS=(
    "https://www.openslr.org/resources/141/test_clean.tar.gz"
    "https://www.openslr.org/resources/141/test_other.tar.gz"
    "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
    "https://www.openslr.org/resources/141/train_clean_360.tar.gz"
)

echo "Downloading and extracting files to: $OUT_DIR"

for URL in "${URLS[@]}"; do
    FILENAME=$(basename "$URL")

    echo "Processing $FILENAME ..."
    
    # Stream the archive directly into tar to save memory and space
    curl -L "$URL" | tar -xz -C "$OUT_DIR"
done

echo "All files downloaded and extracted to: $OUT_DIR"
