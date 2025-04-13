#!/usr/bin/env bash

# Script to download noise and impulse response data for the 5th DNS Challenge (ICASSP 2023)
# The output directory will be passed as a command-line argument

# Check if output directory is specified
if [ $# -lt 1 ]; then
    echo "Error: Please provide an output directory."
    echo "Usage: $0 /path/to/output_directory"
    exit 1
fi

OUTPUT_PATH="$1"
mkdir -p "$OUTPUT_PATH"

# List of .tar.bz2 blobs to download
BLOB_NAMES=(
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2
)

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

for BLOB in "${BLOB_NAMES[@]}"
do
    URL="$AZURE_URL/$BLOB"
    echo "Downloading and extracting: $BLOB"

    # Download and extract in one step
    curl -s "$URL" | tar -C "$OUTPUT_PATH" -xj

    # Optional: confirm extraction
    if [ $? -eq 0 ]; then
        echo "Finished: $BLOB"
    else
        echo "Failed to download or extract: $BLOB"
    fi
done

echo "All files processed. Extracted to: $OUTPUT_PATH"
