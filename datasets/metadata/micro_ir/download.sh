#!/bin/bash

# Check if target path is provided
if [ $# -lt 1 ]; then
    echo "Error: Please provide a target directory to download files."
    echo "Usage: $0 /path/to/output_directory"
    exit 1
fi

OUTPUT_PATH="$1"
mkdir -p "$OUTPUT_PATH"

# List of impulse responses
IRS=(
    IR_STC4035.wav
    IR_OktavaMD57.wav
    IR_Crystal.wav
    IR_AKGD12.wav
    IR_Lomo52A5M.wav
    IR_MelodiumRM6.wav
    IR_GaumontKalee.wav
)

AZURE_URL="http://xaudia.com/MicIRP/IR"

for IR in "${IRS[@]}"
do
    URL="$AZURE_URL/$IR"
    echo "Downloading $IR ..."
    curl -s -o "$OUTPUT_PATH/$IR" "$URL"
done

echo "Download complete. Files saved to: $OUTPUT_PATH"
