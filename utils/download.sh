#!/bin/bash

# Shell script to download files from a list of URLs

# Usage: ./download.sh <input_file> <output_directory>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./download.sh <input_file> <output_directory>"
    exit 1
fi

input_file="$1"
output_dir="$2"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

# Read URLs from the input file and download each file
while IFS= read -r url; do
    # Skip empty lines and comments
    if [[ -z "$url" || "$url" == \#* ]]; then
        continue
    fi
    
    # Extract filename from URL
    filename=$(basename "$url")
    
    echo "Downloading: $url"
    curl -L "$url" -o "$output_dir/$filename"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $filename"
    else
        echo "Failed to download: $url"
    fi
done < "$input_file"

echo "Download process completed."
