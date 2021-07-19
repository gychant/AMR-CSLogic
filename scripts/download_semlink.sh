#!/bin/bash

# Download Semlink from its GitHub repository
# Example: sh scripts/download_semlink.sh [save_dir]
# Usage: sh scripts/download_semlink.sh ./data

set -e
save_dir=$1

git clone https://github.com/cu-clear/semlink.git ${save_dir}

echo "Saved to ${save_dir}/semlink"
echo "DONE."