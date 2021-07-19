#!/bin/bash

# Download stanford core-nlp models
# Example: sh scripts/download_stanford_nlp.sh [save_dir]
# Usage: sh scripts/download_stanford_nlp.sh ./data

set -e
save_dir=$1

wget -P ${save_dir} http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip

echo "Unzip ..."
unzip ${save_dir}/stanford-corenlp-full-2018-10-05.zip
echo "Saved to ${save_dir}/stanford-corenlp-full-2018-10-05"
rm -rf ${save_dir}/stanford-corenlp-full-2018-10-05.zip

echo "Download & unzip DONE."
