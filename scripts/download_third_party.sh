#!/bin/bash

# this is called from setup.py in the main project dir
set -e

echo "** Installing third_party **"
mkdir third_party

rm -rf transition-amr-parser
wget https://github.com/IBM/transition-amr-parser/archive/refs/heads/master.zip
unzip master.zip
pushd transition-amr-parser-master
conda install pyg -c pyg
pip install -e .
popd
rm -rf master.zip

echo "** Downloading BART model **"
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -zxvf bart.large.tar.gz
mv bart.large third_party
rm -rf bart.large.tar.gz
