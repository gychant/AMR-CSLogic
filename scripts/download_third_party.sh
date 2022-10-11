#!/bin/bash

# this is called from setup.py in the main project dir
set -e

echo "** Installing third_party **"
rm -rf third_party
mkdir third_party

rm -rf transition-amr-parser
wget https://github.com/IBM/transition-amr-parser/archive/refs/heads/master.zip
unzip master.zip
pushd transition-amr-parser-master
pip install -e .
popd
rm -rf master.zip

# wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
# tar -zxvf roberta.large.tar.gz
# mv roberta.large third_party
# rm roberta.large.tar.gz
