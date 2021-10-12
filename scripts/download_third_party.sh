#!/bin/bash

mkdir third_party

wget https://github.com/IBM/transition-amr-parser/archive/refs/tags/v0.4.2.zip
unzip v0.4.2.zip
mv transition-amr-parser-0.4.2/transition_amr_parser third_party
mv transition-amr-parser-0.4.2/fairseq_ext third_party
rm -rf transition-amr-parser-0.4.2
rm v0.4.2.zip

wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -zxvf roberta.large.tar.gz
mv roberta.large third_party
rm roberta.large.tar.gz
