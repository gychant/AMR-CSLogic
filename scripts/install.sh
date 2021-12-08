#!/bin/bash

# Install dependencies and setups
# Usage: sh ./install.sh

set -e

pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('verbnet'); nltk.download('propbank')"
unzip ~/nltk_data/corpora/propbank.zip -d ~/nltk_data/corpora/

# Install spaCy
pip install spacy==2.1.0
python -m spacy download en
pip install neuralcoref
pip install benepar --no-cache-dir
python -c "import benepar; benepar.download('benepar_en3')"

pip install torch==1.3

wget https://github.com/blazegraph/database/releases/download/BLAZEGRAPH_2_1_6_RC/blazegraph.jar
