#!/bin/bash

# Install dependencies and setups
# Usage: sh ./install.sh

set -e

pip install --upgrade pip
pip install -e .

# Install spaCy
pip install spacy==2.1.0
python -m spacy download en
# For coreference resolution
pip install neuralcoref
# For constituency parsing
pip install benepar --no-cache-dir
python -c "import benepar; benepar.download('benepar_en3')"

# Install PyTorch 1.3
pip install torch==1.3

# Download Blazegraph for storing the KG
mkdir -p blazegraph
wget -O ./blazegraph/blazegraph.jar https://github.com/blazegraph/database/releases/download/BLAZEGRAPH_2_1_6_RC/blazegraph.jar
