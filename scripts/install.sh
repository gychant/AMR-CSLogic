#!/bin/bash

# Install dependencies and setups
# Usage: sh ./install.sh

set -e

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('verbnet'); nltk.download('propbank')"