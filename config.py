import logging
import os

# Secret for session management
SECRET_KEY = os.getenv("SECRET_KEY", "sup3r-s3cr3t")
LOGGING_LEVEL = logging.INFO

# Configure AMR parsing
AMR_MODEL_CHECKPOINT_PATH = 'DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep60-seed44/checkpoint_wiki.smatch_best1.pt'
ROBERTA_CACHE_PATH = 'roberta.large'
use_cuda = False

# Configure RDF store
# SPARQL_ENDPOINT = "http://goedel.sl.cloud9.ibm.com:9999/blazegraph/namespace/UL_KB_V0"
SPARQL_ENDPOINT = "http://localhost:9999/blazegraph/namespace/UL_KB_V1_PUB"

# Configure VerbNet
VERBNET_VERSION = "verbnet3.2"

# Configure KB source
KB_SOURCE = "corpus"
# KB_SOURCE = "rdf"

# Configure Conda home
CONDA_HOME = "~/anaconda3"

# Configure AMR parsing
# AMR_PARSING_MODEL = "remote"
AMR_PARSING_MODEL = "local"

# Configure remote AMR parsing service
REMOTE_AMR_HOST = "mnlp-demo.sl.cloud9.ibm.com"
REMOTE_AMR_PORT = 59990

# Configure local service
LOCAL_SERVICE_HOST = "0.0.0.0"
LOCAL_SERVICE_PORT = 5000

