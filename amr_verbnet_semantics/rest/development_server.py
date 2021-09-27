import argparse
import os

print(os.getcwd())


from flask import Flask, request

from amr_verbnet_semantics.core.amr_verbnet_enhance import \
    ground_text_to_verbnet
from amr_verbnet_semantics.service.local_amr import AMRClient

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=3000)
parser.add_argument('--use_cuda', action='store_true')
args = parser.parse_args()

parser = AMRClient(use_cuda=args.use_cuda)
app = Flask(__name__)


@app.route('/')
def top():
    return __file__


@app.route('/process', methods=['POST'])
def process():
    print(request)
    text = request.form['text']

    ret = ground_text_to_verbnet(text, local_amr_client=parser,
                                 use_coreference=False)

    return ret


app.run(host=args.host, port=args.port, debug=True)
