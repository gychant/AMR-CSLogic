"""Wrappers for local and remote AMR parsing client"""
import os

from third_party.transition_amr_parser.parse import AMRParser
from nltk.tokenize import word_tokenize
from amr_verbnet_semantics.grpc_clients import AMRClientTransformer

import config

PARSER_DIR = f'{os.path.dirname(__file__)}/../../third_party'


class LocalAMRClient(object):
    def __init__(self, use_cuda=False):
        self.parser = None
        self.use_cuda = use_cuda

    def _get_parser(self, use_cuda=False):
        # cwd = os.getcwd()
        # os.chdir(PARSER_DIR)
        amr_parser = AMRParser.from_checkpoint(
            config.AMR_MODEL_CHECKPOINT_PATH,
            roberta_cache_path=config.ROBERTA_CACHE_PATH,
            use_cuda=use_cuda)
        # for loading resources, parse a test sentence
        amr_parser.parse_sentences([['test']])
        # os.chdir(cwd)
        return amr_parser

    def get_amr(self, text):
        if self.parser is None:
            # Lazy loading
            self.parser = self._get_parser(use_cuda=self.use_cuda)

        res = self.parser.parse_sentences([word_tokenize(text)])
        return res[0][0]


class RemoteAMRClient(object):
    def __init__(self):
        self.amr_host = "mnlp-demo.sl.cloud9.ibm.com"
        self.amr_port = 59990
        self.parser = None

    def get_amr(self, text):
        if self.parser is None:
            # Lazy loading
            self.parser = AMRClientTransformer(f"{self.amr_host}:{self.amr_port}")
        return self.parser.get_amr(text)


if config.AMR_PARSING_MODEL == "local":
    amr_client = LocalAMRClient(config.use_cuda)
elif config.AMR_PARSING_MODEL == "remote":
    amr_client = RemoteAMRClient()
else:
    raise Exception("Missing AMR parsing configuration ...")


if __name__ == "__main__":
    text = "The quick brown fox jumped over the lazy moon."
    amr = amr_client.get_amr(text)
    print("\ntext:", text)
    print("\namr:", amr)

