"""
AMR parsing service using a locally deployed model
"""
import os

from third_party.transition_amr_parser.parse import AMRParser
from nltk.tokenize import word_tokenize
import config

PARSER_DIR = f'{os.path.dirname(__file__)}/../../third_party'


class AMRClient(object):
    def __init__(self, use_cuda=False):
        self.parser = None
        self.use_cuda = use_cuda

    def _get_parser(self, use_cuda=False):
        # cwd = os.getcwd()
        # os.chdir(PARSER_DIR)
        amr_parser = AMRParser.from_checkpoint(config.AMR_MODEL_CHECKPOINT_PATH,
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


amr_client = AMRClient(config.use_cuda)


if __name__ == "__main__":
    amr = AMRClient().get_amr("The boy travels")
    print(amr)

