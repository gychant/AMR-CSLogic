"""PropBank query wrapper"""
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader

from code.service.propbank import query_propbank_roles

vn = LazyCorpusLoader("verbnet3.2", VerbnetCorpusReader, r"(?!\.).*\.xml")


def query_semantics(verbnet_id):
    frames = vn.frames(verbnet_id)
    semantics = frames[0]["semantics"]
    # pprint(semantics)
    return semantics

