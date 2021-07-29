"""PropBank query wrapper"""
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader

from code.service.propbank import query_propbank_roles

vn_dict = {
    "verbnet3.2": LazyCorpusLoader("verbnet3.2", VerbnetCorpusReader, r"(?!\.).*\.xml"),
    "verbnet3.3": LazyCorpusLoader("verbnet3.3", VerbnetCorpusReader, r"(?!\.).*\.xml"),
    "verbnet3.4": LazyCorpusLoader("verbnet3.4", VerbnetCorpusReader, r"(?!\.).*\.xml")
}


def query_semantics(verbnet_id, verbnet_version):
    # print("verbnet_id:", verbnet_id)
    # print("verbnet_version:", verbnet_version)
    frames = vn_dict[verbnet_version].frames(verbnet_id)
    semantics = frames[0]["semantics"]
    # pprint(semantics)
    return semantics

