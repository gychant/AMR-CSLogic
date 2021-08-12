"""PropBank query wrapper"""
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader
from code.corpus_readers.verbnet_reader import VerbnetCorpusReaderEx

from code.service.propbank import query_propbank_roles

vn_dict = {
    "verbnet3.2": LazyCorpusLoader("verbnet3.2", VerbnetCorpusReaderEx, r"(?!\.).*\.xml"),
    "verbnet3.3": LazyCorpusLoader("verbnet3.3", VerbnetCorpusReaderEx, r"(?!\.).*\.xml"),
    "verbnet3.4": LazyCorpusLoader("verbnet3.4", VerbnetCorpusReaderEx, r"(?!\.).*\.xml")
}


def query_semantics(verbnet_id, verbnet_version):
    semantics = dict()
    frames = vn_dict[verbnet_version].frames(verbnet_id)

    for frame in frames:
        roles = set()
        for element in frame["syntax"]:
            role = element["modifiers"]["value"].strip()
            if len(role) > 0 and role.istitle():
                # not empty string and is title-case
                roles.add(role)
        semantics[tuple(roles)] = frame["semantics"]
    return semantics

