"""PropBank query wrapper"""
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader
from amr_verbnet_semantics.corpus_readers.verbnet_reader import VerbnetCorpusReaderEx
from amr_verbnet_semantics.service.propbank import query_propbank_roles

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


if __name__ == '__main__':
    verbnet_id = "escape-51.1-1-1"
    verbnet_version = "verbnet3.4"
    print("\nQuerying verbnet class {} in version {} ...".format(verbnet_id, verbnet_version))
    print("\nresult:")
    print(query_semantics(verbnet_id, verbnet_version))

