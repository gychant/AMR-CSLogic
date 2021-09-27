"""
VerbNet query wrapper
"""
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader

import config
from amr_verbnet_semantics.corpus_readers.rdf_kb import \
    query_semantics_from_rdf


class VerbnetCorpusReaderEx(VerbnetCorpusReader):
    """
    An extension class based on NLTK VerbnetCorpusReader to capture the negation of statement.
    Reference: https://www.nltk.org/_modules/nltk/corpus/reader/verbnet.html
    """

    def __init__(self, root, fileids, wrap_etree=False):
        VerbnetCorpusReader.__init__(self, root, fileids, wrap_etree)

    def _get_semantics_within_frame(self, vnframe):
        """Returns semantics within a single frame

        A utility function to retrieve semantics within a frame in VerbNet
        Members of the semantics dictionary:
        1) Predicate value
        2) Arguments

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: semantics: semantics dictionary
        """
        semantics_within_single_frame = []
        for pred in vnframe.findall("SEMANTICS/PRED"):
            arguments = [
                {"type": arg.get("type"), "value": arg.get("value")}
                for arg in pred.findall("ARGS/ARG")
            ]
            semantics_within_single_frame.append(
                {
                    "predicate_value": pred.get("value"),
                    "arguments": arguments,
                    "is_negative": pred.get("bool", False) == "!"
                }
            )
        return semantics_within_single_frame


vn_dict = {
    "verbnet3.2": LazyCorpusLoader("verbnet3.2", VerbnetCorpusReaderEx, r"(?!\.).*\.xml"),
    "verbnet3.3": LazyCorpusLoader("verbnet3.3", VerbnetCorpusReaderEx, r"(?!\.).*\.xml"),
    "verbnet3.4": LazyCorpusLoader("verbnet3.4", VerbnetCorpusReaderEx, r"(?!\.).*\.xml")
}


def query_semantics_from_corpus(verbnet_id, verbnet_version):
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


def query_semantics(verbnet_id, verbnet_version):
    config.KB_SOURCE = "corpus"
    if config.KB_SOURCE == "rdf":
        return query_semantics_from_rdf(verbnet_id, verbnet_version)
    return query_semantics_from_corpus(verbnet_id, verbnet_version)


if __name__ == '__main__':
    verbnet_id = "escape-51.1"
    verbnet_version = "verbnet3.4"
    print("\nQuerying verbnet class {} in version {} ...".format(verbnet_id, verbnet_version))
    print("\nresult:")
    print(query_semantics(verbnet_id, verbnet_version))

