
import nltk
from nltk.corpus.reader import VerbnetCorpusReader
from nltk.corpus.util import LazyCorpusLoader

verbnet = LazyCorpusLoader("verbnet3.4", VerbnetCorpusReader, r"(?!\.).*\.xml")
print(vn.frames("escape-51.1-1"))
# print(vn.subclasses("escape-51.1-1"))
# print(vn.frames("leave-51.2"))
# print(vn.frames("escape-51.1-2"))

