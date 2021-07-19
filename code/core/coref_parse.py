"""Coreference resolution"""

from pprint import pprint
import nltk
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP


STANFORD_CORENLP_PATH = "./stanford-corenlp-full-2018-10-05"
nlp = None


def full_parsing(text):
    global nlp
    if nlp is None:
        print("Init model ...")
        nlp = StanfordCoreNLP(STANFORD_CORENLP_PATH)
        print("Model created ...")

    annotation = dict()
    try:
        annotation["coreference"] = nlp.coref(text)
    except:
        annotation["coreference"] = None

    """
    annotation["sentences"] = []
    for sentence in sent_tokenize(text):
        try:
            word_tokenize = nlp.word_tokenize(sentence)
        except:
            word_tokenize = None

        try:
            pos_tag = nlp.pos_tag(sentence)
        except:
            pos_tag = None

        try:
            dependency_parse = nlp.dependency_parse(sentence)
        except:
            dependency_parse = None

        try:
            constituency_parse = nlp.parse(sentence)
        except:
            constituency_parse = None

        annotation["sentences"].append({
            "word_tokenize": word_tokenize,
            "dependency_parse": dependency_parse,
            "constituency_parse": constituency_parse,
            "pos_tag": pos_tag
        })
    """
    return annotation