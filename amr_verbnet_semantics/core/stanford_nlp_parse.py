"""StanfordCoreNLP Parsing"""

from pprint import pprint

import nltk
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP

STANFORD_CORENLP_PATH = "./stanford-corenlp-full-2018-10-05"
nlp = None


def full_parsing(text, do_coreference=False, do_word_tokenize=False,
                 do_pos_tag=False, do_dependency_parse=False,
                 do_constituency_parse=False):
    global nlp
    if nlp is None:
        print("Loading StanfordCoreNLP models ...")
        nlp = StanfordCoreNLP(STANFORD_CORENLP_PATH)
        print("Models created ...")

    annotation = dict()

    if do_coreference:
        try:
            annotation["coreference"] = nlp.coref(text)
        except:
            annotation["coreference"] = None

    if any([do_word_tokenize, do_pos_tag, do_dependency_parse, do_constituency_parse]):
        annotation["sentences"] = []

        for sentence in sent_tokenize(text):
            sentence_parse = dict()
            sentence_parse["text"] = sentence
            
            if do_word_tokenize:
                try:
                    sentence_parse["word_tokenize"] = nlp.word_tokenize(sentence)
                except:
                    pass

            if do_pos_tag:
                try:
                    sentence_parse["pos_tag"] = nlp.pos_tag(sentence)
                except:
                    pass

            if do_dependency_parse:
                try:
                    sentence_parse["dependency_parse"] = nlp.dependency_parse(sentence)
                except:
                    pass

            if do_constituency_parse:
                try:
                    sentence_parse["constituency_parse"] = nlp.parse(sentence)
                except:
                    pass

            annotation["sentences"].append(sentence_parse)
    return annotation