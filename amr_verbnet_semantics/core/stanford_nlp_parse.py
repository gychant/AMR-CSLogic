"""StanfordCoreNLP Parsing"""


from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP

from app_config import config

nlp = None


def full_parsing(text, do_coreference=False, do_word_tokenize=False,
                 do_pos_tag=False, do_dependency_parse=False,
                 do_constituency_parse=False):
    global nlp
    if nlp is None:
        print("Loading StanfordCoreNLP models ...")
        if config.STANFORD_CORENLP_PATH is not None:
            nlp = StanfordCoreNLP(config.STANFORD_CORENLP_PATH)
        else:
            # Use an existing server
            nlp = StanfordCoreNLP(config.STANFORD_CORENLP_HOST,
                                  port=config.STANFORD_CORENLP_PORT)
        print("Models loaded ...")

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


if __name__ == "__main__":
    # text = 'My sister has a dog. She loves him .'
    # text = 'Angela lives in Boston. She is quite happy in that city.'
    text = 'Autonomous cars shift insurance liability toward manufacturers.'
    print(full_parsing(text, do_coreference=True, do_dependency_parse=True,
                       do_constituency_parse=True))

