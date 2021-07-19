import json
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from typing import List

from code.grpc_clients import AMRClientTransformer
from code.wikipedia.coref_parse import full_parsing

amr_host = "mnlp-demo.sl.cloud9.ibm.com"
amr_port = 59990
amr_client = AMRClientTransformer(f"{amr_host}:{amr_port}")


def parse_timeline_text(input_path):
    with open(input_path) as f:
        data = json.load(f)

    for cate in data:
        for timeline in data[cate]:
            text = timeline["description"].strip()
            sentences = sent_tokenize(text)
            print("parsing ...")
            parse = full_parsing(text)
            print("\ntext:\n", text)
            print("\nsentences:\n==>", "\n\n==>".join(sentences))
            print("\ncoreference:\n", parse["coreference"])
            for sent in sentences:
                amr = amr_client.get_amr(sent)
                print("\n" + amr)
                input()


def test_amr_endpoint():
    amr = amr_client.get_amr("The quick brown fox jumped over the lazy moon.")
    print(amr)


if __name__ == "__main__":
    """
    amr_host = "mnlp-demo.sl.cloud9.ibm.com"
    amr_port = 59990
    test_amr_endpoint(amr_host, amr_port)
    """
    parse_timeline_text("./data/wikipedia/Timeline_of_the_COVID-19_pandemic_in_January_2020.json")

