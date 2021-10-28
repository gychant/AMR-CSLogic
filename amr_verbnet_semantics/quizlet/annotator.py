"""
Annotator for the Quizlet7 dataset
"""
import os
import json
import pdb

from amr_verbnet_semantics.core.amr_verbnet_enhance import ground_text_to_verbnet

DATA_DIR = "./data/quizlet7"


def annotate_data(path, output_dir):
    with open(path) as f:
        data = json.load(f)

    for corpus_id in data.keys():
        print("Processing corpus {} ...".format(corpus_id))
        corpus = data[corpus_id]
        sentences = corpus['sentences']
        for sent_idx in range(len(sentences.keys())):
            print("Processing sentence {} ...".format(sent_idx))
            sentence = sentences[str(sent_idx)]["sentence"]
            amr = sentences[str(sent_idx)]["amr"]
            parse = ground_text_to_verbnet(sentence, amr=amr)
            verbnet_parse = parse["sentence_parses"][0]
            del verbnet_parse["text"]
            del verbnet_parse["amr"]
            sentences[str(sent_idx)]["verbnet_parse"] = verbnet_parse

    file_name = os.path.basename(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w") as f:
        json.dump(data, f)
    print("Written to file {}".format(output_path))
    print("DONE.")


if __name__ == "__main__":
    output_dir = os.path.join(DATA_DIR, "parsed")

    input_path = os.path.join(DATA_DIR, "quizlet7.textpp+IE+AMR+PB+ETX.ce2002.json")
    annotate_data(input_path, output_dir)

    input_path = os.path.join(DATA_DIR, "quizlet7.textpp+IE+AMR+PB+ETX.ce2004.json")
    annotate_data(input_path, output_dir)

    input_path = os.path.join(DATA_DIR, "quizlet7.textpp+IE+AMR+PB+ETX.ce2039.json")
    annotate_data(input_path, output_dir)
