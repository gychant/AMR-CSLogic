"""Build knowledge graphs of the game using AMR-VerbNet semantics"""
import os
import json
import requests
from pprint import pprint
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

from code.core.amr_verbnet_enhance import generate_enhanced_amr, visualize_enhanced_amr

import random
random.seed(42)

# Add path to the dot package binary to system path
os.environ["PATH"] += os.pathsep + '/home/zliang/.conda/envs/amr-verbnet/bin'

host = "0.0.0.0"
port = 5000

DATA_DIR = "./data/JerichoWorld"
sample_size = 10

print("Loading data ...")
with open(os.path.join(DATA_DIR, "train.json")) as f:
    data = json.load(f)
print("Loaded data ...")

tokenizer = RegexpTokenizer(r'\w+')


def kb_statistics():
    count_pred = Counter()
    count_have_subj = Counter()
    count_no_graph = 0
    pred_examples = dict()

    for sample in data:
        if "graph" not in sample["state"]:
            count_no_graph += 1
            continue

        for triple in sample["state"]["graph"]:
            subj, pred, obj = triple
            count_pred[pred] += 1
            if pred.strip().lower() == "have":
                count_have_subj[subj] += 1

            if pred == "wait" and subj != obj:
                print(triple)
                input()
            if pred == "inventory" and subj != obj:
                print(triple)
                input()

            if pred not in pred_examples:
                pred_examples[pred] = set()
            pred_examples[pred].add(tuple(triple))

    print("\npred counter:")
    pprint(count_pred)
    print("\nNum of pred:", len(count_pred))
    print("\nSubjects of have:")
    pprint(count_have_subj)
    print("\ncount_no_graph:", count_no_graph)

    for pred in pred_examples:
        examples = list(pred_examples[pred])
        random.shuffle(examples)
        print("\npred:", pred)
        print("examples:", examples[:5])
    print("DONE.")


def print_sample(sample):
    print("==========================================")
    print("\nState:")
    print(sample["state"])

    print("\nNext State:")
    print(sample["next_state"])

    print("\nObservation:")
    print(sample["state"]["obs"])

    print("\nLocation Description:")
    print(sample["state"]["loc_desc"])

    print("\nSurrounding Objects:")
    print(sample["state"]["surrounding_objs"])

    print("\nSurrounding Attributes:")
    print(sample["state"]["surrounding_attrs"])

    print("\nInventory Description:")
    print(sample["state"]["inv_desc"])

    print("\nInventory Objects:")
    print(sample["state"]["inv_objs"])

    print("\nInventory Attributes:")
    print(sample["state"]["inv_attrs"])

    print("\nGraph:")
    print(sample["state"]["graph"])
    print("==========================================")


def print_enhanced_amr(amr_object):
    print("\namr:")
    print(amr_object["amr"])
    print("\npb_vn_mappings:")
    pprint(amr_object["pb_vn_mappings"])

    print("\nrole_mappings:")
    pprint(amr_object["role_mappings"])

    print("\namr_cal:")
    print(amr_object["amr_cal"])

    print("\nsem_cal:")
    print(amr_object["sem_cal"])

    print("\ngrounded_stmt:")
    print(amr_object["grounded_stmt"])

    print("\namr_cal_str:")
    print(amr_object["amr_cal_str"])

    print("\nsem_cal_str:")
    print(amr_object["sem_cal_str"])

    print("\ngrounded_stmt_str:")
    print(amr_object["grounded_stmt_str"])


def check_samples():
    # Select samples for initial testing
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:sample_size]

    for idx in sample_indices:
        sample = data[idx]

        text = get_observation_text_from_sample(sample)
        # text = sample["state"]["obs"]
        # text = "You see a dishwasher and a fridge."
        # text = "You flip open the pizza box."
        # text = "The dresser is made out of maple carefully finished with Danish oil."
        res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})

        print("\nres.text:")
        print(res.text)

        res = json.loads(res.text)
        if "amr_parse" in res:
            for i in range(len(res["amr_parse"])):
                print_enhanced_amr(res["amr_parse"][i])

                graph = generate_enhanced_amr(
                    amr=res["amr_parse"][i]["amr"],
                    grounded_stmt=res["amr_parse"][i]["grounded_stmt"],
                    semantic_calculus=res["amr_parse"][i]["sem_cal"])

                visualize_enhanced_amr(graph, out_dir="./test-output/")
                print("visualize_enhanced_amr DONE.")
                input()


def to_pure_letter_string(text):
    """
    Remove all punctuations and then spaces in the text
    :param text:
    :return:
    """
    text = text.lower().replace("\n", " ")
    tokens = tokenizer.tokenize(text)
    text = "".join(tokens)
    return text


def get_observation_text_from_sample(sample):
    text = " ".join([sample["state"]["obs"],
                     sample["state"]["loc_desc"],
                     # " ".join(sample["state"]["surrounding_objs"].keys()),
                     sample["state"]["inv_desc"]])
    return text


def mine_path_patterns(verbose=False):
    for idx in range(len(data)):
        sample = data[idx]

        if "graph" not in sample["state"] or \
                len(sample["state"]["graph"]) == 0:
            continue

        if verbose:
            print_sample(sample)

        text = get_observation_text_from_sample(sample)
        sentences = sent_tokenize(text)
        print("\nsentences:")
        print(sentences)
        input()

        for sent in sentences:
            print("sent:", sent)
            concat_text = to_pure_letter_string(sent)
            extractable_triples = []
            for triple in sample["state"]["graph"]:
                subj, pred, obj = triple
                if to_pure_letter_string(subj) in concat_text \
                        and to_pure_letter_string(obj) in concat_text:
                    extractable_triples.append(triple)

            if len(extractable_triples) > 0:
                print(extractable_triples)
            input()


def check_extractable_kg_triples(verbose=False):
    cnt_triples = 0
    cnt_extractable = 0
    cnt_extractable_by_sent = 0

    for idx in range(len(data)):
        sample = data[idx]

        if "graph" not in sample["state"] or \
                len(sample["state"]["graph"]) == 0:
            continue

        if verbose:
            print_sample(sample)

        text = get_observation_text_from_sample(sample)
        concat_text = to_pure_letter_string(text)

        for triple in sample["state"]["graph"]:
            cnt_triples += 1

            subj, pred, obj = triple
            if to_pure_letter_string(subj) in concat_text \
                    and to_pure_letter_string(obj) in concat_text:
                cnt_extractable += 1
            """
            else:
                print("\ntext:\n========================\n", text)
                print("========================")
                print("\n", sample["state"])
                print("\n")
                print(triple, "xxx")
                print("\n", sample["state"]["graph"])
                input()
            """

        sentences = sent_tokenize(text)
        extractable_triples = set()
        for sent in sentences:
            concat_text = to_pure_letter_string(sent)
            for triple in sample["state"]["graph"]:
                subj, pred, obj = triple
                if to_pure_letter_string(subj) in concat_text \
                        and to_pure_letter_string(obj) in concat_text:
                    extractable_triples.add(tuple(triple))
        cnt_extractable_by_sent += len(extractable_triples)

    ratio = cnt_extractable / cnt_triples
    ratio_by_sent = cnt_extractable_by_sent / cnt_triples
    print("Ratio of extractable:", ratio)
    print("Ratio of extractable by sentence:", ratio_by_sent)


if __name__ == "__main__":
    # check_samples()
    # kb_statistics()
    check_extractable_kg_triples()
    # mine_path_patterns()

