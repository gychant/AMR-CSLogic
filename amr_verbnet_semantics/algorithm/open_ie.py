"""
Module for knowledge graph prediction using Stanford OpenIE tool.
https://github.com/philipperemy/stanford-openie-python
"""
import os
import json
from collections import Counter, defaultdict

from openie import StanfordOpenIE

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

client = StanfordOpenIE(properties=properties)


def mine_relations(data, sample_generator, output_file_path,
                   start_idx=None, debug=False, verbose=False):
    apply_open_ie(data, sample_generator, output_file_path,
                  start_idx, debug, verbose)


def map_subj_obj_pair_to_rel(triples):
    results = defaultdict(set)
    for tpl in triples:
        subj, rel, obj = tpl
        results[(subj.lower(), obj.lower())].add(rel.lower())
    return results


def analyze_relations(train_triple_file_path, test_triple_file_path,
                      out_triple_file_path):
    rel2phrases = defaultdict(Counter)
    phrase2rels = defaultdict(Counter)

    with open(train_triple_file_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            data = json.loads(line)
            pred_dict = map_subj_obj_pair_to_rel(data["pred"])
            true_dict = map_subj_obj_pair_to_rel(data["true"])

            for ent_pair in true_dict:
                if ent_pair in pred_dict:
                    for true_rel in true_dict[ent_pair]:
                        for phrase in pred_dict[ent_pair]:
                            rel2phrases[true_rel][phrase] += 1

                    for pred_rel in pred_dict[ent_pair]:
                        for true_rel in true_dict[ent_pair]:
                            phrase2rels[pred_rel][true_rel] += 1
    print("\nrel2phrases:")
    print(rel2phrases)

    print("\nphrase2rels:")
    print(phrase2rels)

    f_out = open(out_triple_file_path, "w")
    with open(test_triple_file_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            data = json.loads(line)
            new_pred_triples = set()
            for triple in data["pred"]:
                phrase = triple[1].lower()
                if phrase in phrase2rels:
                    most_common_rel = phrase2rels[phrase].most_common(1)[0][0]
                    new_pred_triples.add((triple[0].lower(),
                                          most_common_rel,
                                          triple[2].lower()))
            data["pred"] = list(new_pred_triples)
            f_out.write(json.dumps(data))
            f_out.write("\n")
    f_out.close()
    print("Written to file {}".format(out_triple_file_path))
    print("DONE.")


def apply_open_ie(data, sample_generator, output_file_path,
                  start_idx=None, debug=False, verbose=False,
                  debug_dir=None):
    """
    Apply path patterns to induce KG triples from text.
    :param data: a list of raw samples
    :param output_file_path: the output file path for saving induced triples
    :param start_idx: the index of example to start with
    :param debug: whether to debug
    :param verbose: whether to print intermediate outputs
    :param debug_dir: the dir path to store debug info
    :return:
    """
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f_out = open(output_file_path, "a")
    try:
        for sample_idx, sentences in enumerate(sample_generator(
                data, extractable_only=False, verbose=verbose)):
            if sample_idx < start_idx:
                continue

            if debug:
                if debug_dir is None:
                    debug_dir = "test-debug-openie"

                output_dir = "./{}/sample_{}".format(debug_dir, sample_idx)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                f_debug_info = open(os.path.join(output_dir, "sample_{}.txt".format(sample_idx)), "w")

            sample = data[sample_idx]

            all_triples = set()
            for sent_idx, sent in enumerate(sentences):
                if verbose:
                    print("\nsent:", sent)

                if debug:
                    f_debug_info.write("sent_{}: ".format(sent_idx))
                    f_debug_info.write(sent.strip())
                    f_debug_info.write("\n\n")

                triples = induce_kg_triples(sent, verbose=verbose)
                all_triples.update(triples)

            # print("all_triples:", list(all_triples))
            # input()
            result = {
                "idx": sample_idx,
                "pred": list(all_triples),
                "true": sample["state"]["graph"] if "graph" in sample["state"] else []
            }
            # print("result:")
            # print(result)
            f_out.write(json.dumps(result))
            f_out.write("\n")
            if debug:
                f_debug_info.write("\n\n")
                f_debug_info.write("pred:\n")
                f_debug_info.write(str(all_triples))
                f_debug_info.write("\n\ntrue:\n")
                f_debug_info.write(str(result["true"]))
    except Exception as e:
        print("Exception:", e)
        f_out.close()
        if debug:
            f_debug_info.close()
        raise e
    f_out.close()
    if debug:
        f_debug_info.close()
    print("Triple induction DONE.")


def induce_kg_triples(text, verbose=False):
    triples = set()
    for triple in client.annotate(text):
        new_triple = (triple["subject"], triple["relation"], triple["object"])
        triples.add(new_triple)
        if verbose:
            print(new_triple)
    return triples

