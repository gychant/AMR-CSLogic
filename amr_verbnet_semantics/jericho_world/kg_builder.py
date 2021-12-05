"""Build knowledge graphs of the game using AMR-VerbNet semantics"""
import argparse
import json
import os
import pickle
import random
import time
import requests
from tqdm import tqdm, trange
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pprint import pprint
from prettytable import PrettyTable

import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.simple_paths import all_simple_paths

from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize

from amr_verbnet_semantics.core.amr_verbnet_enhance import (
    build_graph_from_amr,
    build_semantic_graph,
    induce_unique_groundings,
    visualize_semantic_graph)
from amr_verbnet_semantics.algorithm.path_pattern import \
    mine_path_patterns, \
    apply_path_patterns
from amr_verbnet_semantics.algorithm.open_ie import \
    apply_open_ie, \
    analyze_relations
from amr_verbnet_semantics.utils.eval_util import Metric
from amr_verbnet_semantics.utils.amr_util import \
    build_amr_parse_cache
from amr_verbnet_semantics.utils.text_util import \
    is_extractable, \
    split_text_into_sentences

random.seed(42)

# Add path to the dot package binary to system path
os.environ["PATH"] += os.pathsep + '/home/zliang/.conda/envs/amr-verbnet/bin'

host = "0.0.0.0"
port = 5000

tokenizer = RegexpTokenizer(r'\w+')


def kb_statistics(data):
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


def game_name_statistics(data):
    cnt_game = Counter()
    for sample in data:
        cnt_game[sample["rom"]] += 1

    print("\ngame name counter:")
    pprint(cnt_game)


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
    """
    Construct text input from sample.
    :param sample:
    :return:
    """
    text = " ".join([sample["state"]["obs"],
                     sample["state"]["loc_desc"],
                     # " ".join(sample["state"]["surrounding_objs"].keys()),
                     sample["state"]["inv_desc"]])
    return text





def compute_metrics(samples, triple_file_path):
    game2metric = dict()
    game2metric["overall"] = Metric()
    overall_metric = game2metric["overall"]

    sample_idx = 0
    with open(triple_file_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            print("\nsample_idx:", sample_idx)
            print("samples len:", len(samples))
            print("line:", line)
            # input()
            game_name = samples[sample_idx]["rom"]
            sample_idx += 1

            if game_name not in game2metric:
                game2metric[game_name] = Metric()
            metric = game2metric[game_name]

            data = json.loads(line)

            metric.cnt_samples += 1
            overall_metric.cnt_samples += 1

            if len(data["true"]) == 0:
                metric.cnt_samples_wo_true_triples += 1
                overall_metric.cnt_samples_wo_true_triples += 1
                continue

            if len(data["pred"]) == 0:
                metric.cnt_samples_wo_pred_triples += 1
                overall_metric.cnt_samples_wo_pred_triples += 1
                continue

            pred_triples = set()
            for subj, rel, obj in data["pred"]:
                subj = subj.lower().strip()
                rel = rel.lower().strip()
                obj = obj.lower().strip()
                if subj == obj:
                    continue

                triple = (subj, rel, obj)
                pred_triples.add(triple)
                metric.pred_triples_by_rel[rel].add(triple)
                overall_metric.pred_triples_by_rel[rel].add(triple)

            true_triples = set()
            for subj, rel, obj in data["true"]:
                subj = subj.lower().strip()
                rel = rel.lower().strip()
                obj = obj.lower().strip()
                if subj == obj:
                    continue

                triple = (subj, rel, obj)
                true_triples.add(triple)
                metric.rel_counter[rel.lower()] += 1
                overall_metric.rel_counter[rel.lower()] += 1
                metric.true_triples_by_rel[rel].add(triple)
                overall_metric.true_triples_by_rel[rel].add(triple)

            # compute sample-wise scores
            true_pred_triples = pred_triples.intersection(true_triples)
            # compute precision
            if len(pred_triples) == 0:
                prec = 0
            else:
                prec = len(true_pred_triples) / len(pred_triples)
            metric.sum_prec += prec
            overall_metric.sum_prec += prec

            # compute recall
            recall = len(true_pred_triples) / len(true_triples)
            metric.sum_recall += recall
            overall_metric.sum_recall += recall

            # compute f1
            if prec + recall > 0:
                f1 = 2 * prec * recall / (prec + recall)
            else:
                f1 = 0
            metric.sum_f1 += f1
            overall_metric.sum_f1 += f1

    print("\nglobal rel_counter:")
    print(overall_metric.rel_counter)

    for game in game2metric:
        metric = game2metric[game]
        avg_prec = metric.sum_prec / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        avg_recall = metric.sum_recall / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        avg_f1 = metric.sum_f1 / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        print("\ngame:", game)
        print("\ncnt_samples:", metric.cnt_samples)
        print("cnt_samples_wo_true_triples:", metric.cnt_samples_wo_true_triples)
        print("\nrel_counter:")
        print(metric.rel_counter)
        print("\nrel2metric:")
        pprint(metric.compute_relationwise_metric())
        print("\navg_prec:", "{:.3f}".format(avg_prec * 100))
        print("avg_recall:", "{:.3f}".format(avg_recall * 100))
        print("avg_f1:", "{:.3f}".format(avg_f1 * 100))
        print()

    write_results_to_file(game2metric, triple_file_path.replace(".jsonl", "_metrics.txt"))


def write_results_to_file(game2metric, output_file_path):
    base_dir = os.path.dirname(output_file_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    relations = sorted(list(list(game2metric.values())[0].rel_counter.keys()))
    relations_prec_recall_f1 = []
    for rel in relations:
        if rel not in ["have", "in"]:
            continue

        relations_prec_recall_f1.append(rel + " (p)")
        relations_prec_recall_f1.append(rel + " (r)")
        relations_prec_recall_f1.append(rel + " (f)")

    t = PrettyTable()
    t.field_names = ["Game", "avg_prec", "avg_recall", "avg_f1"] + relations_prec_recall_f1

    for game in game2metric:
        metric = game2metric[game]
        avg_prec = metric.sum_prec / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        avg_recall = metric.sum_recall / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        avg_f1 = metric.sum_f1 / (metric.cnt_samples - metric.cnt_samples_wo_true_triples)
        relationwise_metric = metric.compute_relationwise_metric()

        # use percentage points
        row_values = [game,
                      "{:.3f}".format(avg_prec * 100),
                      "{:.3f}".format(avg_recall * 100),
                      "{:.3f}".format(avg_f1 * 100)]

        for rel in relations:
            if rel not in ["have", "in"]:
                continue

            if rel in relationwise_metric:
                score_dict = relationwise_metric[rel]
                row_values.append("{:.3f}".format(score_dict["prec"]))
                row_values.append("{:.3f}".format(score_dict["recall"]))
                row_values.append("{:.3f}".format(score_dict["f1"]))
            else:
                row_values.append("")
                row_values.append("")
                row_values.append("")
        t.add_row(row_values)

    with open(output_file_path, "w") as f_obj:
        f_obj.write(str(t))
    print("Written to file {}".format(output_file_path))


def check_samples(data, sample_size=10):
    # Select samples for initial testing
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:sample_size]

    for idx in sample_indices:
        sample = data[idx]

        # text = get_observation_text_from_sample(sample)
        # text = sample["state"]["obs"]
        # text = "You see a dishwasher and a fridge."
        # text = "You flip open the pizza box."
        # text = "The dresser is made out of maple carefully finished with Danish oil."
        # text = "In accordance with our acceptance of funds from the U.S. Treasury, cash dividends on common stock are not permitted without prior approval from the U.S."
        text = "You are carrying : a bronze-hilted dagger, a clay ocarina, armor and silks ( worn ) ."
        res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})

        print("\nres.text:")
        print(res.text)

        res = json.loads(res.text)
        if "amr_parse" in res:
            for i in range(len(res["amr_parse"])):
                print_enhanced_amr(res["amr_parse"][i])
                list_grounded_stmt, list_semantic_calc = induce_unique_groundings(
                    grounded_stmt=res["amr_parse"][i]["grounded_stmt"],
                    semantic_calc=res["amr_parse"][i]["sem_cal"])

                graph_idx = 0
                for grounded_stmt, semantic_calc in zip(
                        list_grounded_stmt, list_semantic_calc):
                    graph, amr_obj = build_semantic_graph(
                        amr=res["amr_parse"][i]["amr"],
                        grounded_stmt=grounded_stmt,
                        semantic_calculus=semantic_calc)

                    visualize_semantic_graph(
                        graph, graph_name="semantic_graph_{}".format(graph_idx),
                        out_dir="./test-output/")
                    graph_idx += 1
                print("visualize_semantic_graph DONE.")
                input()


def sample_generator(data, extractable_only=True, sample_size=None, verbose=False):
    """
    A generator that produces samples from the training/test data
    :param data: a list of raw samples
    :param extractable_only: if yielding only sample sentences that can be extracted
    :param sample_size: the size of samples for sanity check
    :param verbose:
    :return: a generator
    """
    if sample_size is not None:
        # Select samples for initial testing
        all_indices = list(range(len(data)))
        random.shuffle(all_indices)
        sample_indices = all_indices[:sample_size]
        index_gen = tqdm((n for n in sample_indices), total=sample_size)
    else:
        index_gen = trange(len(data))

    for idx in index_gen:
        sample = data[idx]

        if extractable_only and "graph" not in sample["state"]:
            yield []
            continue

        if verbose:
            print_sample(sample)

        text = get_observation_text_from_sample(sample)
        sentences = split_text_into_sentences(text)
        if verbose:
            print("\nsentences:")
            print(sentences)

        if not extractable_only:
            yield sentences
            continue

        processed_sentences = []
        for sent in sentences:
            for triple in sample["state"]["graph"]:
                subj, pred, obj = triple
                if len(subj.strip()) == 0 or len(pred.strip()) == 0 or len(obj.strip()) == 0:
                    continue

                if subj.strip().lower() == obj.strip().lower():
                    continue

                if is_extractable(sent, triple):
                    if verbose:
                        print("\nsent:", sent)
                    processed_sentences.append(sent)
                    break

        yield processed_sentences


def check_extractable_kg_triples(data, verbose=False):
    """
    Check the statistics of KG triples that can be extracted
    condisdering the text spans.
    :param data:
    :param verbose:
    :return:
    """
    cnt_triples = 0
    cnt_extractable = 0
    cnt_extractable_by_sent = 0

    for idx in trange(len(data)):
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
            # concat_text = to_pure_letter_string(sent)
            for triple in sample["state"]["graph"]:
                # subj, pred, obj = triple
                # if to_pure_letter_string(subj) in concat_text \
                #        and to_pure_letter_string(obj) in concat_text:
                if is_extractable(sent, triple):
                    extractable_triples.add(tuple(triple))
        cnt_extractable_by_sent += len(extractable_triples)

    ratio = cnt_extractable / cnt_triples
    ratio_by_sent = cnt_extractable_by_sent / cnt_triples
    print("cnt_triples:", cnt_triples)
    print("cnt_extractable:", cnt_extractable)
    print("cnt_extractable_by_sent:", cnt_extractable_by_sent)
    print("Ratio of extractable:", ratio)
    print("Ratio of extractable by sentence:", ratio_by_sent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/JerichoWorld")

    parser.add_argument('--check_samples', action='store_true', help="check_samples")
    parser.add_argument('--kb_statistics', action='store_true', help="kb_statistics")
    parser.add_argument('--game_name_statistics', action='store_true', help="game_name_statistics")
    parser.add_argument('--check_extractable_kg_triples', action='store_true',
                        help="check_extractable_kg_triples")
    parser.add_argument('--algorithm', type=str, choices=["path_pattern", "open_ie"],
                        help="algorithm")

    # Path pattern algorithm related
    parser.add_argument('--mine_path_patterns', action='store_true', help="mine_path_patterns")
    parser.add_argument('--apply_path_patterns', action='store_true', help="apply_path_patterns")
    parser.add_argument('--path_pattern_source', type=str, default=None, help="path_pattern_source")
    parser.add_argument('--pattern_file_path', type=str, default=None, help="pattern_file_path")
    parser.add_argument('--output_file_path', type=str, default=None, help="output_file_path")
    parser.add_argument('--graph_type', type=str, default=None, choices=["amr", "amr_verbnet", "verbnet"],
                        help="graph_type")
    parser.add_argument('--top_k_patterns', type=int, default=20, help="top_k_patterns")

    # Open IE related
    parser.add_argument('--mine_relations', action='store_true', help="mine_relations")
    parser.add_argument('--analyze_relations', action='store_true', help="analyze_relations")
    parser.add_argument('--train_triple_file_path', type=str, default=None, help="train_triple_file_path")
    parser.add_argument('--out_triple_file_path', type=str, default=None, help="out_triple_file_path")

    # AMR cache related
    parser.add_argument('--build_amr_parse_cache', action='store_true', help="build_amr_parse_cache")
    parser.add_argument('--amr_cache_path', type=str, default=None, help="amr_cache_path")

    parser.add_argument('--compute_metrics', action='store_true', help="compute_metrics")
    parser.add_argument('--test_triple_file_path', type=str, default=None, help="test_triple_file_path")
    parser.add_argument('--sample_start_idx', type=int, default=0, help="sample_start_idx")
    parser.add_argument('--split_type', type=str, choices=["train", "test"],
                        default="train", help="split_type")
    parser.add_argument('--debug', action='store_true', help="debug")
    parser.add_argument('--debug_dir', type=str, default=None, help="debug_dir")
    parser.add_argument('--verbose', action='store_true', help="verbose")
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    print("Loading training data ...")
    with open(os.path.join(DATA_DIR, "train.json")) as f:
        train_data = json.load(f)
    print("Loaded training data ...")
    print("Size:", len(train_data))

    print("Loading test data ...")
    with open(os.path.join(DATA_DIR, "test.json")) as f:
        test_data = json.load(f)
    print("Loaded test data ...")
    print("Size:", len(test_data))

    if args.check_samples:
        check_samples(train_data)
    elif args.kb_statistics:
        kb_statistics(train_data)
    elif args.game_name_statistics:
        game_name_statistics(test_data)
    elif args.check_extractable_kg_triples:
        check_extractable_kg_triples(train_data)
    elif args.algorithm == "path_pattern":
        if args.mine_path_patterns:
            mine_path_patterns(train_data, sample_generator,
                               output_file_path=args.output_file_path,
                               graph_type=args.graph_type,
                               amr_cache_path=args.amr_cache_path,
                               sample_size=None, verbose=args.verbose)
        elif args.apply_path_patterns:
            apply_path_patterns(test_data, sample_generator,
                                pattern_file_path=args.pattern_file_path,
                                output_file_path=args.output_file_path, graph_type=args.graph_type,
                                top_k_patterns=args.top_k_patterns, amr_cache_path=args.amr_cache_path,
                                start_idx=args.sample_start_idx, debug=args.debug,
                                verbose=args.verbose)
    elif args.algorithm == "open_ie":
        if args.mine_relations:
            apply_open_ie(train_data, sample_generator,
                          output_file_path=args.output_file_path,
                          start_idx=args.sample_start_idx, debug=args.debug,
                          verbose=args.verbose, debug_dir=args.debug_dir)
        elif args.analyze_relations:
            analyze_relations(args.train_triple_file_path,
                              args.test_triple_file_path,
                              args.out_triple_file_path)
        else:
            apply_open_ie(test_data, sample_generator,
                          output_file_path=args.output_file_path,
                          start_idx=args.sample_start_idx, debug=args.debug,
                          verbose=args.verbose)
    elif args.build_amr_parse_cache:
        if args.split_type == "train":
            build_amr_parse_cache(train_data, sample_generator,
                                  args.amr_cache_path,
                                  start_idx=args.sample_start_idx,
                                  extractable_only=True)
        elif args.split_type == "test":
            build_amr_parse_cache(test_data, sample_generator,
                                  args.amr_cache_path,
                                  start_idx=args.sample_start_idx,
                                  extractable_only=False)
    elif args.compute_metrics:
        compute_metrics(test_data, args.test_triple_file_path)
    print("DONE.")

