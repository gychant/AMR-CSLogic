"""Build knowledge graphs of the game using AMR-VerbNet semantics"""
import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pprint import pprint

import networkx as nx
import requests
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.simple_paths import all_simple_paths
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from tqdm import tqdm, trange

from amr_verbnet_semantics.core.amr_verbnet_enhance import (
    build_graph_from_amr, build_semantic_graph, ground_text_to_verbnet,
    induce_unique_groundings, visualize_semantic_graph)
from amr_verbnet_semantics.service.amr import amr_client
from amr_verbnet_semantics.utils.eval_util import Metric

random.seed(42)

# Add path to the dot package binary to system path
os.environ["PATH"] += os.pathsep + '/home/zliang/.conda/envs/amr-verbnet/bin'

host = "0.0.0.0"
port = 5000

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()


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


def map_tokens_to_nodes(tokens, token2node_id):
    """
    Map tokens in text to nodes in the AMR graph (with matching
    using stemming).
    :param tokens: tokens in text
    :param token2node_id: dictionary mapping tokens to node ids
    :return: a list of mapped nodes
    """
    nodes = []
    for tok in tokens:
        if tok in token2node_id:
            nodes.append(token2node_id[tok])
        else:
            stemmed_tok = stemmer.stem(tok)
            if stemmed_tok in token2node_id:
                nodes.append(token2node_id[stemmed_tok])
            else:
                stem_dict = {stemmer.stem(t): t for t in token2node_id}
                if stemmed_tok in stem_dict:
                    nodes.append(token2node_id[stem_dict[stemmed_tok]])
    return nodes


def get_lowest_common_ancestor(g, nodes):
    """
    Get the lowest common ancestor of a set of nodes
    :param g: networkx graph instance
    :param nodes: a set of nodes
    :return:
    """
    if nodes is None or len(nodes) == 0:
        return None

    if len(nodes) == 1:
        return nodes[0]

    while len(nodes) >= 2:
        ancestors = set()
        for pair in list(combinations(nodes, 2)):
            anc = lowest_common_ancestor(g, *pair)
            if anc is None:
                return None
            ancestors.add(anc)
        nodes = list(ancestors)
    return nodes[0]


def query_paths(graph, cutoff):
    """
    Get all node pairs of the graph and search paths for each with the specified
    cutoff.
    :param graph: A undirected graph from AMR parse
    :return: A dictionary with path as key and set of node pair tuple as values
    """
    paths = defaultdict(set)
    # print("\nnodes:")
    # print(graph.nodes())
    # print("cutoff:", cutoff)

    nodes = graph.nodes()
    for node_pair in list(combinations(nodes, 2)):
        src, tgt = node_pair
        node_pair_paths = all_simple_paths(graph, src, tgt, cutoff)
        for path in node_pair_paths:
            if len(path) < cutoff:
                continue
            labeled_path = tuple(convert_to_labeled_path(graph, path))
            paths[labeled_path].add(node_pair)
    return paths


def read_tokenization(amr):
    """
    Read word tokenization from the AMR parse
    :param amr: AMR parse
    :return: List of tokens
    """
    for line in amr.split("\n"):
        if line.startswith("# ::tok"):
            # tokenized text
            text = line[len("# ::tok"):-len("<ROOT>")].strip()
            tokens = text.split()
            return tokens
    return None


def find_text_span(sent_tokens, target_token_set):
    """
    Find the span of text that corresponds to a given token set
    :param sent_tokens: A tokenized sentence
    :param target_token_set: the token set to match
    :return: the found text span
    """
    # print("\nsent_tokens:", sent_tokens)
    # print("\ntarget_token_set:", target_token_set)
    if len(target_token_set) > 6:
        # a large token set would take a lot of time due to permutation
        return None

    for perm_idx, perm_tokens in enumerate(permutations(target_token_set)):
        # print("perm_idx:", perm_idx)
        span_tokens = []
        for idx, tok in enumerate(sent_tokens):
            cur_idx = len(span_tokens)
            if cur_idx < len(perm_tokens) and tok == perm_tokens[cur_idx]:
                span_tokens.append(idx)
                if len(span_tokens) == len(perm_tokens):
                    return " ".join(perm_tokens)
            else:
                span_tokens = []
    return None


def get_descendant_leaf_nodes(graph, ancestor_node):
    leaf_nodes = [node for node in nx.descendants(graph, ancestor_node)
                  if graph.in_degree(node) != 0 and graph.out_degree(node) == 0]
    return leaf_nodes


def get_node_label(graph, node):
    node_dict = graph.nodes(data=True)
    if node in node_dict:
        label = node_dict[node]["label"]
        if label.startswith("\"") and label.endswith("\""):
            label = label[1:-1]
        return label
    return None


def get_token_by_node(node, amr_obj):
    if node not in amr_obj["node_id2token"]:
        return None
    return amr_obj["node_id2token"][node]


def induce_kg_triples(text, pattern_dict, top_k_patterns,
                      graph_type="amr", amr=None, verbose=False):
    """
    Induce triples from text using given patterns
    :param amr: the AMR parse
    :param pattern_dict: a dictionary storing path patterns for
        all prefined relations
    :param top_k_patterns: top k patterns to apply
    :param graph_type: the type of graph for KG triple induction
        with values ["amr", "amr_verbnet", "verbnet"]
    :param verbose:
    :return:
    """
    all_triples = set()
    parse = ground_text_to_verbnet(text, amr=amr, verbose=verbose)
    sentence_parses = parse["sentence_parses"]

    for i in range(len(sentence_parses)):
        # if verbose:
        #     print_enhanced_amr(sentence_parses[i])

        list_grounded_stmt, list_semantic_calc = induce_unique_groundings(
            grounded_stmt=sentence_parses[i]["grounded_stmt"],
            semantic_calc=sentence_parses[i]["sem_cal"])

        for grounded_stmt, semantic_calc in zip(
                list_grounded_stmt, list_semantic_calc):
            graph, amr_obj = build_semantic_graph(
                amr=sentence_parses[i]["amr"],
                grounded_stmt=grounded_stmt,
                semantic_calculus=semantic_calc)

            triples = induce_kg_triples_from_grounding(
                graph, sentence_parses[i]["amr"], grounded_stmt, semantic_calc,
                pattern_dict, top_k_patterns, graph_type, verbose=verbose)
            all_triples.update(list(triples))
    return all_triples


def induce_kg_triples_from_grounding(g_directed, amr, grounded_stmt, semantic_calc,
                                     pattern_dict, top_k_patterns, graph_type="amr",
                                     verbose=False):
    """
    Induce triples from text using given patterns
    :param g_directed: the directed graph constructed from parse
    :param amr: the AMR parse
    :param grounded_stmt: grounded statement
    :param semantic_calc: semantic calculus
    :param pattern_dict: a dictionary storing path patterns for
        all prefined relations
    :param top_k_patterns: top k patterns to apply
    :param graph_type: the type of graph for KG triple induction
            with values ["amr", "amr_verbnet", "verbnet"]
    :param verbose:
    :return:
    """
    triples = set()
    amr_tokens = read_tokenization(amr)

    # print("\namr:\n", amr)
    if graph_type == "amr":
        g_directed, amr_obj = build_graph_from_amr(amr, verbose)
    elif graph_type in ["amr_verbnet", "verbnet"]:
        g_directed, amr_obj = build_semantic_graph(
            amr, grounded_stmt, semantic_calc, verbose)

    g_undirected = g_directed.to_undirected()
    path_cache = dict()

    for rel in pattern_dict:
        # print("\nrel:", rel)
        # print("pattern_dict[rel]:")
        # print(pattern_dict[rel].items())
        # print("\ntop k:")
        # print(pattern_dict[rel].most_common(top_k_patterns))
        # input()
        patterns = pattern_dict[rel].most_common(top_k_patterns)
        for pattern, freq in patterns:
            if len(pattern) == 0:
                continue

            # print("pattern:", pattern)
            # pattern = tuple([':ARG0', 'carry-01', ':ARG1', 'and', ':op2'])
            # paths = query_paths(g_undirected, cutoff=len(pattern))
            cutoff = int((len(pattern) - 1) / 2 + 1)
            if cutoff in path_cache:
                path2node_pairs = path_cache[cutoff]
            else:
                path2node_pairs = query_paths(g_undirected, cutoff)
                path_cache[cutoff] = path2node_pairs

            for path in path2node_pairs:
                if path != pattern:
                    continue
                if len(path) <= 1:
                    continue

                node_pairs = path2node_pairs[path]
                print("\npath:", path)
                print("node_pairs:", node_pairs)
                for node_pair in node_pairs:
                    print("node_pair:", node_pair)
                    subj_node, obj_node = node_pair

                    if graph_type == "verbnet":
                        subj_desc = [subj_node]
                        obj_desc = [obj_node]
                    else:
                        subj_desc = get_descendant_leaf_nodes(g_directed, subj_node)
                        obj_desc = get_descendant_leaf_nodes(g_directed, obj_node)

                    if len(subj_desc) == 0:
                        subj = get_token_by_node(subj_node, amr_obj)
                    else:
                        subj_tokens = set([get_token_by_node(n, amr_obj)
                                           for n in subj_desc])
                        subj_tokens = set([tok for tok in subj_tokens if tok is not None])
                        subj = find_text_span(amr_tokens, subj_tokens)

                    if len(obj_desc) == 0:
                        obj = get_token_by_node(obj_node, amr_obj)
                    else:
                        obj_tokens = set([get_token_by_node(n, amr_obj)
                                          for n in obj_desc])
                        obj_tokens = set([tok for tok in obj_tokens if tok is not None])
                        obj = find_text_span(amr_tokens, obj_tokens)

                    print("subj:", subj)
                    print("obj:", obj)
                    input()
                    if subj is not None and obj is not None:
                        if True:
                            visualize_semantic_graph(
                                g_directed, graph_name="semantic_graph".format(),
                                out_dir="./test-output/")

                            print("\namr:")
                            print(amr)
                            print("\npath:", path)
                            print("\nnode_pair:", node_pair)
                            print("subj_desc:", subj_desc)
                            print("obj_desc:", obj_desc)
                            print("subj:", subj)
                            print("obj:", obj)
                            input()

                        print("\ntriple:", (subj, rel, obj))
                        triples.add((subj, rel, obj))
    return triples


def extract_pattern(text, triple, graph_type="amr", amr=None, verbose=False):
    """
    Extract the path pattern between the node corresponding to the subject
    and the node corresponding to the object, in the input triple
    :param text: the input text
    :param triple: a triple in the format of (subj, rel, obj)
    :param graph_type: the type of graph for KG triple induction
        with values ["amr", "amr_verbnet", "verbnet"]
    :param amr: amr parse from cache
    :param verbose:
    :return: a set of patterns
    """
    if verbose:
        print("\ntext:", text)
        print("triple:", triple)
        print("graph_type:", graph_type)
        
    all_patterns = set()
    parse = ground_text_to_verbnet(text, amr=amr, verbose=verbose)
    sentence_parses = parse["sentence_parses"]

    for i in range(len(sentence_parses)):
        # if verbose:
        #     print_enhanced_amr(sentence_parses[i])

        list_grounded_stmt, list_semantic_calc = induce_unique_groundings(
            grounded_stmt=sentence_parses[i]["grounded_stmt"],
            semantic_calc=sentence_parses[i]["sem_cal"])

        for grounded_stmt, semantic_calc in zip(
                list_grounded_stmt, list_semantic_calc):
            graph, amr_obj = build_semantic_graph(
                amr=sentence_parses[i]["amr"],
                grounded_stmt=grounded_stmt,
                semantic_calculus=semantic_calc)

            def filter_edge_func(n1, n2):
                return graph[n1][n2]["source"] != "amr"

            if graph_type == "verbnet":
                # prune AMR edges
                pruned_graph = nx.subgraph_view(graph, filter_edge=filter_edge_func)
                pattern = extract_pattern_from_graph(pruned_graph, amr_obj, triple, verbose=verbose)
            else:
                pattern = extract_pattern_from_graph(graph, amr_obj, triple, verbose=verbose)

            if pattern is not None:
                all_patterns.add(pattern)

    if verbose:
        print("\nall_patterns:", all_patterns)
        input()
    return all_patterns


def extract_pattern_from_graph(g_directed, amr_obj, triple, verbose=False):
    """
    Extract the path pattern between the node corresponding to the subject
    and the node corresponding to the object, in the input triple
    :param g_directed: the directed graph constructed from parse
    :param amr_obj: the amr object that contains the mappings needed
    :param triple: a triple in the format of (subj, rel, obj)
    :param verbose:
    :return:
    """
    g_undirected = g_directed.to_undirected()

    subj, pred, obj = triple
    subj_tokens = subj.lower().split()
    obj_tokens = obj.lower().split()
    if verbose:
        print("\ntoken2node_id:", amr_obj["token2node_id"])
        print("\nsubj_tokens:", subj_tokens)
        print("\nobj_tokens:", obj_tokens)

    subj_leaf_nodes = map_tokens_to_nodes(subj_tokens, amr_obj["token2node_id"])
    obj_leaf_nodes = map_tokens_to_nodes(obj_tokens, amr_obj["token2node_id"])
    if verbose:
        print("\nsubj_leaf_nodes:", subj_leaf_nodes)
        print("\nobj_leaf_nodes:", obj_leaf_nodes)

    if len(subj_leaf_nodes) == 0 or len(obj_leaf_nodes) == 0:
        if verbose:
            print("len(subj_leaf_nodes) == 0 or len(obj_leaf_nodes) == 0")
        return None

    try:
        subj_ancestor = get_lowest_common_ancestor(g_directed, subj_leaf_nodes)
        obj_ancestor = get_lowest_common_ancestor(g_directed, obj_leaf_nodes)
    except nx.exception.NetworkXError:
        # LCA only defined on directed acyclic graphs.
        if verbose:
            print("LCA only defined on directed acyclic graphs.")
        return None

    if subj_ancestor is None or obj_ancestor is None:
        return None

    try:
        path = shortest_path(g_undirected, subj_ancestor, obj_ancestor)
    except nx.exception.NetworkXNoPath:
        if verbose:
            print("No path between {} and {}".format(subj_ancestor, obj_ancestor))
        return None

    if verbose:
        print("\nsubj_ancestor:", subj_ancestor)
        print("\nobj_ancestor:", obj_ancestor)
        print("\npath:", path)

    labeled_path = convert_to_labeled_path(g_undirected, path)
    if verbose:
        print("labeled_path:", labeled_path)
        # input()
    return tuple(labeled_path)


def convert_to_labeled_path(graph, path):
    """
    Convert the path of node id's into a path with node labels and edge labels
    :param graph: the graph
    :param path: a path represented by node id's
    :return: a labeled path
    """
    # print("\npath:", path)
    labeled_path = []
    for idx, node in enumerate(path):
        if idx < len(path) - 1:
            src_idx = idx
            tgt_idx = idx + 1
            edge_label = graph.get_edge_data(path[src_idx], path[tgt_idx])["label"]
            labeled_path.append(edge_label)
            if tgt_idx < len(path) - 1:
                # print("path[tgt_idx]:", path[tgt_idx])
                # print("node data:", graph.nodes(data=True)[path[tgt_idx]])
                labeled_path.append(graph.nodes(data=True)[path[tgt_idx]]["label"])
    # print("labeled_path:", labeled_path)
    return labeled_path


def is_extractable(text, triple):
    """
    Check if a triple can be extracted from text by examing the text spans
    of subject and object.
    :param text: the text to extract triple from
    :param triple: the target triple to extract
    :return:
    """
    subj, pred, obj = triple
    text_tokens = word_tokenize(text.lower())

    subj_spans = []
    obj_spans = []
    subj_tokens = word_tokenize(subj.lower())
    obj_tokens = word_tokenize(obj.lower())

    subj_matched = False
    obj_matched = False
    for idx, tok in enumerate(text_tokens):
        cur_subj_idx = len(subj_spans)
        cur_obj_idx = len(obj_spans)
        if cur_subj_idx < len(subj_tokens) and tok == subj_tokens[cur_subj_idx]:
            subj_spans.append(idx)
            if len(subj_spans) == len(subj_tokens):
                subj_matched = True
        else:
            subj_spans = []

        if cur_obj_idx < len(obj_tokens) and tok == obj_tokens[cur_obj_idx]:
            obj_spans.append(idx)
            if len(obj_spans) == len(obj_tokens):
                obj_matched = True
        else:
            obj_spans = []

    extractable = subj_matched and obj_matched
    return extractable


def apply_path_patterns(data, pattern_file_path, output_file_path,
                        graph_type, top_k_patterns=5, amr_cache_path=None,
                        start_idx=None, verbose=False):
    """
    Apply path patterns to induce KG triples from text.
    :param data: a list of raw samples
    :param pattern_file_path: the path to the pattern file
    :param output_dir: the output directory for saving induced triples
    :param amr_cache_path: the path to the AMR parse for the datas
    :param verbose:
    :return:
    """
    # save patterns to file
    with open(pattern_file_path, "rb") as file_obj:
        pattern_dict = pickle.load(file_obj)
    print("Loaded patterns from {}".format(pattern_file_path))

    for rel in pattern_dict:
        patterns = pattern_dict[rel]
        print("\nrel:", rel)
        print(patterns.most_common(10))
    input()

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    amr_cache = None
    if amr_cache_path is not None:
        amr_cache = load_amr_cache(amr_cache_path)

    f_out = open(output_file_path, "a")
    try:
        for sample_idx, sentences in enumerate(sample_generator(
                data, extractable_only=False, verbose=verbose)):
            if sample_idx < start_idx:
                continue

            print("sample_idx:", sample_idx)
            sample = data[sample_idx]
            if verbose:
                print_sample(sample)

            all_triples = set()

            if verbose:
                print("\nsentences:")
                print(sentences)

            for sent_idx, sent in enumerate(sentences):
                if verbose:
                    print("\nsent:", sent)

                if amr_cache is not None:
                    assert sent == amr_cache[sample_idx][sent_idx]["sent"]
                    amr = amr_cache[sample_idx][sent_idx]["amr"]
                else:
                    amr = amr_client.get_amr(sent)

                triples = induce_kg_triples(sent, pattern_dict, top_k_patterns,
                                            graph_type, amr=amr, verbose=verbose)
                all_triples.update(triples)

            # print("all_triples:", list(all_triples))
            # input()
            result = {
                "idx": sample_idx,
                "pred": list(all_triples),
                "true": sample["state"]["graph"]
            }
            f_out.write(json.dumps(result))
            f_out.write("\n")
    except Exception as e:
        print("Exception:", e)
        f_out.close()
        raise e
    f_out.close()
    print("Triple induction DONE.")


def load_amr_cache(path):
    """
    Load AMR cache from file.
    :param path: the path to the cache file
    :return:
    """
    amr_cache = dict()
    with open(path, "r") as f:
        for line in f:
            sample_idx, amr_str = line.strip().split("\t")
            sentences = json.loads(amr_str)
            amr_cache[int(sample_idx)] = sentences
    return amr_cache


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
                triple = (subj.lower(), rel.lower(), obj.lower())
                pred_triples.add(triple)
                metric.pred_triples_by_rel[rel].add(triple)
                overall_metric.pred_triples_by_rel[rel].add(triple)

            true_triples = set()
            for subj, rel, obj in data["true"]:
                triple = (subj.lower(), rel.lower(), obj.lower())
                true_triples.add(triple)
                metric.rel_counter[rel.lower()] += 1
                overall_metric.rel_counter[rel.lower()] += 1
                metric.true_triples_by_rel[rel].add(triple)
                overall_metric.true_triples_by_rel[rel].add(triple)

            # compute sample-wise scores
            true_pred_triples = pred_triples.intersection(true_triples)
            # compute precision
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
                    graph = build_semantic_graph(
                        amr=res["amr_parse"][i]["amr"],
                        grounded_stmt=grounded_stmt,
                        semantic_calculus=semantic_calc)

                    visualize_semantic_graph(
                        graph, graph_name="semantic_graph_{}".format(graph_idx),
                        out_dir="./test-output/")
                    graph_idx += 1
                print("visualize_semantic_graph DONE.")
                input()


def mine_path_patterns(data, output_file_path, graph_type="amr",
                       amr_cache_path=None, sample_size=None, verbose=False):
    """
    Mine path patterns between subject and object of a triple
    :param data: a list of raw samples
    :param output_file_path: the path of the output pattern file
    :param graph_type: the type of semantic graph to use
    :param amr_cache_path: the path to the cache of AMR parse for the samples
    :param sample_size: the size of samples for sanity check
    :param verbose:
    :return:
    """
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pattern_dict = defaultdict(Counter)

    amr_cache = None
    if amr_cache_path is not None:
        amr_cache = load_amr_cache(amr_cache_path)

    for sample_idx, sentences in enumerate(tqdm(
            sample_generator(data, extractable_only=True, verbose=verbose))):
        sample = data[sample_idx]

        if "graph" not in sample["state"] or \
                len(sample["state"]["graph"]) == 0:
            continue

        # if verbose:
        #     print_sample(sample)

        if verbose:
            print("\nsentences:")
            print(sentences)

        for sent_idx, sent in enumerate(sentences):
            if verbose:
                print("\nsent:", sent)

            extractable_triples = []
            for triple in sample["state"]["graph"]:
                subj, pred, obj = triple
                if len(subj.strip()) == 0 or len(pred.strip()) == 0 or len(obj.strip()) == 0:
                    continue

                if subj.strip().lower() == obj.strip().lower():
                    continue

                if is_extractable(sent, triple):
                    extractable_triples.append(triple)
                    if amr_cache is not None:
                        assert sent == amr_cache[sample_idx][sent_idx]["sent"]
                        amr = amr_cache[sample_idx][sent_idx]["amr"]
                    else:
                        amr = amr_client.get_amr(sent)

                    try:
                        patterns = extract_pattern(sent, triple, graph_type=graph_type,
                                                   amr=amr, verbose=verbose)
                    except Exception as e:
                        print("Exception:", e)
                        patterns = extract_pattern(sent, triple, graph_type=graph_type,
                                                   amr=amr, verbose=True)
                        # raise e
                        input()
                        continue

                    if patterns is None or len(patterns) == 0:
                        continue

                    for pattern in patterns:
                        pattern_dict[pred][pattern] += 1

                    if True:
                        print("\nsent:", sent)
                        print("\ntriple:", triple)
                        print("\npattern:", pattern)
                        # input()
                # else:
                    # print("Invalid triple:", triple)
                    # input()

            if verbose and len(extractable_triples) > 0:
                print(extractable_triples)

        # if len(pattern_dict) > 0:
        #     break

    for rel in pattern_dict:
        patterns = pattern_dict[rel]
        print("\nrel:", rel)
        print(patterns.most_common(10))

    # save patterns to file
    with open(output_file_path, "wb") as file_obj:
        pickle.dump(pattern_dict, file_obj)
    print("\nWritten patterns to {}".format(output_file_path))


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

        if verbose:
            print_sample(sample)

        text = get_observation_text_from_sample(sample)
        sentences = sent_tokenize(text)
        if verbose:
            print("\nsentences:")
            print(sentences)

        if not extractable_only:
            yield sentences
            continue

        processed_sentences = []
        for sent in sentences:
            if "graph" not in sample["state"]:
                continue

            sent = sent.replace("\n\n", ": ").replace("\n", " ")
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


def build_amr_parse_cache(data, output_path, start_idx=0,
                          extractable_only=True, verbose=False):
    """
    Build a cache of AMR parse for the training/test data.
    :param data: the list of samples for AMR parsing
    :param output_path: the output path of AMR parse cache
    :param start_idx: the index of sample to start with
    :param extractable_only: if a sentence to parse needs to be mapped
        to at least one triple in the KG
    :param verbose:
    :return:
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = open(output_path, "a")
    for sample_idx, sentences in enumerate(tqdm(sample_generator(
            data, extractable_only=extractable_only, verbose=verbose))):
        if sample_idx < start_idx:
            continue

        sentence_parses = []
        for text in sentences:
            amr = amr_client.get_amr(text)
            sentence_parses.append({
                "sent": text,
                "amr": amr
            })
            print("\nsample_idx:", sample_idx)
            print("text:", text)
            print("amr:\n", amr)
        f.write(str(sample_idx))
        f.write("\t")
        f.write(json.dumps(sentence_parses))
        f.write("\n")
    f.close()
    print("AMR cache DONE.")


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
    parser.add_argument('--check_extractable_kg_triples', action='store_true', help="check_extractable_kg_triples")
    parser.add_argument('--mine_path_patterns', action='store_true', help="mine_path_patterns")
    parser.add_argument('--path_pattern_source', type=str, default=None, help="path_pattern_source")
    parser.add_argument('--pattern_file_path', type=str, default=None, help="pattern_file_path")
    parser.add_argument('--output_file_path', type=str, default=None, help="output_file_path")
    parser.add_argument('--apply_path_patterns', action='store_true', help="apply_path_patterns")
    parser.add_argument('--graph_type', type=str, default=None, choices=["amr", "amr_verbnet", "verbnet"],
                        help="graph_type")
    parser.add_argument('--top_k_patterns', type=int, default=20, help="top_k_patterns")
    parser.add_argument('--build_amr_parse_cache', action='store_true', help="build_amr_parse_cache")
    parser.add_argument('--compute_metrics', action='store_true', help="compute_metrics")
    parser.add_argument('--amr_cache_path', type=str, default=None, help="amr_cache_path")
    parser.add_argument('--triple_file_path', type=str, default=None, help="triple_file_path")
    parser.add_argument('--sample_start_idx', type=int, default=0, help="sample_start_idx")
    parser.add_argument('--split_type', type=str, choices=["train", "test"],
                        default="train", help="split_type")
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
    elif args.mine_path_patterns:
        mine_path_patterns(train_data,
                           output_file_path=args.output_file_path,
                           graph_type=args.graph_type,
                           amr_cache_path=args.amr_cache_path,
                           sample_size=None, verbose=args.verbose)
    elif args.apply_path_patterns:
        apply_path_patterns(test_data, pattern_file_path=args.pattern_file_path,
                            output_file_path=args.output_file_path, graph_type=args.graph_type,
                            top_k_patterns=args.top_k_patterns, amr_cache_path=args.amr_cache_path,
                            start_idx=args.sample_start_idx, verbose=args.verbose)
    elif args.build_amr_parse_cache:
        if args.split_type == "train":
            build_amr_parse_cache(train_data, "./data/JerichoWorld/train_amr.json",
                                  extractable_only=True,
                                  start_idx=args.sample_start_idx)
        elif args.split_type == "test":
            build_amr_parse_cache(test_data, "./data/JerichoWorld/test_amr.json",
                                  extractable_only=False,
                                  start_idx=args.sample_start_idx)
    elif args.compute_metrics:
        compute_metrics(test_data, args.triple_file_path)
    print("DONE.")

