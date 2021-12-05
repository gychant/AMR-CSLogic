"""Module for the path pattern based algorithm for knowledge graph prediction"""

import os
import json
import pickle
from collections import Counter, defaultdict
from itertools import combinations, permutations

import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.simple_paths import all_simple_paths
from tqdm import tqdm

from nltk.stem.porter import PorterStemmer

from amr_verbnet_semantics.core.amr_verbnet_enhance import \
    build_graph_from_amr, \
    build_semantic_graph, \
    ground_text_to_verbnet
from amr_verbnet_semantics.utils.amr_util import \
    load_amr_cache, \
    read_tokenization
from amr_verbnet_semantics.utils.text_util import \
    is_extractable


stemmer = PorterStemmer()


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


def query_node_pairs_by_path_pattern_set(graph, pattern_set):
    """
    Get all node pairs of the graph and search paths for each that match
    elements in the pattern set.
    :param graph: A undirected graph from AMR parse
    :param pattern_set: a path pattern set, each in tuple
    :return: A dictionary with pattern as key and set of node pair tuple as values
    """
    pattern2node_pairs = defaultdict(set)
    pattern_lens = [len(p) for p in pattern_set]
    min_cutoff = int((min(pattern_lens) - 1) / 2 + 1)
    max_cutoff = int((max(pattern_lens) - 1) / 2 + 1)

    nodes = graph.nodes()
    for node_pair in list(combinations(nodes, 2)):
        src, tgt = node_pair
        for cutoff in range(min_cutoff, max_cutoff + 1):
            node_pair_paths = all_simple_paths(graph, src, tgt, cutoff)
            for path in node_pair_paths:
                if len(path) < cutoff:
                    continue
                labeled_path = tuple(convert_to_labeled_path(graph, path))
                if labeled_path in pattern_set:
                    pattern2node_pairs[labeled_path].add(node_pair)
    return pattern2node_pairs


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
    :param text: the input text
    :param amr: the AMR parse
    :param pattern_dict: a dictionary storing path patterns for
        all prefined relations
    :param top_k_patterns: top k patterns to apply
    :param graph_type: the type of graph for KG triple induction
        with values ["amr", "amr_verbnet", "verbnet"]
    :param verbose:
    :return:
    """
    all_triples = defaultdict(set)

    if graph_type == "amr":
        triples = induce_kg_triples_from_grounding(
            amr, pattern_dict, top_k_patterns,
            graph_type=graph_type, verbose=verbose)
        all_triples.update(triples)
    else:
        parse = ground_text_to_verbnet(text, amr=amr, verbose=verbose)
        sentence_parses = parse["sentence_parses"]

        for i in range(len(sentence_parses)):
            list_grounded_stmt, list_semantic_calc = induce_unique_groundings(
                grounded_stmt=sentence_parses[i]["grounded_stmt"],
                semantic_calc=sentence_parses[i]["sem_cal"])

            for grounded_stmt, semantic_calc in zip(
                    list_grounded_stmt, list_semantic_calc):
                triples = induce_kg_triples_from_grounding(
                    sentence_parses[i]["amr"], pattern_dict, top_k_patterns,
                    grounded_stmt, semantic_calc, graph_type, verbose=verbose)
                all_triples.update(triples)
    return all_triples


def induce_kg_triples_from_grounding(amr, pattern_dict, top_k_patterns,
                                     grounded_stmt=None, semantic_calc=None,
                                     graph_type="amr", verbose=False):
    """
    Induce triples from text using given patterns
    :param amr: the AMR parse
    :param pattern_dict: a dictionary storing path patterns for
        all prefined relations
    :param top_k_patterns: top k patterns to apply
    :param grounded_stmt: grounded statement
    :param semantic_calc: semantic calculus
    :param graph_type: the type of graph for KG triple induction
            with values ["amr", "amr_verbnet", "verbnet"]
    :param verbose:
    :return: a dictionary with triples as keys and their corresponding
            pattern as values
    """
    triples = defaultdict(set)
    amr_tokens = read_tokenization(amr)

    # print("\namr:\n", amr)
    if graph_type == "amr":
        g_directed, amr_obj = build_graph_from_amr(amr, verbose)
    elif graph_type in ["amr_verbnet", "verbnet"]:
        g_directed, amr_obj = build_semantic_graph(
            amr, grounded_stmt, semantic_calc, verbose)

    g_undirected = g_directed.to_undirected()
    # path_cache = dict()

    for rel in pattern_dict:
        # print("\nrel:", rel)
        # print("pattern_dict[rel]:")
        # print(pattern_dict[rel].items())
        # print("\ntop k:")
        # print(pattern_dict[rel].most_common(top_k_patterns))
        # input()
        patterns = pattern_dict[rel].most_common(top_k_patterns)
        # print("\nNum of patterns:", len(patterns))
        pattern_set = set([pattern for pattern, freq in patterns])

        # start_time = time.time()
        pattern2node_pairs = query_node_pairs_by_path_pattern_set(
            g_undirected, pattern_set)
        # print("cost time:", time.time() - start_time)

        for pattern in pattern2node_pairs:
            """
            if pattern in path_cache:
                node_pairs = path_cache[pattern]
            else:
                path_cache[pattern] = node_pairs
            """
            node_pairs = pattern2node_pairs[pattern]
            # print("Num of node_pairs:", len(node_pairs))
            for node_pair in node_pairs:
                # print("node_pair:", node_pair)
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

                # print("subj:", subj)
                # print("obj:", obj)
                # input()
                if subj is not None and obj is not None and subj != obj:
                    if False:
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
                    print("pattern:", pattern)
                    # print("node_pairs:", node_pairs)
                    triples[(subj, rel, obj)].add(pattern)
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

    if graph_type == "amr":
        graph, amr_obj = build_semantic_graph(amr)
        pattern = extract_pattern_from_graph(graph, amr_obj, triple, verbose=verbose)
        if pattern is not None:
            all_patterns.add(pattern)
    else:
        parse = ground_text_to_verbnet(text, amr=amr, verbose=verbose)
        sentence_parses = parse["sentence_parses"]

        for i in range(len(sentence_parses)):
            list_grounded_stmt, list_semantic_calc = induce_unique_groundings(
                grounded_stmt=sentence_parses[i]["grounded_stmt"],
                semantic_calc=sentence_parses[i]["sem_cal"])

            for grounded_stmt, semantic_calc in zip(
                    list_grounded_stmt, list_semantic_calc):
                graph, amr_obj = build_semantic_graph(
                    amr=sentence_parses[i]["amr"],
                    grounded_stmt=grounded_stmt,
                    semantic_calculus=semantic_calc)

                def filter_amr_edges(n1, n2):
                    return graph[n1][n2]["source"] != "amr"

                if graph_type == "verbnet":
                    # prune amr edges
                    pruned_graph = nx.subgraph_view(graph, filter_edge=filter_amr_edges)
                    pattern = extract_pattern_from_graph(pruned_graph, amr_obj, triple, verbose=verbose)
                elif graph_type == "amr_verbnet":
                    pattern = extract_pattern_from_graph(graph, amr_obj, triple, verbose=verbose)
                else:
                    raise Exception("Invalid graph type: {}".format(graph_type))

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
            if edge_label.startswith(":op") and edge_label[-1].isdigit():
                edge_label = ":op"
            labeled_path.append(edge_label)

            if tgt_idx < len(path) - 1:
                # print("path[tgt_idx]:", path[tgt_idx])
                # print("node data:", graph.nodes(data=True)[path[tgt_idx]])
                labeled_path.append(graph.nodes(data=True)[path[tgt_idx]]["label"])
    # print("labeled_path:", labeled_path)
    return labeled_path


def mine_path_patterns(data, sample_generator, output_file_path, graph_type="amr",
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
    sentence_triple_cache = set()

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
                cache_key = tuple([sent, tuple(triple)])
                if cache_key in sentence_triple_cache:
                    continue
                else:
                    sentence_triple_cache.add(cache_key)

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
                        print("\n\n\nsent:", sent)
                        print("triple:", triple)
                        print("pattern:", pattern)
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


def remove_path_op_num(path):
    new_path = list()
    for elem in path:
        if elem.startswith(":op") and elem[-1].isdigit():
            new_path.append(":op")
        else:
            new_path.append(elem)
    return tuple(new_path)


def remove_op_num(pattern_dict):
    new_pattern_dict = defaultdict(Counter)
    for rel in pattern_dict:
        patterns = pattern_dict[rel]
        for path in patterns:
            new_path = remove_path_op_num(path)
            cnt = patterns[path]
            new_pattern_dict[rel][new_path] += cnt
    return new_pattern_dict


def apply_path_patterns(data, sample_generator, pattern_file_path, output_file_path,
                        graph_type, top_k_patterns=5, amr_cache_path=None,
                        start_idx=None, debug=False, verbose=False):
    """
    Apply path patterns to induce KG triples from text.
    :param data: a list of raw samples
    :param pattern_file_path: the path to the pattern file
    :param output_dir: the output directory for saving induced triples
    :param amr_cache_path: the path to the AMR parse for the datas
    :param verbose:
    :return:
    """
    # read patterns from file
    with open(pattern_file_path, "rb") as file_obj:
        pattern_dict = pickle.load(file_obj)
    print("Loaded patterns from {}".format(pattern_file_path))

    # remove the number suffix of the :op relation from the paths
    pattern_dict = remove_op_num(pattern_dict)

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
            if debug:
                output_dir = "./test-debug/sample_{}".format(sample_idx)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                f_debug_info = open(os.path.join(output_dir, "sample_{}.txt".format(sample_idx)), "w")

            sample = data[sample_idx]
            if verbose:
                print_sample(sample)

            all_triples = defaultdict(set)

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

                if debug:
                    graph, amr_obj = build_semantic_graph(amr=amr)
                    visualize_semantic_graph(
                        graph, graph_name="semantic_graph_sent_{}".format(sent_idx),
                        out_dir="./test-debug/sample_{}".format(sample_idx))
                    f_debug_info.write("sent_{}: ".format(sent_idx))
                    f_debug_info.write(sent.strip())
                    f_debug_info.write("\n\n")

                triples = induce_kg_triples(sent, pattern_dict, top_k_patterns,
                                            graph_type, amr=amr, verbose=verbose)
                all_triples.update(triples)

            # print("all_triples:", list(all_triples))
            # input()
            result = {
                "idx": sample_idx,
                "pred": list(all_triples.keys()),
                "true": sample["state"]["graph"]
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

