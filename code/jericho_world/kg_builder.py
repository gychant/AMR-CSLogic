"""Build knowledge graphs of the game using AMR-VerbNet semantics"""
import os
import json
import requests
import pickle
import penman
from tqdm import tqdm, trange

import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.simple_paths import all_simple_paths

from pprint import pprint
from collections import Counter, defaultdict
from itertools import combinations, permutations
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from code.service.amr import amr_client
from code.core.amr_verbnet_enhance import generate_enhanced_amr, visualize_enhanced_amr

import random
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


def check_samples(data, sample_size=10):
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
        text = "The dresser is made out of maple carefully finished with Danish oil."
        # text = "You are carrying : a bronze - hilted dagger a clay ocarina armor and silks ( worn ) ."
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

                visualize_enhanced_amr(graph, amr_only=False, out_dir="./test-output/")
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


def map_tokens_to_nodes(tokens, token2node_id):
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


def build_graph_from_amr(amr, verbose=False):
    """
    Build undirected and directed networkx graphs from AMR parse.
    :param amr: the AMR parse
    :param verbose:
    :return:
    """
    amr_graph = penman.decode(amr)
    g_directed = nx.DiGraph()
    g_undirected = nx.Graph()

    node_dict = dict()

    # read AMR alignment
    token2node_id, node_idx2node_id, node_id2node_idx = read_node_alignment(amr)
    if verbose:
        print("\ntoken2node_id:", token2node_id)
        print("\nnode_idx2node_id:", node_idx2node_id)
        print("\nnode_id2node_idx:", node_id2node_idx)

    # construct graph from AMRs
    for inst in amr_graph.instances():
        if verbose:
            print("inst:", inst)

        g_directed.add_node(inst.source, label=inst.target, source="amr")
        g_undirected.add_node(inst.source, label=inst.target, source="amr")
        node_dict[inst.source] = inst.target

        for attr in amr_graph.attributes(inst.source):
            if verbose:
                print("attr:", attr)

            if attr.target.startswith("\"") and attr.target.endswith("\""):
                attr_constant = attr.target[1:-1]
            else:
                attr_constant = attr.target

            # use the parent node of attributes for pattern mining
            token2node_id[attr_constant.lower()] = attr.source

    for edge in amr_graph.edges():
        if verbose:
            print("edge:", edge)

        g_directed.add_edge(edge.source, edge.target, label=edge.role, source="amr")
        g_undirected.add_edge(edge.source, edge.target, label=edge.role, source="amr")

    if verbose:
        print("\nnode_dict:", node_dict)
        print("\ntoken2node_id post:", token2node_id)
    return g_directed, g_undirected, token2node_id


def query_paths(graph, cutoff):
    """
    Get all node pairs of the graph and search paths for each with the specified
    cutoff.
    :param graph: A undirected graph from AMR parse
    :return: A dictionary with path as key and set of node pair tuple as values
    """
    paths = defaultdict(set)
    print("\nnodes:")
    print(graph.nodes())
    print("cutoff:", cutoff)

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
    for perm_tokens in permutations(target_token_set):
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


def induce_kg_triples(text, pattern_dict, verbose=True):
    """
    Induce triples from text using given patterns
    :param text:
    :param patterns:
    :return:
    """
    triples = set()
    amr = amr_client.get_amr(text)

    if verbose:
        print("\ntext:", text)
        print("\namr:")
        print(amr)

    amr_tokens = read_tokenization(amr)
    g_directed, g_undirected, token2node_id = build_graph_from_amr(amr)

    for rel in pattern_dict:
        patterns = pattern_dict[rel]
        for pattern in patterns:
            # pattern = tuple([':ARG0', 'carry-01', ':ARG1', 'and', ':op2'])
            # paths = query_paths(g_undirected, cutoff=len(pattern))
            cutoff = int((len(pattern) - 1) / 2 + 1)
            path2node_pairs = query_paths(g_undirected, cutoff)

            for path in path2node_pairs:
                print("\npath:", path)
                print("pattern:", pattern)
                input()
                if path != pattern:
                    continue

                for node_pair in path2node_pairs[path]:
                    subj_node, obj_node = node_pair
                    print("\nnode_pair:", node_pair)
                    subj_desc = nx.descendants(g_directed, subj_node)
                    obj_desc = nx.descendants(g_directed, obj_node)
                    print("subj_desc:", subj_desc)
                    print("obj_desc:", obj_desc)
                    if len(subj_desc) == 0:
                        subj = g_directed.nodes(data=True)[subj_node]["label"]
                    else:
                        subj_tokens = set([g_directed.nodes(data=True)[n]["label"]
                                           for n in subj_desc])
                        subj = find_text_span(amr_tokens, subj_tokens)

                    if len(obj_desc) == 0:
                        obj = g_directed.nodes(data=True)[obj_node]["label"]
                    else:
                        obj_tokens = set([g_directed.nodes(data=True)[n]["label"]
                                          for n in obj_desc])
                        obj = find_text_span(amr_tokens, obj_tokens)
                    print("subj:", subj)
                    print("obj:", obj)
                    input()
                    triples.add((subj, rel, obj))
    return triples


def extract_pattern(text, triple, verbose=False):
    amr = amr_client.get_amr(text)

    if verbose:
        print("\ntext:", text)
        print("\ntriple:", triple)
        print("\namr:")
        print(amr)

    g_directed, g_undirected, token2node_id = build_graph_from_amr(amr, verbose)

    subj, pred, obj = triple
    subj_tokens = subj.lower().split()
    obj_tokens = obj.lower().split()
    if verbose:
        print("\nsubj_tokens:", subj_tokens)
        print("\nobj_tokens:", obj_tokens)

    subj_leaf_nodes = map_tokens_to_nodes(subj_tokens, token2node_id)
    obj_leaf_nodes = map_tokens_to_nodes(obj_tokens, token2node_id)
    if verbose:
        print("\nsubj_leaf_nodes:", subj_leaf_nodes)
        print("\nobj_leaf_nodes:", obj_leaf_nodes)

    if len(subj_leaf_nodes) == 0 or len(obj_leaf_nodes) == 0:
        return None

    try:
        subj_ancestor = get_lowest_common_ancestor(g_directed, subj_leaf_nodes)
        obj_ancestor = get_lowest_common_ancestor(g_directed, obj_leaf_nodes)
    except nx.exception.NetworkXError:
        # LCA only defined on directed acyclic graphs.
        return None

    if subj_ancestor is None or obj_ancestor is None:
        return None

    path = shortest_path(g_undirected, subj_ancestor, obj_ancestor)
    if verbose:
        print("\nsubj_ancestor:", subj_ancestor)
        print("\nobj_ancestor:", obj_ancestor)
        print("\npath:", path)

    labeled_path = convert_to_labeled_path(g_undirected, path)
    if verbose:
        print("labeled_path:", labeled_path)
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
                labeled_path.append(graph.nodes(data=True)[path[tgt_idx]]["label"])
    # print("labeled_path:", labeled_path)
    return labeled_path


def double_quote(json_str):
    return json_str.replace(" ", "").replace("{", "{\"")\
        .replace(":", "\":").replace(",", ",\"").replace("'", "\"")


def read_node_alignment(amr):
    token2node_idx = dict()
    token2node_id = dict()

    for line in amr.split("\n"):
        if line.startswith("# ::tok"):
            # tokenized text
            text = line[len("# ::tok"):-len("<ROOT>")].strip()
            tokens = text.split()
        elif line.startswith("# ::node"):
            node_info = line[len("# ::node"):].strip()
            columns = node_info.split()
            if len(columns) < 3:
                continue

            node_idx, node_label, token_range = columns
            if node_label.startswith("\"") and node_label.endswith("\""):
                # the node is an attribute constant
                continue
            start_tok_idx, end_tok_idx = token_range.split("-")
            token = " ".join(tokens[int(start_tok_idx):int(end_tok_idx)]).lower()
            token2node_idx[token] = node_idx
        elif line.startswith("# ::short"):
            node_idx2node_id = json.loads(double_quote(line[len("# ::short"):].strip()))
            node_id2node_idx = {v: k for k, v in node_idx2node_id.items()}

    for token in token2node_idx:
        token2node_id[token] = node_idx2node_id[token2node_idx[token]]
    return token2node_id, node_idx2node_id, node_id2node_idx


def is_extractable(text, triple):
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


def apply_path_patterns(data, pattern_file_path, output_dir,
                        sample_size=None, verbose=False):
    # save patterns to file
    with open(pattern_file_path, "rb") as file_obj:
        pattern_dict = pickle.load(file_obj)
    print("Loaded patterns from {}".format(pattern_file_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sample_size is not None:
        # Select samples for initial testing
        all_indices = list(range(len(data)))
        random.shuffle(all_indices)
        sample_indices = all_indices[:sample_size]
        index_gen = tqdm((n for n in sample_indices), total=sample_size)
    else:
        index_gen = trange(len(data))

    f_out = open(os.path.join(output_dir, "extracted_triples.jsonl"), "w")
    try:
        for idx in index_gen:
            sample = data[idx]

            if "graph" not in sample["state"] or \
                    len(sample["state"]["graph"]) == 0:
                continue

            if verbose:
                print_sample(sample)

            text = get_observation_text_from_sample(sample)
            sentences = sent_tokenize(text)
            all_triples = set()

            if verbose:
                print("\nsentences:")
                print(sentences)

            for sent in sentences:
                sent = sent.replace("\n\n", ": ")
                sent = sent.replace("\n", " ")
                if verbose:
                    print("\nsent:", sent)

                triples = induce_kg_triples(sent, pattern_dict)
                all_triples.update(triples)

            result = {
                "idx": idx,
                "pred": all_triples,
                "true": sample["state"]["graph"]
            }
            f_out.write(json.dumps(result))
            f_out.write("\n")
    except Exception as e:
        print("Exception:", e)
        f_out.close()
        raise e
    f_out.close()
    print("Triple mining DONE.")


def mine_path_patterns(data, output_dir, sample_size=None, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pattern_dict = defaultdict(Counter)

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

        if "graph" not in sample["state"] or \
                len(sample["state"]["graph"]) == 0:
            continue

        if verbose:
            print_sample(sample)

        text = get_observation_text_from_sample(sample)
        sentences = sent_tokenize(text)
        if verbose:
            print("\nsentences:")
            print(sentences)

        for sent in sentences:
            sent = sent.replace("\n\n", ": ")
            sent = sent.replace("\n", " ")
            if verbose:
                print("\nsent:", sent)
            # concat_text = to_pure_letter_string(sent)
            extractable_triples = []
            for triple in sample["state"]["graph"]:
                subj, pred, obj = triple
                if len(subj.strip()) == 0 or len(pred.strip()) == 0 or len(obj.strip()) == 0:
                    continue

                if subj.strip().lower() == obj.strip().lower():
                    continue

                # if to_pure_letter_string(subj) in concat_text \
                #         and to_pure_letter_string(obj) in concat_text:
                if is_extractable(sent, triple):
                    # print("Valid triple:", triple)
                    # input()
                    extractable_triples.append(triple)
                    try:
                        pattern = extract_pattern(sent, triple, verbose)
                    except Exception as e:
                        print("Exception:", e)
                        continue

                    if pattern is None:
                        continue

                    print("\npred:", pred)
                    print("pattern:", pattern)
                    pattern_dict[pred][pattern] += 1
                    if verbose:
                        print("\nsent:", sent)
                        print("\ntriple:", triple)
                        print("\npattern:", pattern)
                        input()
                # else:
                    # print("Invalid triple:", triple)
                    # input()

            if verbose and len(extractable_triples) > 0:
                print(extractable_triples)

    for rel in pattern_dict:
        patterns = pattern_dict[rel]
        print("\nrel:", rel)
        print(patterns.most_common(10))

    # save patterns to file
    out_path = os.path.join(output_dir, "patterns_train.pkl")
    with open(out_path, "wb") as file_obj:
        pickle.dump(pattern_dict, file_obj)
    print("Written patterns to {}".format(out_path))


def sample_generator(data, sample_size=None, verbose=False):
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

        processed_sentences = []
        for sent in sentences:
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


def build_amr_parse_cache(data, output_path, verbose=False):
    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    f = open(output_path, "w")
    for sample_idx, sentences in enumerate(tqdm(sample_generator(data, verbose=verbose))):
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
    parser.add_argument('--check_extractable_kg_triples', action='store_true', help="check_extractable_kg_triples")
    parser.add_argument('--mine_path_patterns', action='store_true', help="mine_path_patterns")
    parser.add_argument('--apply_path_patterns', action='store_true', help="apply_path_patterns")
    parser.add_argument('--build_amr_parse_cache', action='store_true', help="build_amr_parse_cache")
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
    elif args.check_extractable_kg_triples(train_data):
        check_extractable_kg_triples(train_data)
    elif args.mine_path_patterns:
        # mine_path_patterns(train_data, "./path_output/", sample_size=200, verbose=False)
        mine_path_patterns(train_data, "./path_output/", sample_size=None, verbose=False)
    elif args.apply_path_patterns:
        apply_path_patterns(test_data, pattern_file_path="./path_output/patterns_train.pkl",
                            output_dir="./path_output/", sample_size=500, verbose=True)
    elif args.build_amr_parse_cache:
        build_amr_parse_cache(train_data, "./data/JerichoWorld/train_amr.json")
        build_amr_parse_cache(test_data, "./data/JerichoWorld/test_amr.json")
    print("DONE.")

