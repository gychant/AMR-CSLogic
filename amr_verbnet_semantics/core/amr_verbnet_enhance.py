"""
Core functions that enhance AMR with VerbNet semantics
"""
import os
import copy
import json
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from typing import List
from pprint import pprint
from collections import Counter
import penman
import networkx as nx
import graphviz

from amr_verbnet_semantics.grpc_clients import AMRClientTransformer
from amr_verbnet_semantics.core.stanford_nlp_parse import full_parsing
from amr_verbnet_semantics.utils.format_util import to_json
from amr_verbnet_semantics.core.models import PredicateCalculus
from amr_verbnet_semantics.service.propbank import query_propbank_roles
from amr_verbnet_semantics.service.verbnet import query_semantics
from amr_verbnet_semantics.service.semlink import query_pb_vn_mapping
from amr_verbnet_semantics.utils.amr_util import read_amr_annotation
# from amr_verbnet_semantics.service.ontology import query_pb_vn_mapping
from amr_verbnet_semantics.service.amr import amr_client


def parse_text(text):
    sentences = sent_tokenize(text)
    print("\ntext:\n", text)
    print("\nsentences:\n==>", "\n\n==>".join(sentences))

    print("parsing ...")
    sentence_parses = []
    for idx, sent in enumerate(sentences):
        amr = amr_client.get_amr(sent)
        sentence_parses.append(amr)
    return sentence_parses


def ground_text_to_verbnet(text, amr=None, local_amr_client=None,
                           use_coreference=True, verbose=False):
    sentences = sent_tokenize(text)
    if verbose:
        print("parsing ...")
        print("\ntext:\n", text)
        print("\nsentences:\n==>", "\n\n==>".join(sentences))

    parse = {"coreference": []}
    if use_coreference:
        parse = full_parsing(text, do_coreference=True)
        if verbose:
            print("\ncoreference:\n", parse["coreference"])

    sentence_parses = []
    for idx, sent in enumerate(sentences):
        sent_res = dict()

        if amr is None:
            if local_amr_client:
                amr = local_amr_client.get_amr(sent)
            else:
                amr = amr_client.get_amr(sent)

        g_res = ground_amr(amr, verbose=verbose)
        sent_res["text"] = sent
        sent_res["amr"] = amr
        sent_res.update(g_res)
        sentence_parses.append(sent_res)

    results = {
        "coreference": parse["coreference"],
        "sentence_parses": sentence_parses
    }
    return results


def match_semantics_by_role_set(semantics, amr_role_set, verbose=False):
    raw_vn_role_sets = list(semantics.keys())
    raw_amr_role_set = amr_role_set
    if verbose:
        print("\nraw_vn_role_sets:", raw_vn_role_sets)
        print("\nraw_amr_role_set:", raw_amr_role_set)

    if len(raw_vn_role_sets) > 1:
        # remove common roles first
        common_roles = set(raw_vn_role_sets[0]).intersection(*[set(s) for s in raw_vn_role_sets[1:]])
        vn_role_sets = [set(s).difference(common_roles) for s in raw_vn_role_sets]
        amr_role_set = raw_amr_role_set.difference(common_roles)
        if verbose:
            print("\ncommon_roles:", common_roles)
            print("\nvn_role_sets:", vn_role_sets)
            print("\namr_role_set:", amr_role_set)
        set_diff_sizes = [(idx, len(set(vn_role_set).symmetric_difference(amr_role_set)))
                          for idx, vn_role_set in enumerate(vn_role_sets)]
    else:
        set_diff_sizes = [(idx, len(set(vn_role_set).symmetric_difference(raw_amr_role_set)))
                          for idx, vn_role_set in enumerate(raw_vn_role_sets)]

    min_diff_idx = sorted(set_diff_sizes, key=lambda x: x[1])[0][0]

    return raw_vn_role_sets[min_diff_idx], semantics[raw_vn_role_sets[min_diff_idx]]


def build_role_set_from_mappings(node_name, verbnet_id, arg_map, role_mappings, verbose=False):
    role_set = set()
    vn_class_name = "-".join(verbnet_id.split("-")[1:])
    for src in arg_map:
        if src != node_name:
            continue

        for arg_role in arg_map[src]:
            if arg_role in role_mappings:
                role_info = role_mappings[arg_role]
                for vn_cls_info in role_info:
                    if vn_cls_info["vncls"] == vn_class_name:
                        role_set.add(vn_cls_info["vntheta"].title())

    if verbose:
        print("\nbuild_role_set_from_mappings ...")
        print("arg_map:", arg_map)
        print("role_mappings:", role_mappings)
        print("role_set:", role_set)
    return role_set


def ground_amr(amr, verbose=False):
    g = penman.decode(amr)

    role_mappings = dict()
    semantics = dict()
    pb_vn_mappings = dict()

    raw_amr_cal, arg_map = construct_calculus_from_amr(amr)
    if verbose:
        print("\nraw_amr_cal:", raw_amr_cal)
        print("\narg_map:", arg_map)

    for inst in g.instances():
        node_name = inst.source

        # if it is a propbank frame
        if len(inst.target) > 3 and inst.target[-3] == "-" and inst.target[-2:].isnumeric():
            pb_id = inst.target[:-3] + "." + inst.target[-2:]
            if pb_id not in role_mappings:
                role_map = query_propbank_roles(pb_id)
                if role_map is None:
                    continue
                role_mappings[pb_id] = role_map

            if pb_id not in arg_map:
                continue

            mapping_res = query_pb_vn_mapping(pb_id)
            if mapping_res is not None and pb_id not in semantics:
                # deal with multiple mappings
                pb_vn_mappings[pb_id] = mapping_res
                for mapping in mapping_res:
                    verbnet_id = mapping["mapping"]
                    verbnet_version = mapping["source"]
                    amr_role_set = build_role_set_from_mappings(node_name, verbnet_id, arg_map[pb_id], role_mappings[pb_id])
                    semantics_by_role_set = query_semantics(verbnet_id, verbnet_version)
                    matched_role_set, matched_semantics = match_semantics_by_role_set(semantics_by_role_set, amr_role_set)
                    # print("\nmatched_role_set:", matched_role_set)
                    # print("\nmatched_semantics:", matched_semantics)
                    if pb_id not in semantics:
                        semantics[pb_id] = dict()
                    semantics[pb_id][verbnet_id] = matched_semantics

    if verbose:
        print("\nrole_mappings:", role_mappings)
        print("\nsemantics:", semantics)

    amr_cal = process_and_operator(raw_amr_cal)
    sem_cal = construct_calculus_from_semantics(semantics)
    grounded_stmt = ground_semantics(arg_map, sem_cal, role_mappings)
    unique_grounded_stmt, unique_sem_cal = induce_unique_groundings(grounded_stmt, sem_cal)

    if verbose:
        print("\namr_cal:", amr_cal)
        print("\nsem_cal:", sem_cal)
        print("\ngrounded_stmt:", grounded_stmt)

    results = {
        "pb_vn_mappings": pb_vn_mappings,
        "role_mappings": to_json(role_mappings),
        "amr_cal": to_json(amr_cal),
        "sem_cal": to_json(sem_cal),
        "unique_sem_cal": to_json(unique_sem_cal),
        "grounded_stmt": to_json(grounded_stmt),
        "unique_grounded_stmt": to_json(unique_grounded_stmt),
        "amr_cal_str": str(amr_cal),
        "sem_cal_str": str(sem_cal),
        "unique_sem_cal_str": str(unique_sem_cal),
        "grounded_stmt_str": str(grounded_stmt),
        "unique_grounded_stmt_str": str(unique_grounded_stmt)
    }

    if verbose:
        print("\nresults:")
        print(results)
    return results


def construct_calculus_from_semantics(semantics):
    """
    :param amr: AMR string
    :return: list of PredicateCalculus objects
    """
    results = dict()
    for propbank_id in semantics:
        if propbank_id not in results:
            results[propbank_id] = dict()

        for vn_class in semantics[propbank_id]:
            list_calculus = []
            results[propbank_id][vn_class] = list_calculus

            for event in semantics[propbank_id][vn_class]:
                list_calculus.append(PredicateCalculus(
                    predicate=event["predicate_value"].upper(),
                    arguments=[arg["value"] for arg in event["arguments"]],
                    is_negative=event["is_negative"]))
    # print(results)
    # input()
    return results


def ground_semantics(arg_map, semantic_calc, role_mappings, verbose=False):
    """
    :param arg_map:
    :param semantic_calc:
    :param role_mappings:
    :param verbose:
    :return:
    """
    if verbose:
        print("\narg_map:")
        print(arg_map)
        print("\nsemantic_calc:")
        print(semantic_calc)
        print("\nrole_mappings:")
        pprint(role_mappings)

    results = dict()
    for propbank_id in arg_map:
        if propbank_id not in semantic_calc:
            continue
        if propbank_id not in role_mappings:
            continue

        if propbank_id not in results:
            results[propbank_id] = dict()

        cur_role_mappings = role_mappings[propbank_id]
        for vn_class in semantic_calc[propbank_id]:
            semantic = semantic_calc[propbank_id][vn_class]
            for src in arg_map[propbank_id]:
                cur_calculus = copy.deepcopy(semantic)
                to_add_stmt = []
                for stmt_idx in range(len(cur_calculus)):
                    stmt = cur_calculus[stmt_idx]
                    for arg_idx in range(len(stmt.arguments)):
                        for role in cur_role_mappings:
                            role_info = cur_role_mappings[role]
                            for vn_cls_info in role_info:
                                if stmt.arguments[arg_idx].lower() == vn_cls_info["vntheta"].lower():
                                    # print("role:", role)
                                    # print("arg_map:", arg_map)
                                    if role not in arg_map[propbank_id][src]:
                                        continue

                                    stmt.arguments[arg_idx] = arg_map[propbank_id][src][role]
                                    if "and" in arg_map and arg_map[propbank_id][src][role] in arg_map["and"]:
                                        op_role_dict = arg_map["and"][arg_map[propbank_id][src][role]]
                                        for idx, op_role in enumerate(op_role_dict):
                                            if idx == 0:
                                                stmt.arguments[arg_idx] = op_role_dict[op_role]
                                            else:
                                                stmt_copy = copy.deepcopy(stmt)
                                                stmt_copy.arguments[arg_idx] = op_role_dict[op_role]
                                                to_add_stmt.append(stmt_copy)

                final_calculus = []
                for stmt in (cur_calculus + to_add_stmt):
                    # PATH(during(E), Theme, ?Initial_Location, ?Trajectory, Destination)
                    if stmt.predicate == "PATH":
                        # LOCATION(start(E), Theme, ?Initial_Location)
                        # and LOCATION(end(E), Theme, Destination)
                        theme = stmt.arguments[1]
                        dest = stmt.arguments[4]
                        final_calculus.append(PredicateCalculus("LOCATION", ["start(E)", theme, "?Initial_Location"]))
                        final_calculus.append(PredicateCalculus("LOCATION", ["end(E)", theme, dest]))
                    else:
                        final_calculus.append(stmt)

                if vn_class not in results[propbank_id]:
                    results[propbank_id][vn_class] = []
                results[propbank_id][vn_class].append(final_calculus)

    for pb_id in semantic_calc:
        if pb_id not in arg_map:
            for vn_class in semantic_calc[pb_id]:
                if pb_id not in results:
                    results[pb_id] = dict()
                if vn_class not in results[pb_id]:
                    results[pb_id][vn_class] = []
                results[pb_id][vn_class].append(semantic_calc[pb_id][vn_class])
    return results


def process_and_operator(amr_calc):
    """
    Handle the situation where the "and" operator should be replaced by copies of statements

    Example:
    You see a dishwasher and a fridge .

    (s / see-01
        :ARG0 (y / you)
        :ARG1 (a / and
            :op1 (d / dishwasher)
            :op2 (f / fridge)))

    AMR Parse in the predicate calculus form:
    see-01(s) AND see-01.arg0(s, y) AND see-01.arg1(s, a) AND and(a) AND and.op1(a, d)
    AND  and.op1(a, f) AND dishwasher(d) AND fridge(f)

    AND-operator processing:
    see-01.arg1(s, a) AND and(a) AND and.op1(a, d) AND  and.op1(a, f)
    => see-01.arg1(s, d) AND see-01.arg1(s, f) AND see-01(s) AND see-01.arg0(s, y)
    AND see-01.arg1(s, d) AND see-01.arg1(s, f) AND dishwasher(d) AND fridge(f)

    :param amr_calc:
    :return:
    """
    # One pass to build dict
    op2args = dict()
    not_and_calc = []
    for calc in amr_calc:
        if calc.predicate.startswith("and."):
            if calc.arguments[0] not in op2args:
                op2args[calc.arguments[0]] = []
            op2args[calc.arguments[0]].append(calc.arguments[1])
        elif calc.predicate != "and":     # exclude and(a)
            not_and_calc.append(calc)

    # Another pass to replace calculus
    to_add = []
    for idx in range(len(not_and_calc)):
        cur_calc = not_and_calc[idx]
        if len(cur_calc.arguments) > 1 and cur_calc.arguments[1] in op2args:
            for arg_idx, arg in enumerate(op2args[cur_calc.arguments[1]]):
                if arg_idx == 0:
                    not_and_calc[idx].arguments[1] = arg
                else:
                    calc_copy = copy.deepcopy(cur_calc)
                    calc_copy.arguments[1] = arg
                    to_add.append(calc_copy)
    not_and_calc.extend(to_add)
    return not_and_calc


def construct_calculus_from_amr(amr):
    """
    Construct predicate calculus from AMR parse
    :param amr: AMR string
    :return: list of PredicateCalculus objects
    """
    amr_calc = []
    arg_map = dict()
    src2tgt = dict()
    g = penman.decode(amr)
    for inst in g.instances():
        src2tgt[inst.source] = inst.target
        amr_calc.append(PredicateCalculus(inst.target, [inst.source]))
    for edge in g.edges():
        if edge.role.lower().startswith(":arg") \
                or edge.role.lower().startswith(":op"):
            predicate = src2tgt[edge.source] + "." + edge.role[1:].lower()
        else:
            predicate = edge.role[1:].lower()
        amr_calc.append(PredicateCalculus(predicate, [edge.source, edge.target]))
        tgt = src2tgt[edge.source]
        if tgt != "and" and "-" in tgt and tgt.index("-") == len(tgt) - 3:
            tgt = tgt[:-3] + "." + tgt[-2:]
        if tgt not in arg_map:
            arg_map[tgt] = dict()
        if edge.source not in arg_map[tgt]:
            arg_map[tgt][edge.source] = dict()
        arg_map[tgt][edge.source][edge.role] = edge.target
    # print(amr_calc)
    # print(arg_map)
    # input()
    return amr_calc, arg_map


def get_event_from_argument(argument):
    """
    Infer event from arguments (basically for VerbNet 3.2 semantics)
    :param argument:
    :return:
    """
    argument = argument.strip()
    prefixes = ["start(", "end(", "during(", "result("]
    for pre in prefixes:
        if argument.startswith(pre) and argument.endswith(")"):
            event = argument[len(pre):-1]
            event_time_point = pre[:-1]
            return event, event_time_point
    return None, None


def induce_unique_groundings(grounded_stmt, semantic_calc, verbose=False):
    """
    Generate unique groundings from multiple mappings in grounded statements
    and the corresponding semantic calculus.
    :param grounded_stmt:
    :param semantic_calc:
    :param verbose:
    :return:
    """
    if verbose:
        print("\ngrounded_stmt:")
        print(grounded_stmt)
        print("\nsemantic_calculus:")
        print(semantic_calc)

    stmt_pools = [dict()]
    calc_pools = [dict()]
    for pb_id in grounded_stmt:
        new_stmt_pools = []
        new_calc_pools = []
        for vn_class in grounded_stmt[pb_id]:
            for unique_stmts in stmt_pools:
                unique_stmts[pb_id] = copy.deepcopy(grounded_stmt[pb_id][vn_class])
                new_stmt_pools.append(copy.deepcopy(unique_stmts))

            for unique_calcs in calc_pools:
                unique_calcs[pb_id] = copy.deepcopy(semantic_calc[pb_id][vn_class])
                new_calc_pools.append(copy.deepcopy(unique_calcs))
        stmt_pools = new_stmt_pools
        calc_pools = new_calc_pools

    if verbose:
        print("\nstmt_pools:")
        print(str(stmt_pools))
        print("\ncalc_pools:")
        print(str(calc_pools))
    return stmt_pools, calc_pools


def build_graph_from_amr(amr, verbose=False):
    """
    Build undirected and directed networkx graphs from the
    annotation of AMR parse
    :param amr: the AMR parse
    :param verbose:
    :return:
    """
    amr_graph = penman.decode(amr)

    g_directed = nx.DiGraph()
    node_dict = dict()

    # read AMR annotation
    amr_obj = read_amr_annotation(amr)
    if verbose:
        print("\nnodes:", amr_obj["nodes"])
        print("\nedges:", amr_obj["edges"])
        print("\ntoken2node_id:", amr_obj["token2node_id"])
        print("\nnode_idx2node_id:", amr_obj["node_idx2node_id"])
        print("\nnode_id2node_idx:", amr_obj["node_id2node_idx"])

    # construct graph from AMRs
    for node in amr_obj["nodes"]:
        if verbose:
            print("node:", node)

        node_id = amr_obj["node_idx2node_id"][node["node_idx"]]
        g_directed.add_node(node_id, label=node["node_label"], source="amr")
        node_dict[node_id] = node["node_label"]

        for attr in amr_graph.attributes(node_id):
            if verbose:
                print("attr:", attr)

            if attr.target.startswith("\"") and attr.target.endswith("\""):
                attr_constant = attr.target[1:-1]
            else:
                attr_constant = attr.target

            # use the parent node of attributes for pattern mining
            amr_obj["token2node_id"][attr_constant.lower()] = attr.source

    for edge in amr_obj["edges"]:
        if verbose:
            print("edge:", edge)

        src_node_id = amr_obj["node_idx2node_id"][edge["src_node_idx"]]
        tgt_node_id = amr_obj["node_idx2node_id"][edge["tgt_node_idx"]]
        g_directed.add_edge(src_node_id, tgt_node_id, label=":" + edge["edge_label"], source="amr")

    if verbose:
        print("\nnode_dict:", node_dict)
        print("\ntoken2node_id post:", amr_obj["token2node_id"])
    return g_directed, amr_obj


def build_graph_from_amr_penman(amr, verbose=False):
    """
    Build undirected and directed networkx graphs from AMR parse
    using penman parsed graph
    :param amr: the AMR parse
    :param verbose:
    :return:
    """
    amr_graph = penman.decode(amr)
    g_directed = nx.DiGraph()
    g_undirected = nx.Graph()

    node_dict = dict()

    # read AMR annotation
    amr_obj = read_amr_annotation(amr)
    if verbose:
        print("\nnodes:", amr_obj["nodes"])
        print("\nedges:", amr_obj["edges"])
        print("\ntoken2node_id:", amr_obj["token2node_id"])
        print("\nnode_idx2node_id:", amr_obj["node_idx2node_id"])
        print("\nnode_id2node_idx:", amr_obj["node_id2node_idx"])

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
            amr_obj["token2node_id"][attr_constant.lower()] = attr.source

    for edge in amr_graph.edges():
        if verbose:
            print("edge:", edge)

        g_directed.add_edge(edge.source, edge.target, label=edge.role, source="amr")
        g_undirected.add_edge(edge.source, edge.target, label=edge.role, source="amr")

    if verbose:
        print("\nnode_dict:", node_dict)
        print("\ntoken2node_id post:", amr_obj["token2node_id"])
    return g_directed, g_undirected, amr_obj


def build_semantic_graph(amr, grounded_stmt, semantic_calculus, verbose=False):
    """
    To represent the graph of semantics, we use event as pivot, then use event
    time point (e.g. start, end, or, during) as edge to connect a predicate
    instance of a calculus statement, and attach grounded argument nodes to the
    predicate using edges with types defined from argument roles in the semantics.
    All predicate instances are connected to the predicate type using edges of ":type".

    :param amr: amr parse in string. Set to None if not used.
    :param grounded_stmt: grounded statements returned by the AMR-VerbNet service. Set to None if not used.
    :param semantic_calculus: semantic calculus returned by the AMR-VerbNet service
    :param verbose: if printing intermediate results
    :return: a networkx graph instance representing the enhanced AMR graph

    Example:
    graph = build_semantic_graph(
        amr=res["amr_parse"][i]["amr"],
        grounded_stmt=res["amr_parse"][i]["grounded_stmt"],
        semantic_calculus=res["amr_parse"][i]["sem_cal"])
    where res is the result returned by the AMR-VerbNet service.
    """
    amr_obj = None
    if amr is not None:
        g, amr_obj = build_graph_from_amr(amr, verbose=verbose)
    else:
        g = nx.DiGraph()

    node_dict = dict()
    event_inst_counter = Counter()
    pred_inst_counter = Counter()
    free_arg_inst_counter = Counter()

    # graph from AMRs
    if amr is not None:
        amr_graph = penman.decode(amr)
        for inst in amr_graph.instances():
            if verbose:
                print("inst.source:", inst.source)
                print("inst.target:", inst.target)
            g.add_node(inst.source, label=inst.target, source="amr")
            node_dict[inst.source] = inst.target

        for edge in amr_graph.edges():
            if verbose:
                print("edge:", edge)
            g.add_edge(edge.source, edge.target, label=edge.role, source="amr")

    if grounded_stmt is None:
        return g, amr_obj

    print("\ngrounded_stmt:", grounded_stmt)
    # graph from grounded statements
    for pb_id in grounded_stmt:
        for g_idx, group in enumerate(grounded_stmt[pb_id]):
            # statements a group share the same group of events
            event2id = dict()

            # increment for id generation
            for evt in event_inst_counter:
                event_inst_counter[evt] += 1

            for arg in free_arg_inst_counter:
                free_arg_inst_counter[arg] += 1

            for s_idx, stmt in enumerate(group):
                # create node for predicate
                if stmt["is_negative"]:
                    predicate = "NOT_" + stmt["predicate"]
                else:
                    predicate = stmt["predicate"]

                # create a node for the predicate type and add edges to its instances
                if predicate not in node_dict:
                    g.add_node(predicate, label=predicate, source="verbnet")
                    node_dict[predicate] = predicate

                predicate_id = predicate + "-{}".format(pred_inst_counter[predicate])
                if predicate_id not in node_dict:
                    g.add_node(predicate_id, label=predicate + "-inst", source="verbnet")
                    node_dict[predicate_id] = predicate
                    pred_inst_counter[predicate] += 1
                    g.add_edge(predicate_id, predicate, label=":type", source="verbnet")

                print("\n\n\n\n\nstmt arguments:", stmt["arguments"])
                print("stmt:", stmt)
                for arg_idx, arg in enumerate(stmt["arguments"]):
                    # check whether the argument is event-related
                    event, event_time_point = get_event_from_argument(arg)

                    if event is not None:
                        event_id = event + "-{}".format(event_inst_counter[event])
                        if event_id not in node_dict:
                            g.add_node(event_id, label=event, source="verbnet")
                            node_dict[event_id] = event
                            if event not in event2id:
                                event2id[event] = event_id

                        # add edge between event and predicate
                        g.add_edge(event_id, predicate_id, label=":" + event_time_point, source="verbnet")
                        continue

                    print("\narg:", arg)
                    print("arg_idx:", arg_idx)

                    if len(arg) <= 2 and arg.startswith("E"):
                        # the arg represents an event
                        event = arg
                        event_id = event + "-{}".format(event_inst_counter[event])
                        if event not in event2id and event_id not in node_dict:
                            g.add_node(event_id, label=event, source="verbnet")
                            node_dict[event_id] = event
                            event2id[event] = event_id
                        g.add_edge(predicate_id, event_id, label=":event", source="verbnet")
                    else:
                        # handle cases where grounded statements are expanded due to the AND phrase
                        sem_cal_stmt_idx = s_idx % len(semantic_calculus[pb_id])

                        print("len(semantic_calculus[pb_id]):", len(semantic_calculus[pb_id]))
                        print("s_idx:", s_idx)
                        print("ungrounded arguments:", semantic_calculus[pb_id][sem_cal_stmt_idx])
                        label = semantic_calculus[pb_id][sem_cal_stmt_idx]["arguments"][arg_idx]
                        if label == arg:
                            # indicate it is unknown
                            if not arg.startswith("?"):
                                arg_name = "?" + arg
                            else:
                                arg_name = arg
                            arg_id = arg + "-{}".format(free_arg_inst_counter[arg])
                            if arg_id not in node_dict:
                                g.add_node(arg_id, label=arg_name, source="verbnet")
                                node_dict[arg_id] = arg_name
                        else:
                            arg_id = arg

                        # remove "?" for edge labels
                        if label.startswith("?"):
                            label = label[1:]
                        g.add_edge(predicate_id, arg_id, label=":" + label, source="verbnet")

    # print("\nEdges of enhanced AMR graph:\n", g.edges())
    return g, amr_obj


def visualize_semantic_graph(graph, out_dir, graph_name="semantic_graph", figure_format="png"):
    """
    Generate a figure that visualize the enhanced AMR graph with VerbNet semantics
    :param graph: a networkx graph instance
    :param out_dir: the directory to save the figure
    :param graph_name: the name of the graph for specifying the file name
    :param figure_format: format/extension of the output figure file
    :return:
    """
    engine = "dot"  # ["neato", "circo"]
    dot = graphviz.Digraph(name=graph_name, format=figure_format, engine=engine)

    color_map = {
        "amr": "blue",
        "verbnet": "red"
    }

    for node in graph.nodes.data():
        # print("node:", node)
        node_id, node_attrs = node
        dot.node(node_id, label=node_attrs["label"] if "label" in node_attrs else node_id,
                 color=color_map[node_attrs["source"]])

    for edge in graph.edges().data():
        # print("edge:", edge)
        edge_src, edge_tgt, edge_attrs = edge
        dot.edge(edge_src, edge_tgt, label=edge_attrs["label"],
                 color=color_map[edge_attrs["source"]])

    out_path = os.path.join(out_dir, graph_name)
    dot.render(out_path)
    print("\nWritten graph to file {}".format(out_path))


if __name__ == "__main__":
    # parse_text("You enter a kitchen.")
    # parse_text("The quick brown fox jumped over the lazy moon.")
    # parse_text("You see a dishwasher and a fridge.")
    # parse_text("Here 's a dining table .")
    # parse_text("You see a red apple and a dirty plate on the table .")

    # ground_text_to_verbnet("You enter a kitchen.")
    # ground_text_to_verbnet("You see a dishwasher and a fridge.")
    # ground_text_to_verbnet("Here 's a dining table .")
    # ground_text_to_verbnet("You see a red apple and a dirty plate on the table .")
    # ground_text_to_verbnet("The dresser is made out of maple carefully finished with Danish oil.", verbose=True)
    ground_text_to_verbnet("In accordance with our acceptance of funds from the U.S. Treasury, cash dividends on common stock are not permitted without prior approval from the U.S.", verbose=True)

