import copy
import json
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from typing import List
from pprint import pprint
import penman

from code.grpc_clients import AMRClientTransformer
from code.core.stanford_nlp_parse import full_parsing
from code.core.format_util import to_json

from code.service.propbank import query_propbank_roles
from code.service.verbnet import query_semantics
from code.service.semlink import query_pb_vn_mapping
# from code.service.ontology import query_pb_vn_mapping
from code.service.models import PredicateCalculus
from code.service.amr import amr_client

verbose = False


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


def ground_text_to_verbnet(text):
    sentences = sent_tokenize(text)
    print("parsing ...")
    parse = full_parsing(text, do_coreference=True)
    print("\ntext:\n", text)
    print("\nsentences:\n==>", "\n\n==>".join(sentences))
    print("\ncoreference:\n", parse["coreference"])

    sentence_parses = []
    for idx, sent in enumerate(sentences):
        sent_res = dict()
        amr = amr_client.get_amr(sent)
        g_res = ground_amr(amr)
        sent_res["amr"] = amr
        sent_res.update(g_res)
        sentence_parses.append(sent_res)

    results = {
        "coreference": parse["coreference"],
        "sentence_parses": sentence_parses
    }
    return results


def ground_amr(amr):
    g = penman.decode(amr)

    """
    print("g.instances():", g.instances())
    print("g.edges():", g.edges())
    print("g.variables():", g.variables())
    print("g.metadata:", g.metadata)
    print("g.epidata:", g.epidata)
    print("g.reentrancies():", g.reentrancies())
    """

    role_mappings = dict()
    semantics = dict()
    pb_vn_mappings = dict()
    for inst in g.instances():
        # if it is a propbank frame
        if len(inst.target) > 3 and inst.target[-3] == "-" and inst.target[-2:].isnumeric():
            pb_id = inst.target[:-3] + "." + inst.target[-2:]
            if pb_id not in role_mappings:
                role_map = query_propbank_roles(pb_id)
                if role_map is None:
                    continue
                role_mappings[pb_id] = role_map

            mapping_res = query_pb_vn_mapping(pb_id)
            if mapping_res is not None and pb_id not in semantics:
                verbnet_id = mapping_res["mapping"]
                verbnet_version = mapping_res["source"]
                pb_vn_mappings[pb_id] = mapping_res
                semantics[pb_id] = query_semantics(verbnet_id, verbnet_version)

    ori_amr_cal, arg_map = construct_calculus_from_amr(amr)

    if verbose:
        print("\nori_amr_cal:", ori_amr_cal)
        print("arg_map:", arg_map)
        print("role_mappings:", role_mappings)

    amr_cal = process_and_operator(ori_amr_cal)
    sem_cal = construct_calculus_from_semantics(semantics)
    grounded_stmt = ground_semantics(arg_map, sem_cal, role_mappings)

    if verbose:
        print("\namr_cal:", amr_cal)
        print("\nsem_cal:", sem_cal)
        print("\ngrounded_stmt:", grounded_stmt)

    results = {
        "pb_vn_mappings": pb_vn_mappings,
        "amr_cal": to_json(amr_cal),
        "sem_cal": to_json(sem_cal),
        "grounded_stmt": to_json(grounded_stmt),
        "amr_cal_str": str(amr_cal),
        "sem_cal_str": str(sem_cal),
        "grounded_stmt_str": str(grounded_stmt)
    }
    return results


def construct_calculus_from_semantics(semantics):
    """
    :param amr: AMR string
    :return: list of PredicateCalculus objects
    """
    results = dict()
    for propbank_id in semantics:
        for event in semantics[propbank_id]:
            if propbank_id not in results:
                results[propbank_id] = []
            results[propbank_id].append(PredicateCalculus(
                event["predicate_value"].upper(),
                [arg["value"] for arg in event["arguments"]]))
    # print(results)
    # input()
    return results


def ground_semantics(arg_map, semantic_calc, role_mappings):
    """
    :param arg_map:
    :param semantic_calc:
    :param role_mappings:
    :return:
    """

    if verbose:
        print("\narg_map:")
        print(arg_map)
        print("\nsemantic_calc:")
        print(semantic_calc)
        print("\nrole_mappings:")
        pprint(role_mappings)

    results = []
    for propbank_id in arg_map:
        if propbank_id not in semantic_calc:
            continue
        if propbank_id not in role_mappings:
            continue

        cur_role_mappings = role_mappings[propbank_id]
        for src in arg_map[propbank_id]:
            cur_calculus = copy.deepcopy(semantic_calc[propbank_id])
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
            results.append(final_calculus)

    for pb_id in semantic_calc:
        if pb_id not in arg_map:
            results.append(semantic_calc[pb_id])
    return results


def process_and_operator(amr_calc):
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


if __name__ == "__main__":
    verbose = True
    # parse_text("You enter a kitchen.")
    # parse_text("The quick brown fox jumped over the lazy moon.")
    # parse_text("You see a dishwasher and a fridge.")
    # parse_text("Here 's a dining table .")
    # parse_text("You see a red apple and a dirty plate on the table .")

    # ground_text_to_verbnet("You enter a kitchen.")
    # ground_text_to_verbnet("You see a dishwasher and a fridge.")
    # ground_text_to_verbnet("Here 's a dining table .")
    ground_text_to_verbnet("You see a red apple and a dirty plate on the table .")

