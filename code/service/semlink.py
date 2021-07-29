#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 8 2021
@author: Zhicheng (Jason) Liang
"""

import json
import os

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader, PropbankCorpusReader

from fuzzywuzzy import process

pb_vn_mappings = json.load(open("./data/semlink/instances/pb-vn2.json"))
vn_fn_mappings = json.load(open("./data/semlink/instances/vn-fn2.json"))

vn_dict = {
    "verbnet3.2": LazyCorpusLoader("verbnet3.2", VerbnetCorpusReader, r"(?!\.).*\.xml"),
    "verbnet3.3": LazyCorpusLoader("verbnet3.3", VerbnetCorpusReader, r"(?!\.).*\.xml"),
    "verbnet3.4": LazyCorpusLoader("verbnet3.4", VerbnetCorpusReader, r"(?!\.).*\.xml")
}


def sanity_check(cls, all_classes, vn_version):
    for sub_cls in vn_dict[vn_version].subclasses(cls):
        if sub_cls not in all_classes:
            print("invalid:", sub_cls)
            input()
            continue
        sanity_check(sub_cls, all_classes, vn_version)


def build_class_id_dict(vn_version):
    results = dict()
    all_classes = set(vn_dict[vn_version].classids())
    for cls in all_classes:
        sanity_check(cls, all_classes, vn_version)

    for cls in all_classes:
        cls_num = "-".join(cls.split("-")[1:])
        results[cls_num] = cls
    return results


vn2class_id_dict = {
    "verbnet3.2": build_class_id_dict("verbnet3.2"),
    "verbnet3.3": build_class_id_dict("verbnet3.3"),
    "verbnet3.4": build_class_id_dict("verbnet3.4")
}


def check_mapping_completeness(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    invalid_out_path = os.path.join(output_dir, "invalid_mappings.tsv")
    invalid_out_file = open(invalid_out_path, "w")
    invalid_out_file.write("verb_roleset\tvn_class\tfuzzy_matches\tvn_version\n")

    valid_out_path = os.path.join(output_dir, "valid_mappings.tsv")
    valid_out_file = open(valid_out_path, "w")
    valid_out_file.write("verb_roleset\tvn_class\tmatches\tvn_version\n")
    
    valid_cnt = 0
    for verb_roleset in pb_vn_mappings:
        vn_mappings = pb_vn_mappings[verb_roleset]
        # check the version that has a valid mapping
        invalid_out_file_buffer = []
        is_mapped = False
        for vn_version in vn2class_id_dict:
            class_id_dict = vn2class_id_dict[vn_version]
            all_class_ids = class_id_dict.keys()
            if vn_mappings["vn_class"] not in class_id_dict:
                # print("\nInvalid mappings in vn {}:".format(vn_version))
                # print("verb_roleset:", verb_roleset)
                # print("vn_class:", vn_mappings["vn_class"])
                fuzzy_matches = process.extract(vn_mappings["vn_class"], all_class_ids)[:5]
                fuzzy_matches = [class_id_dict[m] for m, score in fuzzy_matches]
                # print(fuzzy_matches)
                invalid_out_file_buffer.append("{}\t{}\t{}\t{}\n".format(
                    verb_roleset, vn_mappings["vn_class"],
                    str(fuzzy_matches), vn_version))
            else:
                valid_cnt += 1
                is_mapped = True
                # print("[{}] mapped to [{}] in {}".format(
                #     verb_roleset, class_id_dict[vn_mappings["vn_class"]], vn_version))
                valid_out_file.write("{}\t{}\t{}\t{}\n".format(
                    verb_roleset, vn_mappings["vn_class"],
                    class_id_dict[vn_mappings["vn_class"]], vn_version))
                break

        if not is_mapped:
            for row in invalid_out_file_buffer:
                invalid_out_file.write(row)

    invalid_out_file.close()
    valid_out_file.close()
    print("\nInvalid cnt:", len(pb_vn_mappings) - valid_cnt)
    print("Total:", len(pb_vn_mappings))
    print("\nWritten to file {}".format(invalid_out_path))
    print("\nWritten to file {}".format(valid_out_path))


def query_pb_vn_mapping(verb_roleset):
    if verb_roleset not in pb_vn_mappings:
        return None

    # check the version that has a valid mapping
    vn_mappings = pb_vn_mappings[verb_roleset]
    for vn_version in vn2class_id_dict:
        class_id_dict = vn2class_id_dict[vn_version]
        if "vn_class" in vn_mappings and vn_mappings["vn_class"] in class_id_dict:
            res = {
                "mapping": class_id_dict[vn_mappings["vn_class"]],
                "source": vn_version
            }
            return res
    return None


if __name__ == '__main__':
    check_mapping_completeness(output_dir="./data/semlink_out")

