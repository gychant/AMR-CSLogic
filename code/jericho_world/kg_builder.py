"""Build knowledge graphs of the game using AMR-VerbNet semantics"""
import os
import json
import requests
from pprint import pprint
from collections import Counter
import random
random.seed(42)

host = "0.0.0.0"
port = 5000

DATA_DIR = "./data/JerichoWorld"
sample_size = 10

print("Loading data ...")
with open(os.path.join(DATA_DIR, "train.json")) as f:
    data = json.load(f)
print("Loaded data ...")


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


def check_samples():
    # Select samples for initial testing
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:sample_size]

    for idx in sample_indices:
        sample = data[idx]
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

        text = sample["state"]["obs"]
        res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})

        print("\nres.text:")
        print(res.text)

        res = json.loads(res.text)
        if "amr_parse" in res:
            for i in range(len(res["amr_parse"])):
                print("\namr:")
                print(res["amr_parse"][i]["amr"])
                print("\npb_vn_mappings:")
                pprint(res["amr_parse"][i]["pb_vn_mappings"])

                print("\namr_cal:")
                print(res["amr_parse"][i]["amr_cal"])

                print("\nsem_cal:")
                print(res["amr_parse"][i]["sem_cal"])

                print("\ngrounded_stmt:")
                print(res["amr_parse"][i]["grounded_stmt"])

                print("\namr_cal:")
                print(res["amr_parse"][i]["amr_cal_str"])

                print("\nsem_cal:")
                print(res["amr_parse"][i]["sem_cal_str"])

                print("\ngrounded_stmt:")
                print(res["amr_parse"][i]["grounded_stmt_str"])
            input()


if __name__ == "__main__":
    check_samples()
    # kb_statistics()

