"""
Utility functions for AMR processing
"""
import json


def double_quote(json_str):
    return json_str.replace(" ", "").replace("{", "{\"")\
        .replace(":", "\":").replace(",", ",\"").replace("'", "\"")


def read_amr_annotation(amr):
    token2node_idx = dict()
    node_idx2token = dict()
    token2node_id = dict()
    node_id2token = dict()
    nodes = []
    edges = []

    for line in amr.split("\n"):
        if line.startswith("# ::tok"):
            # tokenized text
            text = line[len("# ::tok"):-len("<ROOT>")].strip()
            tokens = text.split()

        elif line.startswith("# ::node"):
            node_info = line[len("# ::node"):].strip()
            columns = node_info.split()
            if len(columns) < 2:
                continue

            if len(columns) == 2:
                node_idx, node_label = columns
                token = node_label
            else:
                node_idx, node_label, token_range = columns
                start_tok_idx, end_tok_idx = token_range.split("-")
                token = " ".join(tokens[int(start_tok_idx):int(end_tok_idx)]).lower()

            token2node_idx[token] = node_idx
            node_idx2token[node_idx] = token
            nodes.append({
                "node_idx": node_idx,
                "node_label": node_label,
                "token": token
            })

        elif line.startswith("# ::edge"):
            edge_info = line[len("# ::edge"):].strip()
            columns = edge_info.split()
            if len(columns) < 5:
                continue

            src_node_label, edge_label, tgt_node_label, \
                src_node_idx, tgt_node_idx = columns

            if edge_label.endswith("-of") and edge_label.lower() != "consist-of":
                edges.append({
                    "src_node_label": tgt_node_label,
                    "edge_label": edge_label[:-len("-of")],
                    "tgt_node_label": src_node_label,
                    "src_node_idx": tgt_node_idx,
                    "tgt_node_idx": src_node_idx
                })
            else:
                edges.append({
                    "src_node_label": src_node_label,
                    "edge_label": edge_label,
                    "tgt_node_label": tgt_node_label,
                    "src_node_idx": src_node_idx,
                    "tgt_node_idx": tgt_node_idx
                })

        elif line.startswith("# ::short"):
            node_idx2node_id = json.loads(double_quote(line[len("# ::short"):].strip()))
            node_id2node_idx = {v: k for k, v in node_idx2node_id.items()}

    for token in token2node_idx:
        token2node_id[token] = node_idx2node_id[token2node_idx[token]]

    try:
        for node_id in node_id2node_idx:
            node_id2token[node_id] = node_idx2token[node_id2node_idx[node_id]]
    except Exception as e:
        print(amr)
        print(e)
        input()

    amr_data = {
        "nodes": nodes,
        "edges": edges,
        "token2node_idx": token2node_idx,
        "node_idx2token": node_idx2token,
        "token2node_id": token2node_id,
        "node_id2token": node_id2token,
        "node_idx2node_id": node_idx2node_id,
        "node_id2node_idx": node_id2node_idx
    }

    return amr_data

