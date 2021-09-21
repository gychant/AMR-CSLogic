"""PropBank query wrapper"""
import re

from nltk.corpus import treebank
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import VerbnetCorpusReader, PropbankCorpusReader


propbank = LazyCorpusLoader(
    "propbank-latest",
    PropbankCorpusReader,
    "prop.txt",
    r"frames/.*\.xml",
    "verbs.txt",
    lambda filename: re.sub(r"^wsj/\d\d/", "", filename),
    treebank,
)  # Must be defined *after* treebank corpus.


def query_propbank_roles(propbank_id):
    # print("query_propbank_roles for propbank_id {}".format(propbank_id))
    try:
        role_set = propbank.roleset(propbank_id)
    except ValueError:
        return None

    role_mappings = dict()
    for role in role_set.findall("roles/role"):
        for vn_role in role.findall('vnrole'):
            arg_key = ":ARG{}".format(role.attrib['n'])
            if arg_key not in role_mappings:
                role_mappings[arg_key] = []
            role_mappings[arg_key].append({
                "vncls": vn_role.attrib["vncls"],
                "vntheta": vn_role.attrib["vntheta"],
                "description": role.attrib['descr']
            })
    # print("query_propbank_roles role_mappings:", role_mappings)
    return role_mappings


def query_propbank_roleset_ids():
    roleset_ids = set()
    for roleset in propbank.rolesets():
        roleset_ids.add(roleset.attrib["id"])
    return roleset_ids


if __name__ == "__main__":
    print(query_propbank_roleset_ids())

