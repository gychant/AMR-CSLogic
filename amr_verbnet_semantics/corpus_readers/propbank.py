"""PropBank query wrapper"""
import re
from xml.etree import ElementTree

from nltk.corpus import treebank
from nltk.corpus.reader import PropbankCorpusReader, VerbnetCorpusReader
from nltk.corpus.util import LazyCorpusLoader

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
    print("\nquery_propbank_roles for propbank_id {}".format(propbank_id))
    try:
        role_set = propbank.roleset(propbank_id)
    except Exception as e:
        print(e)
        return None

    role_mappings = dict()
    print("role_set:", ElementTree.tostring(role_set, encoding='unicode'))
    for role in role_set.findall("roles/role"):
        print("role:", ElementTree.tostring(role, encoding='unicode'))
        for vn_role in role.findall('vnrole'):
            print("vn_role:", ElementTree.tostring(vn_role, encoding='unicode'))
            arg_key = ":ARG{}".format(role.attrib['n'])
            if arg_key not in role_mappings:
                role_mappings[arg_key] = []
            role_mappings[arg_key].append({
                "vncls": vn_role.attrib["vncls"],
                "vntheta": vn_role.attrib["vntheta"],
                "description": role.attrib['descr']
            })
    print("query_propbank_roles role_mappings:", role_mappings)
    return role_mappings


def query_propbank_roleset_ids():
    roleset_ids = set()
    for roleset in propbank.rolesets():
        roleset_ids.add(roleset.attrib["id"])
    return roleset_ids


if __name__ == "__main__":
    # print(query_propbank_roleset_ids())
    print(query_propbank_roles("make_out.23"))

