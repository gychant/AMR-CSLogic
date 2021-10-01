"""
Load configuration from .yaml file.
"""
import os
import logging
import yaml

from amr_verbnet_semantics.utils.format_util import DictObject, to_json

if not os.path.exists("config.yaml"):
    raise Exception("Please create a config.yaml file following the config_template.yaml in the project root dir ...")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config = DictObject(config)


if __name__ == "__main__":
    print(to_json(config))
    print(config.SPARQL_ENDPOINT)

