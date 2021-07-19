"""
Test for services
"""
import requests
import json
from pprint import pprint

host = "10.208.109.152"
port = 5000

# text = "You enter a kitchen."
# text = "You see a dishwasher and a fridge."
# text = "Here 's a dining table ."
text = "You see a red apple and a dirty plate on the table ."

# res = requests.get("http://{}:{}/amr_parsing".format(host, port), params={'text': text})
res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})

print("\nres.text")
print(res.text)

print("\npb_vn_mappings:")
pprint(json.loads(res.text)["amr_parse"][0]["pb_vn_mappings"])

print("\namr_cal:")
print(json.loads(res.text)["amr_parse"][0]["amr_cal"])

print("\nsem_cal:")
print(json.loads(res.text)["amr_parse"][0]["sem_cal"])

print("\ngrounded_stmt:")
print(json.loads(res.text)["amr_parse"][0]["grounded_stmt"])

