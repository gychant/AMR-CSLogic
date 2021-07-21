"""
Test for services
"""
import requests
import json
from pprint import pprint

# host = "10.208.109.152"
host = "0.0.0.0"
port = 5000

# text = "You enter a kitchen."
# text = "You see a dishwasher and a fridge."
# text = "Here 's a dining table ."
# text = "You see a red apple and a dirty plate on the table ."

text = """You've entered a kitchen. You see a dishwasher and a fridge. Here's a dining table. You see a dirty plate and a red apple on the table."""

# res = requests.get("http://{}:{}/amr_parsing".format(host, port), params={'text': text})
res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})

print("\nres.text")
print(res.text)
l = len(json.loads(res.text)["amr_parse"])

for i in range(l):
    print("\namr:")
    print(json.loads(res.text)["amr_parse"][i]["amr"])
    print("\npb_vn_mappings:")
    pprint(json.loads(res.text)["amr_parse"][i]["pb_vn_mappings"])

    print("\namr_cal:")
    print(json.loads(res.text)["amr_parse"][i]["amr_cal"])

    print("\nsem_cal:")
    print(json.loads(res.text)["amr_parse"][i]["sem_cal"])

    print("\ngrounded_stmt:")
    print(json.loads(res.text)["amr_parse"][i]["grounded_stmt"])
    
    print("\namr_cal:")
    print(json.loads(res.text)["amr_parse"][i]["amr_cal_str"])

    print("\nsem_cal:")
    print(json.loads(res.text)["amr_parse"][i]["sem_cal_str"])

    print("\ngrounded_stmt:")
    print(json.loads(res.text)["amr_parse"][i]["grounded_stmt_str"])

