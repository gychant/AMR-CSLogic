
import requests
import json

from amr_verbnet_semantics.core.amr_verbnet_enhance import (
    build_semantic_graph,
    visualize_semantic_graph
)

host = "0.0.0.0"
port = 5000

# text = "For the fifth grade play , the chairs have been put into 27 rows with 16 chairs in each row . How many chairs have been put out for the play ?"
# text = "You are carrying : a bronze-hilted dagger, a clay ocarina, armor, and silks ( worn ) ."
text = "The dresser is made out of maple carefully finished with Danish oil."
res = requests.get("http://{}:{}/verbnet_semantics".format(host, port), params={'text': text})
res = json.loads(res.text)
print(res)
target_propbank_ids = ['make.01']
for i in range(len(res["amr_parse"])):
    graph, amr_obj = build_semantic_graph(
        amr=res["amr_parse"][i]["amr"],
        grounded_stmt=res["amr_parse"][i]["grounded_stmt"],
        semantic_calculus=res["amr_parse"][i]["sem_cal"],
        target_propbank_ids=target_propbank_ids)
    visualize_semantic_graph(graph, graph_name="amr_semantic_graph_{}".format(i),
                             out_dir="./test-output/")
print("amr_semantic_graph DONE ...")

