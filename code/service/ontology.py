from code.core.sparql_util import request_wrapper

url_query = "http://goedel.sl.cloud9.ibm.com:9999/blazegraph/namespace/VerbnetAndGroupings/#query"

# To query all PropBank and VerbNet mappings
query_mapping_str = """
prefix rrp:<http://www.ibm.com/RRP#>
prefix glo:<http://www.ibm.com/GLO_V2#>
prefix ulkb: <http://www.ibm.com/UL_KB#>
prefix ulvn: <http://www.ibm.com/UL_VN#>

SELECT DISTINCT ?mappingGroupName ?PropbankMapID ?VerbnetMapID WHERE {
  ?mappingGroup a rrp:MappingGroup .
  ?mappingGroup rrp:hasComponent ?mapping .

  #Verbnet
  ?mapping rrp:mapsTo ?vnsense .
  ulvn:VerbNet rrp:hasComponent ?vnsense.
  ?vnsense rrp:identifier ?VerbnetMapID .

  #PropBank
  ?mapping rrp:mapsTo ?pbsense .
  ulkb:PropBank rrp:hasComponent ?pbsense.
  ?pbsense rrp:identifier ?PropbankMapID .

  ?mappingGroup rdfs:label ?mappingGroupName .
  } ORDER BY ?mappingGroupName
"""


def load_kg_pb_vn_mappings():
    mappings = dict()
    res_data = request_wrapper(url_query, query_mapping_str).json()
    for result in res_data["results"]["bindings"]:
        propbank_id = result["PropbankMapID"]["value"]
        verbnet_id = result["VerbnetMapID"]["value"]
        mappings[propbank_id] = verbnet_id
    return mappings


pb_vn_mappings = load_kg_pb_vn_mappings()


def query_pb_vn_mapping(verb_roleset):
    if verb_roleset not in pb_vn_mappings:
        return None

    res = {
        "mapping": pb_vn_mappings[verb_roleset],
        "source": "ontology"
    }
    return res

