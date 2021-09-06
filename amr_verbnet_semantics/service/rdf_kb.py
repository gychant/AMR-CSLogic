"""
Accessing the unified KB using SPARQL
"""
from SPARQLWrapper import SPARQLWrapper, JSON

query_prefix = """
prefix rrp: <http://www.ibm.com/RRP#>
prefix glo: <http://www.ibm.com/GLO_V2#>
prefix ulvn: <http://www.ibm.com/UL_VN#>
prefix ulwn: <http://www.ibm.com/UL_WN#>
prefix ulpb: <http://www.ibm.com/UL_PB#>
prefix ulkb: <http://www.ibm.com/UL_KB#>
"""

sparql = SPARQLWrapper("http://goedel.sl.cloud9.ibm.com:9999/blazegraph/namespace/UL_KB_V0")


def query_verbnet_semantic_roles(propbank_id):
    query_text = """SELECT DISTINCT ?verb ?pbSemRole ?vnVerbLabel ?vnSemRole WHERE {

      ?verb rdfs:label "%s" . 
      #?verb rrp:inKB rrp:PropBank .
      ?verb rrp:hasParameter ?pbParam . 
      ?pbParam rdfs:label ?pbSemRole . 
      ?verb rrp:hasMapping ?mapping .  
      ?vnVerb rrp:hasMapping ?mapping . 
      ?vnVerb rrp:inKB rrp:VerbNet . 
      ?vnVerb rdfs:label ?vnVerbLabel . 
      OPTIONAL {
      ?vnVerb rrp:hasParameter ?vnParam . 
      ?pbParam rrp:mapsTo ?vnParam . 
      ?vnParam rdfs:label ?vnSemRole 
       } 
      } """ % propbank_id

    sparql.setQuery(query_prefix + query_text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        pb_sem_role = result["pbSemRole"]["value"]
        vn_verb_label = result["vnVerbLabel"]["value"]
        if "vnSemRole" in result:
            vn_sem_role = result["vnSemRole"]["value"]
        else:
            vn_sem_role = "N/A"
        print(pb_sem_role + "\t(" + vn_verb_label + ", " + vn_sem_role + ")")


def query_verbnet_semantic_predicates(verbnet_class_id):
    verbnet_class_id = verbnet_class_id.replace(".", "_")

    query_text = """SELECT DISTINCT  ?example ?semanticPredicate ?predicateText  WHERE {
                      ulvn:%s rrp:hasComponent ?frame . 
                      ?frame rrp:example ?example . 
                      ?frame rrp:hasComponent ?semanticPredicate . 
                      ?semanticPredicate a rrp:SemanticPredicate . 
                      ?semanticPredicate rrp:textInfo ?predicateText .  
                    } ORDER BY ?frame
                      """ % verbnet_class_id

    sparql.setQuery(query_prefix + query_text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        predicate_text = result["predicateText"]["value"]
        example = result["example"]["value"]
        semantic_predicate = result["semanticPredicate"]["value"]
        # print("semantic_predicate:", semantic_predicate)
        print(example + "\t" + semantic_predicate.split('#')[1] + "\t" + predicate_text)


if __name__ == "__main__":
    query_verbnet_semantic_roles("admire.01")
    # query_verbnet_semantic_roles("enter.01")
    query_verbnet_semantic_predicates("admire-31_2")

