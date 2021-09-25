"""
Accessing the unified KB using SPARQL
"""
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON
import config

query_prefix = """
prefix rrp: <http://www.ibm.com/RRP#>
prefix glo: <http://www.ibm.com/GLO_V2#>
prefix ulvn: <http://www.ibm.com/UL_VN#>
prefix ulwn: <http://www.ibm.com/UL_WN#>
prefix ulpb: <http://www.ibm.com/UL_PB#>
prefix ulkb: <http://www.ibm.com/UL_KB#>
"""

sparql = SPARQLWrapper(config.SPARQL_ENDPOINT)


def query_pb_vn_mapping_from_rdf(propbank_id, vn_version="verbnet3.4"):
    query_text = """SELECT DISTINCT ?verb ?pbSemRole ?vnVerbLabel ?vnParamLabel WHERE {   
        ?verb rdfs:label "%s" . 
        #?verb rrp:inKB rrp:PropBank .
        ?verb rrp:hasParameter ?pbParam . 
        ?pbParam rdfs:label ?pbSemRole . 
        OPTIONAL {
            ?vnVerb rrp:inKB rrp:VerbNet . 
            ?vnVerb rdfs:label ?vnVerbLabel . 
            ?vnVerb rrp:hasComponent ?vnFrame . 
            ?vnFrame rrp:hasComponent ?semPred . 
            ?semPred rrp:hasParameter ?vnParam . 
            ?pbParam rrp:mapsTo ?vnParam . 
            ?vnParam rdfs:label ?vnParamLabel . 
        } 
    } """ % propbank_id

    sparql.setQuery(query_prefix + query_text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    vn_class_set = set()
    for result in results["results"]["bindings"]:
        vn_verb_label = result["vnVerbLabel"]["value"]
        vn_class_set.add(vn_verb_label)

    mapping_res = []
    for vn_class in vn_class_set:
        mapping_res.append({
            "mapping": vn_class,
            "source": vn_version
        })
    if len(mapping_res) == 0:
        return None
    return mapping_res


def query_semantics_from_rdf(verbnet_class_id, verbnet_version="verbnet3.4"):
    verbnet_class_id = verbnet_class_id.replace(".", "_").replace("-", "_")
    print("verbnet_class_id:", verbnet_class_id)

    query_text = """SELECT DISTINCT ?example ?operator ?semanticPredicate ?semanticPredicateLabel ?param ?type ?value  WHERE {{
                  ulvn:%s rrp:hasComponent ?frame . 
                  ?frame rrp:example ?example . 
                  ?frame rrp:hasComponent ?semanticPredicate . 
                  ?semanticPredicate a rrp:SemanticPredicate .
                  ?semanticPredicate rdfs:label ?semanticPredicateLabel. 
                  ?semanticPredicate rrp:hasParameter ?param . 
                  OPTIONAL {{
                    ?semanticPredicate rrp:logicOperatorName ?operator .  
                   }}
                  ?param rrp:varType ?type . 
                  ?param rrp:varName ?value . 
                  ?semanticPredicate rrp:textInfo ?predicateText .  
                }} ORDER BY ?semanticPredicate
                """ % verbnet_class_id

    sparql.setQuery(query_prefix + query_text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # print("bindings:", results["results"]["bindings"])
    # input()

    output = []
    thisFrame = {}
    output.append(thisFrame)
    curPredicateID = ""
    thisPredicate = {}

    # PROCESS OUTPUT IN A JSON-FRIENDLY FORMAT
    for result in results["results"]["bindings"]:
        example = result["example"]["value"]

        if 'example' not in thisFrame:
            thisFrame['example'] = example
        if example != thisFrame['example']:
            thisFrame = {}
            output.append(thisFrame)
            thisFrame['example'] = example
            curPredicateID = ""
        thisPredicateID = result["semanticPredicate"]["value"]

        if 'predicates' not in thisFrame:
            thisFrame['predicates'] = []
        predicates = thisFrame['predicates']

        if thisPredicateID != curPredicateID:
            thisPredicate = {}
            predicates.append(thisPredicate)
            curPredicateID = thisPredicateID

        # predicateText = result["predicateText"]["value"]
        predLabel = result["semanticPredicateLabel"]["value"]

        thisPredicate['label'] = predLabel
        if "operator" in result:
            thisPredicate['operator'] = result["operator"]["value"]

        if 'params' not in thisPredicate:
            thisPredicate['params'] = []
        params = thisPredicate['params']
        params.append({
            'type': result["type"]["value"],
            'value': result["value"]["value"],
            'predicate': curPredicateID
        })
    print(str(output))
    input()

    example2semantics = defaultdict(list)
    for semantic_example in output:
        example = semantic_example["example"]
        statements = []
        for predicate in semantic_example["predicates"]:
            arguments = []
            for param in predicate["params"]:
                arguments.append({
                    "type": param["type"],
                    "value": param["value"]
                })
            statements.append({
                "predicate_value": predicate["label"],
                "arguments": arguments,
                "is_negative": "operator" in predicate
            })
        predicate_text = result["predicateText"]["value"]
        example = result["example"]["value"]
        semantic_predicate = result["semanticPredicate"]["value"]
        print("semantic_predicate:", semantic_predicate)
        print(example + "\t" + semantic_predicate.split('#')[1] + "\t" + predicate_text)
        example2semantics[example].append(predicate_text.split()[1])
    print("example2semantics:", example2semantics)
    return example2semantics


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

    query_text = """SELECT DISTINCT ?example ?semanticPredicate ?predicateText ?frame WHERE {
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

    example2semantics = defaultdict(list)
    for result in results["results"]["bindings"]:
        predicate_text = result["predicateText"]["value"]
        example = result["example"]["value"]
        semantic_predicate = result["semanticPredicate"]["value"]
        print(example + "\t" + semantic_predicate.split('#')[1] + "\t" + predicate_text)
        example2semantics[example].append(predicate_text.split()[1])
    print("example2semantics:", example2semantics)
    return example2semantics


def test_query_provenance(verb):
    query_text = """SELECT DISTINCT ?provenance WHERE {
                      ?entity rdfs:label "%s" .  # FOR the specific verb
                      ?entity rrp:provenance ?provenance .
                    }
                """ % verb

    sparql.setQuery(query_prefix + query_text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["provenance"]["value"])


if __name__ == "__main__":
    # query_verbnet_semantic_roles("admire.01")
    # query_verbnet_semantic_roles("enter.01")
    # query_verbnet_semantic_predicates("admire-31_2")
    # print(query_pb_vn_mapping_from_rdf("admire.01"))
    # print(query_pb_vn_mapping_from_rdf("enter.01"))
    # print(query_semantics_from_rdf("escape-51_1-1"))
    test_query_provenance("put-9.1-2")

