## Path pattern based

To build AMR parse cache for the training samples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --build_amr_parse_cache \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --split_type train 
```

To build AMR parse cache for the test samples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --build_amr_parse_cache \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --split_type test 
```

To mine path patterns from the training data using AMR only,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --mine_path_patterns \
    --graph_type amr \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_amr.pkl 
```

To mine path patterns from the training data using AMR+VerbNet,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --mine_path_patterns \
    --graph_type amr_verbnet \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_amr_verbnet.pkl 
```

To mine path patterns from the training data using only VerbNet,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --mine_path_patterns \
    --graph_type verbnet \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_verbnet.pkl 
```

To apply path patterns from AMR+VerbNet graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --apply_path_patterns \
    --graph_type amr_verbnet \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_amr_verbnet.pkl \
    --output_file_path ./path_output/extracted_triples_amr_verbnet.jsonl
```

To apply path patterns from AMR graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --apply_path_patterns \
    --graph_type amr \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_amr.pkl \
    --output_file_path ./path_output/extracted_triples_amr.jsonl
```

To apply path patterns from VerbNet graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm path_pattern \
    --apply_path_patterns \
    --graph_type verbnet \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_verbnet.pkl \
    --output_file_path ./path_output/extracted_triples_verbnet.jsonl
```

To compute the precision and recall of extracted triples with the ground truth,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --compute_metrics \
    --test_triple_file_path ./path_output/extracted_triples_amr.jsonl
```

## Open IE based
To apply Stanford Open IE on the train samples to mine relations,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm open_ie \
    --output_file_path ./path_output/extracted_triples_openie_train.jsonl \
    --mine_relations \
    --debug \
    --debug_dir train-debug-openie
```

To apply Stanford Open IE on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm open_ie \
    --output_file_path ./path_output/extracted_triples_openie.jsonl
```

To analyze the relations extracted by Open IE,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --algorithm open_ie \
    --analyze_relations \
    --train_triple_file_path ./path_output/extracted_triples_openie_train.jsonl \
    --test_triple_file_path ./path_output/extracted_triples_openie.jsonl \
    --out_triple_file_path ./path_output/extracted_triples_openie_converted.jsonl
```

To compute the precision and recall of extracted triples with the ground truth,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --compute_metrics \
    --test_triple_file_path ./path_output/extracted_triples_openie_converted.jsonl
```