# amr-verbnet-semantics

## Install
Please read [INSTALLATION.md](./INSTALLATION.md)


## Note
- how to clear a cache
  - This implementation uses a cache to speed up the AMR parser. A snapshot is stored on disk to store the cache permanently. The file name of the snapshot takes this format `snapshot.pickle_%Y-%m-%d_%H-%M-%S`. Here is an example of a file name; `snapshot.pickle_2021-10-06_14-52-41`
  - If you want to delete snapshot files, use `bash scripts/remove_snapshot.sh /`. Then please start up the server again to clear a cache. The method is written in [INSTALLATION.md](./INSTALLATION.md). 


## How to use
To build AMR parse cache for the training samples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --build_amr_parse_cache \
    --split_type train 
```

To build AMR parse cache for the test samples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --build_amr_parse_cache \
    --split_type test 
```

To mine path patterns from the training data using AMR only,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --mine_path_patterns \
    --graph_type amr \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_amr.pkl 
```

To mine path patterns from the training data using AMR+VerbNet,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --mine_path_patterns \
    --graph_type amr_verbnet \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_amr_verbnet.pkl 
```

To mine path patterns from the training data using only VerbNet,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --mine_path_patterns \
    --graph_type verbnet \
    --amr_cache_path ./data/JerichoWorld/train_amr.json \
    --output_file_path ./path_output/patterns_train_verbnet.pkl 
```

To apply path patterns from AMR+VerbNet graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --apply_path_patterns \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_amr_verbnet.pkl \
    --output_file_path ./path_output/extracted_triples_amr_verbnet.jsonl
```

To apply path patterns from AMR graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --apply_path_patterns \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_amr.pkl \
    --output_file_path ./path_output/extracted_triples_amr.jsonl
```

To apply path patterns from VerbNet graph on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --apply_path_patterns \
    --amr_cache_path ./data/JerichoWorld/test_amr.json \
    --pattern_file_path ./path_output/patterns_train_verbnet.pkl \
    --output_file_path ./path_output/extracted_triples_verbnet.jsonl
```

To compute the precision and recall of extracted triples with the ground truth,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --compute_metrics \
    --triple_file_path ./path_output/extracted_triples.jsonl
```

