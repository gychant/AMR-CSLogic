# amr-verbnet-semantics

To create a project specific virtual environment, 
```
$ . set_environment.sh
```
With this, an environment named venv will be created.

To install dependencies, run
```
$ bash scripts/install.sh
```

If we want to visualize the enhanced AMR graph, we need to install the pygraphviz package, which requires 
installing graphviz first:
```
$ conda create -n amr-verbnet python=3.7
$ conda activate amr-verbnet
$ conda install -c anaconda graphviz
$ conda deactivate
$ CONDA_HOME=/path/to/conda/home/dir pip install --global-option=build_ext --global-option="-I{$CONDA_HOME}/envs/amr-verbnet/include" --global-option="-L{$CONDA_HOME}/envs/amr-verbnet/lib" --global-option="-R{$CONDA_HOME}/envs/amr-verbnet/lib" pygraphviz
```

To download corpora and related data, run
```
$ bash scripts/download_verbnet.sh ~/nltk_data/corpora/
$ bash scripts/download_propbank.sh ~/nltk_data/corpora/
$ bash scripts/download_semlink.sh ./data
$ bash scripts/download_stanford_nlp.sh ./
```

To set PYTHONPATH, run
```
$ export PYTHONPATH=.
```

To start the service, run
```
$ export FLASK_APP=./amr_verbnet_semantics/web_app/__init__.py
$ python -m flask run --host=0.0.0.0
```
The Flask logs indicate what URL the service is running on.

To test the service, try a test example:
```
$ python amr_verbnet_semantics/test/test_service.py
```

If you want to use jupyter notebook for development, use the following commands to install the virtual environment into the jupyter notebook,
```
$ pip install ipykernel
$ python -m ipykernel install --user --name=venv
$ jupyter notebook
```
Then the notebook will be running at port 8888 by default. To open the remote notebook in your local browser, port forwarding can be used as follows
```
$ ssh -NL 8888:localhost:8888 {USER_NAME}@{SERVER_URL}
```

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

To mine path patterns from the training data,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --mine_path_patterns \
    --amr_cache_path ./data/JerichoWorld/train_amr.json 
```

To apply path patterns on the test samples to extract triples,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --apply_path_patterns \
    --amr_cache_path ./data/JerichoWorld/test_amr.json 
```

To compute the precision and recall of extracted triples with the ground truth,
```
$ python amr_verbnet_semantics/jericho_world/kg_builder.py \
    --compute_metrics \
    --triple_file_path ./path_output/extracted_triples.jsonl
```


### Using AMR parser directly
To use AMR parser directly instead of a gRPC endpoint, You have to download the model file from the following path on CCC. Then you have to unzip the file in third_party directory.
```
/dccstor/ykt-parse/SHARED/MODELS/AMR/transition-amr-parser/amr2.0_v0.4.1_youngsuk_ensemble_destillation.zip
```

Secondly you have to execute the following command. 
```
cd third_party
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -zxvf roberta.large.tar.gz
rm roberta.large.tar.gz
```
