# amr-verbnet-semantics

To create project specific virtual environment, 
```
$ python3 -m venv .venv
$ source .venv/bin/activate
```

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
$ pip install --global-option=build_ext --global-option="-I/home/zliang/.conda/envs/amr-verbnet/include" --global-option="-L/home/zliang/.conda/envs/amr-verbnet/lib" --global-option="-R/home/zliang/.conda/envs/amr-verbnet/lib" pygraphviz
```

To download corpora and related data, run
```
$ bash scripts/download_verbnet.sh ~/nltk_data/corpora/
$ bash scripts/download_propbank.sh ~/nltk_data/corpora/
$ bash scripts/download_semlink.sh ./data
$ bash scripts/download_stanford_nlp.sh ./
```

To start the service, run
```
$ export FLASK_APP=./code/web_app/__init__.py
$ python -m flask run --host=0.0.0.0
```
The Flask logs indicate what URL the service is running on.

To test the service, try a test example:
```
$ python code/test/test_service.py
```

To build AMR parse cache
```
python code/jericho_world/kg_builder.py --build_amr_parse_cache --split_type train
```



