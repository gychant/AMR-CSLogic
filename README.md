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

To download corpora and related data, run
```
$ bash scripts/download_verbnet.sh
$ bash scripts/download_propbank.sh ~/nltk_data/corpora/
$ bash scripts/download_semlink.sh
$ bash scripts/download_stanford_nlp.sh
```

To start the service, run
```
$ export FLASK_APP=./code/service/__init__.py
$ python -m flask run --host=0.0.0.0
```
The Flask logs indicate what URL the service is running on.

To test the service, try a test example:
```
$ python code/test/test_service.py
```



