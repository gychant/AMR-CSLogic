import requests

from app_config import config

if __name__ == '__main__':

    text = "You're in a corridor. You can see a shoe cabinet."
    params = {'text': text, 'use_coreference': 0}
    endpoint = f'http://{config.LOCAL_SERVICE_HOST}:{config.LOCAL_SERVICE_PORT}/verbnet_semantics'
    r = requests.get(endpoint, params=params)
    print(r.json())
