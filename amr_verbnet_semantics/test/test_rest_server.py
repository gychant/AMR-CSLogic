import argparse

import requests

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=3000)
args = parser.parse_args()

endpoint = 'http://' + args.host + ':%d/process' % args.port

text = 'You can make out a clothesline.'
text = "You're in a corridor. You can see a shoe cabinet. The shoe cabinet contains a top hat. You lean against the wall, inadvertently pressing a secret button. The wall opens up to reveal a coat hanger. The coat hanger is normal. But the thing is empty. You can make out a hat rack. Looks like someone's already been here and taken everything off it, though. Hm. Oh well You hear a noise behind you and spin around, but you can't see anything other than an umbrella stand. The umbrella stand is usual. But the thing is empty, unfortunately. Hm. Oh well You hear a noise behind you and spin around, but you can't see anything other than a key holder! The key holder is ordinary. But the thing is empty. It would have been so cool if there was stuff on the key holder."
r = requests.post(endpoint, data={'text': text})


print(r.json())
