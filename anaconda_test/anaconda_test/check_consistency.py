import json

with open('./mnist_data.json', 'r') as f:
    d = json.loads(f.read())
    print(d['img_266.jpg'])

with open('./last_checkpoint.txt', 'r') as lc:
    print(lc.read())