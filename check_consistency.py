import json
from glob import glob  

paths_of_images = glob('static/mnist/*.jpg')
image_key_72 = str(paths_of_images[72][13:])
image_key_73 = str(paths_of_images[73][13:])
image_key_74 = str(paths_of_images[74][13:])

with open('mnist_data.json','r') as f:
    m = json.load(f)
    print(m[image_key_72])
    print(m[image_key_73])
    print(m[image_key_74])