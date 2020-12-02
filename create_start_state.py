import json  
from glob import glob

# paths_of_images = glob('static/f_le/*.png')
paths_of_images = glob('static/mnist/*.jpg')
req_dict = {le[13:]:'-1' for le in paths_of_images}
with open("mnist_data.json", "w") as outfile:  
    json.dump(req_dict, outfile) 