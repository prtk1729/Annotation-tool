import json  
from glob import glob

def reset_json_file(is_os_win=1):
    '''When r'''
    # paths_of_images = glob('static/f_le/*.png')
    paths_of_images = glob('static/mnist/*.jpg')
    req_dict = {le[13:]:'-1' for le in paths_of_images}
    with open("mnist_data.json", "w") as outfile:  
        json.dump(req_dict, outfile) 

    print('\nSuccessfully reset the json file!! Can Start parsing again from scratch\n')
    return 

# uncomment when hard-reset 
# reset_json_file()