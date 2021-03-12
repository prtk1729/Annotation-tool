import json
import os
import glob


image_paths = glob.glob('*.png')
fp = open('../1_180.json', 'r')
m = json.load(fp)
# print(m['78'])


def rename(src_folder_path, dest_folder_path, original_file_name, renamed_file_name):
    os.rename(os.path.join(src_folder_path, original_file_name),  os.path.join(dest_folder_path, renamed_file_name ))



for path in image_paths:
    k = path.split('_')[1].split('.')[0]
    renamed = f'{k}__' + str(m[str(k)].split('/')[-1].split('.')[0] ) + '.' + str(m[str(k)].split('/')[-1].split('.')[1] ) 

    # 5311995_21016_1_0__132__.png

    # print(renamed.split('__')[0])
    rename('./', './', path, renamed)

