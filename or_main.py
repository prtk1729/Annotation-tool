import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import pickle
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Button
from scipy.spatial import cKDTree
import glob
import pandas as pd
# from skimage import io

f_re = glob.glob('/data/neuroretinal/AA_Biobank/Fundus_RE/*.png')
# dataframe retrieval
df = pd.read_csv('./88242df.csv', delimiter=',')
# print(df[:5])
x = list(df['x'][:])
y = list(df['y'][:])
absolute_pts = [[x[i], y[i]] for i in range(len(x))]
absolute_pts = np.array(absolute_pts)



# on every button press go to the key of this dict and modify accordingly and load this on start of every iteration and dump on end of every iteration
idx_seen = {i:'-1' for i in range(len(f_re))}

current_20 = set({})
path = f_re
dirty_type = ['0' for i in range(len(path))]

button_list = []
subplot_axs_list = []
prev_b_point = 0
prev_g_point = 0




# print(search_space_idx, type(search_space_idx))
my_file = open("./your_file.txt", "r")
content = my_file.read()
if len(list(content)) == 0:
    search_space_idx = set([i for i in range(len(path))])
    search_list = list(search_space_idx)
else:
    ssi = list(content.split('\n'))
    ssi = [int(le) for le in ssi if le != '']
    # print(f'ssi: {ssi}')
    search_space_idx = set(ssi)
    search_list = list(search_space_idx)



def display_img(path):
    img = plt.imread(path) 
    return img

# on_click functionality is done!
def on_click(event):
    global prev_g_point
    global prev_b_point
    global dirty_type
    global idx_seen
    # print(idx_seen)

    if event.button == 1:
        plt.ion()
        print('\nYour decision has been recorded!')
        btn = event.inaxes
    
        spn = (int(btn.get_title())%80 )//4

        color_id = int(btn.get_title()) % 4

        colors_temp = ['green', 'blue', 'gray', 'orange']

        req_axis = subplot_axs_list[spn]

        autoAxis = req_axis.axis()
        rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=5, color=colors_temp[color_id])
        rec = req_axis.add_patch(rec)
        rec.set_clip_on(False)
        
        plt.draw()

        print('subplot_title: {}\n'.format(req_axis.get_title()))
        
        row_no = int(req_axis.get_title())
        dirty_val = int(btn.get_title()) % 4
        idx_seen[row_no] = str(dirty_val)




        if dirty_val == 0:
            prev_b_point = row_no
        elif dirty_val == 3:
            prev_g_point = row_no

    dirty_type[row_no] = dirty_val
    rk = spn
    rv = dirty_val%4
    d = ['{}'.format(path[int(req_axis.get_title())]), '{}'.format(rv)]
    
    field_names = ['path', 'dirty_value']
    # ------dctp------


        


def image_grid_n_selected_objects(path, req_idx, parsed_uptil):
    length = len(req_idx)

    req_obj = [display_img(path[int(idx)]) for idx in req_idx ]
    fig, axs = plt.subplots(5, 4, figsize=(6, 6))
    
    fig.suptitle('{:.2f}% parsed!'.format( float(parsed_uptil*20)/ float(len(path)) * 100 ))

    [axi.set_axis_off() for axi in axs.ravel()]
    axs = axs.flatten()
    for i in range(length):
        axs[i].set_picker(True)
    j=0

    for img, ax in zip(req_obj[:length], axs):
        ax.set_title('{}'.format(req_idx[j]), fontweight='bold', fontsize=6)
        j += 1
        ax.imshow(img)

    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95)

    btn_axes = []
    colors = []
    colors_temp = ['green', 'blue', 'gray', 'orange']
    [colors.extend(colors_temp) for i in range(20)]

    labels = ['g{}'.format(i) for i in range(length*4)] 
    for i in range(length*4):
        
        axButton1 = plt.axes([(0.1+ 0.05*(i%16)+ ((i%16)//4)*0.025) , 0.757-(i//16)*0.181, 0.019, 0.019])
        btn_axes.append(axButton1)

        btn1 = Button( ax=axButton1,
                    label=' ',
                    color=colors[i],
                    hovercolor='red'
                    )            
        axButton1.set_title('{}'.format(i), fontsize=1)
        
    axes_list = fig.axes #list of matplotlib AxesSubplot objects
    return fig, axs, btn_axes

        

def next_potential_list(prev_g_point, prev_b_point):
    '''i/p:- prev_b_point index, prev_g_point index
        o/p:- absolute req_indices'''
    global absolute_pts
    search_list = list(search_space_idx)

    filtered_pts = np.array([absolute_pts[le] for le in search_list])
    # print(f'filtered_pts: {filtered_pts}')

    points_ref = filtered_pts

    tree = cKDTree(points_ref)
    _, idx_g = tree.query(absolute_pts[prev_g_point], k=20)
    _, idx_b = tree.query(absolute_pts[prev_b_point], k=20)
    su = set(idx_g).union(set(idx_b))



    req_indices_g = [search_list[le] for le in idx_g]
    req_indices_b = [search_list[le] for le in idx_b]

    
    # print(f"idx_g:- {idx_g}")
    # print(f"idx_b:- {idx_b}")
    # print(f"abs_idx_g:- {req_indices_g}")
    # print(f"abs_idx_b:- {req_indices_b}")


    absolute_req_indices = list(set(req_indices_b).union(set(req_indices_g)))
    absolute_req_indices = absolute_req_indices[:20]
    
    return absolute_req_indices       
    


def control(dirty_type=dirty_type, path=path ):
    # print(len(dirty_type), len(path))
    # print(dirty_type[0], path[0])
    global button_list 
    global subplot_axs_list 
    global search_space_idx

    req_idx = list(range(0,len(path))) 

    j = 0
    with open('./last_checkpoint.txt') as lc:
        j = lc.read()
        # print(j)
    lc.close()
         
    j = int(j)
    
    print('\nNew session resuming with iteration number: {}'.format(j))

    

# mkm
#---------decision on what req_idx should be?-----------------
    global idx_seen
    for i in range(j, len(path)//20 + 1):
        if i == 0:
            # plt.ion()
            req_indices = req_idx[i*20:(i+1)*20] 
            current_20 = set(req_indices)

            print('iteration number: {}\n'.format(i))
            print('images corresponding to the following indices under scrutiny: {}\n'.format(current_20))
            # print(f'idx_seen after {i-1}:{idx_seen}')
            

            for le in req_indices:
                idx_seen[int(le)] = '5'
            

            file1 = open("./last_checkpoint.txt","w") 
            file1.write('{}'.format(str(i)))
            file1.close()

            fig,axs,btn_axes  = image_grid_n_selected_objects(path, req_indices, i)
            subplot_axs_list,button_list = list(axs), btn_axes
            
            fig.canvas.mpl_connect("button_press_event",on_click)
            plt.ioff()
            plt.show()
            plt.close()
            search_space_idx = search_space_idx.difference(current_20)
            ssil = list(search_space_idx)
            with open('your_file.txt', 'w') as f:
                for item in ssil:
                    f.write("%s\n" % item)

            # print(search_space_idx)

            # print(f'idx_seen after {i}:{idx_seen}')

            df = pd.DataFrame.from_dict(idx_seen, orient = 'index')
            df.to_csv('./idx_seen.csv')

        # req_idx logic
        elif (i < (len(path)//20)) and (i > 0):
            df1 = pd.read_csv("./idx_seen.csv", index_col=0)
            # print(df1)
            d = df1.to_dict("split")
            d = dict(zip(d["index"], d["data"]))
            idx_seen = d
            # print(f'type idx_seen: {type(idx_seen)}')

            req_indices = next_potential_list(prev_g_point, prev_b_point)
            # print(req_indices)

            # print(i)
            # req_indices = req_idx[i*20:(i+1)*20] 
            current_20 = set(req_indices)
            print('iteration number: {}\n'.format(i))
            print('images corresponding to the following indices under scrutiny: {}\n'.format(req_indices))
            # print(current_20)
            search_space_idx = search_space_idx.difference(current_20)
            ssil = list(search_space_idx)
            with open('your_file.txt', 'w') as f:
                for item in ssil:
                    f.write("%s\n" % item)
                    
            # print(search_space_idx)


            for le in req_indices:
                idx_seen[int(le)] = '5'

            file1 = open("./last_checkpoint.txt","w") 
            file1.write('{}'.format(str(i)))
            file1.close()


            fig,axs,btn_axes  = image_grid_n_selected_objects(path, req_indices, i)
            subplot_axs_list,button_list = list(axs), btn_axes            
            fig.canvas.mpl_connect("button_press_event",on_click)
            plt.ioff()
            plt.show()
            plt.close()
            # print(f'idx_seen after {i}:{idx_seen}')
            df = pd.DataFrame.from_dict(idx_seen, orient = 'index')
            df.to_csv('./idx_seen.csv')            

        else:
            df1 = pd.read_csv("./idx_seen.csv", index_col=0)
            d = df1.to_dict("split")
            d = dict(zip(d["index"], d["data"]))
            idx_seen = d
            # print(f'type idx_seen: {type(idx_seen)}')

            # print(i)
            # req_indices = req_idx[i*20:] 
            req_indices = list(search_space_idx)
            print('iteration number: {}\n'.format(i))
            print('images corresponding to the following indices under scrutiny: {}\n'.format(req_indices))

            current_20 = set(req_indices)
            # print(current_20)
            search_space_idx = search_space_idx.difference(current_20)
            ssil = list(search_space_idx)
            with open('your_file.txt', 'w') as f:
                for item in ssil:
                    f.write("%s\n" % item)
                    
            # print(search_space_idx)
            # print(req_indices)

            for le in req_indices:
                idx_seen[le] = '5'

            file1 = open("./last_checkpoint.txt","w") 
            file1.write('{}'.format(str(i)))
            file1.close()

            fig,axs,btn_axes  = image_grid_n_selected_objects(path, req_indices, i)
            subplot_axs_list,button_list = list(axs), btn_axes

            
    c        fig.canvas.mpl_connect("button_press_event",on_click)
            plt.ioff()
            plt.show()
            plt.close()

            # print(f'idx_seen after {i}:{idx_seen}')
            df = pd.DataFrame.from_dict(idx_seen, orient = 'index')
            df.to_csv('./idx_seen.csv')
            l = list(search_space_idx)
            with open('your_file.txt', 'w') as f:
                for item in l:
                    f.write("%s\n" % item)

control(dirty_type, path)

