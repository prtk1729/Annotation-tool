import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import random
from glob import glob
import json
import argparse
from create_start_state import reset_json_file
import random
import sys
from time import perf_counter
from datetime import date
import os






# =========CLI Parsing===============
arg_container = argparse.ArgumentParser(description='Specify the Operating System')

# should be optional arguments container.add_arguments
arg_container.add_argument('--is_os_win', '-Is_Operating_System_Windows', type=int, required=True, help='Is your OS Windows? (--os True) else False')

# container.parse_args() and store in args
args = arg_container.parse_args()
# ====================================
# print(args.is_os_win)


# ===========global variables==========================
unseen_idx_set = set({})
unseen_idx_list = []
gl_state_24 = []
gl_current_24 = []
state_24 = []
# session_start_time = 0
batch_start_time = 0
batch_end_time = 0
paths_of_images = glob('static/mnist/*.jpg')
class_of_all_images = [-1]*len(paths_of_images) # stores the class annotations of all the images by initialising -1.s
glob_idx = [i for i in range(len(paths_of_images))]
time_logs = []

# ===========global variables==========================




def calculate_ann_time(batch_start_time):
    # calculate batch_end_time, print b/w next time write into file file and return batch end_time
    global time_logs

    batch_end_time = perf_counter()
    # print(batch_end_time - batch_start_time, "\nin SECs\n")
    time_elapsed = batch_end_time - batch_start_time
    time_logs.append(time_elapsed)
    path = './StatsIO/{}/{}_{}_{}'.format(name_initials,day, month, year)
    file_name = "time_logs.json"
    fp = open(os.path.join(path, file_name), 'w')
    d = {'time_logs':time_logs}
    fp.write(json.dumps(d))

    return batch_end_time, time_elapsed




def check_point():
    '''function reads the last_checkpoint file and returns the current iter_no'''
    with open('./last_checkpoint.txt') as lc:
        iter_no = lc.read()
    lc.close()
    iter_no = int(iter_no)
    return iter_no

iter_no = check_point()
sess_start_iter_no = iter_no
print('New Session Resuming from iteration: {}'.format(iter_no))




# =============create folder for a particular session per person per day============================
name_initials = input("Enter your name initials: ") #Use this to make folders
today = date.today()
day, month, year = date.today().day, date.today().month,  date.today().year

# create a folder for this person if not exist and navigate into that if it doesn't exist
# def create_file(today, name_initials, iter_no):
#     
#     pass


def create_folder(today, name_initials):
    # the input files will contain predicted values of 72 images from pool set
    # expected output file will contain actual annotations of converted_train_set
    # file_name_format: {name_initials}_{day}_{month}_{year}_{i/o}.json
    day, month, year = today.day, today.month, today.year
    if args.is_os_win == 0:
        path = './StatsIO/{}/{}_{}_{}'.format(name_initials,day, month, year)
    else:
        path = '.\\StatsIO\\{}/{}_{}_{}'.format(name_initials,day, month, year)
    # os.makedirs(path)
    try:
        os.makedirs(path)
        print("Directory created successfully" )
    except OSError as error:
        print("Directory already present")

create_folder(today, str(name_initials))


def create_file(path):
    pass

# =============xxxx particular session xxxx============================




# print(paths_of_images[0])
# print(paths_of_images[0].split("/")[-1])



# data retrieval for states
def read_json():
    global class_of_all_images
    with open('mnist_data.json','r') as f:
    # with open('fundus_data.json','r') as f:
        m = json.loads(f.read())
        if args.is_os_win:
            for i in range(len(paths_of_images)):
                class_of_all_images[i] = m[paths_of_images[i].split("\\")[-1]]
        else:
            for i in range(len(paths_of_images)):
                class_of_all_images[i] = m[paths_of_images[i].split("/")[-1]]
    

# print(class_of_all_images) #states of all the images uptil this point.



req_dict = {le:'-1' for le in paths_of_images} #for this load dict



    
# ================ start timer after asking for a_name =======================================================
# ============ Just after preprocessing and just before start_new_session() ==========
batch_start_time = perf_counter()
# ====================================================================================



# start of session reads from your_file.txt for unseen_idx_set
def start_new_session():
    global unseen_idx_set
    global paths_of_images
    global unseen_idx_list
    global session_start_time

    my_file = open("./your_file.txt", "r")
    content = my_file.read()
    if len(list(content)) == 0:
        # handle case when iter_no != 0 and len(list(content)) == 0: inconsistent case return with prompt.
        unseen_idx_set = set([i for i in range(len(paths_of_images))])
        unseen_idx_list = list(unseen_idx_set)
    else:
        ssi = list(content.split('\n'))
        ssi = [int(le) for le in ssi if le != '']
        # print(f'ssi: {ssi}')
        unseen_idx_set = set(ssi)
        unseen_idx_list = list(unseen_idx_set)


start_new_session()


def card_body(card_id):
    global current_24
    global state_24
    global paths_of_images
    global class_of_all_images


    for i in range(len(current_24)):
        if current_24[i]==card_id:
            break
    # print(current_24[i])
    # print(state_24[i])
    # print(f'\nInside card_body() {str(state_24[i])}\n')

    # class_of_all_images[current_24[i]] = state_24[i]
    # print(f'{i} {current_24[i]}, {class_of_all_images[current_24[i]]}')
    # print(f"inside card_body with current_24[i]: {current_24[i]}")
    return [
        dbc.CardImg(src=paths_of_images[card_id], top=True, style={"height":"150px", "width": "180px"}),
        dbc.CardBody([                
            dcc.RadioItems(
                id={
                'type': 'label-option',
                'index': "{}".format(card_id) #global ids
                },
                options=[

                    {'label': 'Normal', 'value': "0"},
                    {'label': 'Grade 1', 'value': "1"},
                    {'label': 'Grade 2', 'value': "2"},
                    {'label': 'Grade 3', 'value': "3"},
                    {'label': 'POOR QUALITY', 'value': "4"}

                ],
                value=str(state_24[i]),
                labelStyle={'display': 'inline-block', "padding": "0px 5px 0px 2px", "margin-bottom": "1px"},
                inputStyle = {"margin-right": "2px"},
                className=""
            )  
        ],  style = {"padding": "0.2rem"})
        ]

def card(card_id):
    # print('hi')
    global current_24
    # print(f"inside card(): {current_24[int(i)]}")
    title = "title"
    description = "desc"
    return dbc.Card(card_body(card_id), id={
                'type': 'card',
                'index': "{}".format(card_id) #global ids
                }, style={"height":"222px", "width": "180px"})
   

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Button("Load",    id="load", style={'backgroundColor': "green", "color": "#ffffff", "margin": "20px", "padding": "10px"}),
    html.Button("Next",    id="next", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Save",    id="save", style={'backgroundColor': "SlateBlue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Export",    id="export", style={'backgroundColor': "Tomato", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Div(id="card-deck"),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "1em 0em"})

def gen_cards(current_24):
    global gl_state_24
    global gl_current_24
    # print("inside gen_cards with current_24: {}".format(current_24))
    # l = [i+16 for i in range(len(current_24[:8])) ]
    # cur_states = [current_24[int(le)] for le in current_24]
    # print('hi')
    # print('states',cur_states)
    # print('hi')

    # print(state_24)

    gl_current_24 = current_24
    gl_state_24 = state_24

    # print('Inside gen_cards() gl_24')
    # print(gl_state_24, gl_current_24)


    # print(l)
    return [
        dbc.Row([
            dbc.Col([card(i)]) for i in current_24[:8]], justify="start"),
        dbc.Row([
            dbc.Col([card(i)]) for i in current_24[8:16]], justify="center"),
        dbc.Row([
            dbc.Col([card(i)]) for i in current_24[16:24]], justify="end"),
        ]

def predict_next_24_states(next_24):
    '''invoked from next i.e when "next_btn" is clicked and Uses ML to predict
    placeholders for next set of 24 points.....for now placeholders follow the following logic

    Here, I already have 24_predicted states predicted by model "yesterday" in yesterday_file. Just read the file and
    associate |yesterday_set|//3 inp-labels.json and assign to state_24'''
    global state_24

    state_24 = [le%5 for le in next_24]
    # print(f"inside predict_next_24_states() with state_24: {state_24}")
    return state_24

def most_confused_24():
    '''invoked from next ; calculates most confused 24 images based on HITL....
    for now random 24 from unseen_idx_set
    
    Here, I already have 24_predicted indices predicted by model "yesterday" in yesterday_file. Just read the file and
    associate |yesterday_set|//3 inp-labels.json and assign to next_24'''
    global unseen_idx_set
    global class_of_all_images

    # read file here now I have deterministically (|yesterday_set|//3) idxs Of the "new_indices" put batches of 24 here.
    next_24 = list(unseen_idx_set)[:24]


    # print(f"inside most_confused() with next_24: {next_24}")
    # print('\ninside most_confused()\n')

    # print([class_of_all_images[next_24[i]] for i in range(len(next_24))])

    # print('\ninside most_confused()\n')

    return next_24


@app.callback(
    Output('card-deck', 'children'),
    Input("next", "n_clicks")
)
def next(c1):
    '''calculates next set of 24 indices and assigns placeholders to these before loading
    invoke most_confused_24 and then predict_next_24_states'''
    global class_of_all_images
    global unseen_idx_set
    global current_24
    global state_24
    global iter_no
    global paths_of_images
    global batch_start_time
    global batch_end_time


    batch_start_time,time_elapsed = calculate_ann_time(batch_start_time) 

    

    # check iter_no
    if int(iter_no) >= (len(paths_of_images)//24 + 1):
        # resets to start_state
        file1 = open("./last_checkpoint.txt","w") 
        file1.write('0')
        file1.close()
        print("\nlast_checkpoint file reset to 0. Do a reality check\n")

        # before reseting mnist_json/fundus_json files 1st copy the recordings to another file
        with open("mnist_data.json", "r") as f1, open("previous_recordings.json", "w") as f2:
            f2.write(f1.read())
        reset_json_file(args.is_os_win)
        print("\n\n\n ======= No more images to be parsed. Stop the session to start from scratch again. \n\n\n")
        
        unseen_idx_set = set([i for i in range(len(paths_of_images))])
        read_json()
        iter_no = check_point()
        print('New Session Resuming from iteration: {}'.format(iter_no))

        start_new_session()

        current_24 = most_confused_24()
        # print('inside next current_24: ', current_24)
        state_24 = predict_next_24_states(current_24) 

    else:
        # no need to read from your_file.txt
        current_24 = most_confused_24() #next_24 will be current_24 for next iter
        # print(f"inside next() with current_24: {current_24}")
        state_24 = predict_next_24_states(current_24)   #returned state list of these current_24 points 
        # print(f"inside next() with state_24: {state_24}")

    return gen_cards(current_24)


@app.callback(
    Output('save', 'className'),
    Input("save", "n_clicks")
)
def save(c1):
    '''On Clicking save save (1)recordings into mnist_data.json, 
    (2)save unseen idx already calculated in its next call into your_file.txt'''

    global iter_no
    global class_of_all_images
    global unseen_idx_set
    global state_24
    global current_24
    global gl_state_24
    global gl_current_24
    
    current_24 = list(unseen_idx_set)[:24]    
    print('\nInside save()\n')
    print(gl_state_24, gl_current_24)

    m = {}
    print(c1)

#   modify class_of_all_images before writing
    for i in range(len(gl_current_24)):
        class_of_all_images[gl_current_24[i]] = gl_state_24[i]

    # print('\nclass\n',class_of_all_images)
    
    req_dict = {f'img_{i}.jpg':class_of_all_images[i] for i in range(len(class_of_all_images)) }
    print('req_dict')
    print('\n',req_dict)
    print('req_dict')
    # Save recordings
    # for i in range(len(class_of_all_images)):
    #     m[paths_of_images[i].split("/")[-1]] = class_of_all_images[i]

    # print('\nm\n',m)
    # print('json',m)

    # create a dict1 and dump here. How to create idx from class_of_all_images and from index we know the img_name?
    with open('mnist_data.json', "w") as f:
        f.write(json.dumps(req_dict))

    # Q) Which is optimized a new_file your_file.txt and load and read everytime 
    # or deduce everything from mnist_data.json
    # Save unseen_idx_set calculated in previous next
    unseen_idx_set = unseen_idx_set.difference(set(current_24))
    ssil = list(unseen_idx_set)
    with open('your_file.txt', 'w') as f:
        for item in ssil:
            f.write("%s\n" % item)

    # write next iteration number onto last_checkpoint file
    iter_no += 1
    file1 = open("./last_checkpoint.txt","w") 
    file1.write('{}'.format(str(iter_no)))
    file1.close()

    print('\nEOSAVE')
    # print(c1)
    return ""


@app.callback(
    Output({'type': 'label-option','index': MATCH}, 'className'),
    Input({'type': 'label-option','index': MATCH}, 'value'),
    Input({'type': 'label-option','index': MATCH}, 'id')
)
def button_click(value, id):
    '''What are the states of the radio buttons clicked?'''
    # I have the value of the recording here , 
    # Manipulate here and store in a datastructure and fire when save is clicked
    # print(f"val: {value}, id: {id}")

    global class_of_all_images
    global glob_idx
    global current_24
    global state_24
    class_of_all_images[int(id['index'])] = int(value)
    # req_dict[ int(id['index'] 
    print("class of {} set to {}".format( id['index'],  value))
    print(f"class_of_all_images[{int(id['index'])}]", class_of_all_images[int(id['index'])])

    for i in range(len(current_24)):
        if int(current_24[i]) == int(id['index']):
            break
    state_24[i] = int(value)
    

    # print(class_of_all_images)
    # print(id)
    return ""


if __name__ == '__main__':
    port = random.randrange(2000, 7999)
    app.run_server(host='127.0.0.1', port=port ,debug=True)
