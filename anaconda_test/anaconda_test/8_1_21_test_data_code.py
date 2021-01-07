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
# from create_start_state import reset_json_file
import random
import sys
from time import perf_counter
from datetime import date
import os
import signal


paths_of_images = glob('static/pool_Set/*.jpg')
class_of_all_images = [-1] * len(paths_of_images)  # stores the class annotations of all the images by initialising -1.s

# =========CLI Parsing===============
arg_container = argparse.ArgumentParser(description='Specify the Operating System')

# should be optional arguments container.add_arguments
arg_container.add_argument('--is_os_win', '-Is_Operating_System_Windows', type=int, required=True,
                           help='Is your OS Windows? (--os True) else False')

arg_container.add_argument('--initials', "-Annotator's_name_initials", type=str, required=True,
                           help='example: if your name is Prateek Pani, type "--initials pp"')

args = arg_container.parse_args()
# ====================================

# ===========global variables==========================
unseen_idx_set = set({})
unseen_idx_set_start = set({})
unseen_idx_list = []
gl_state_18 = []
gl_current_18 = []
state_18 = []
current_18 = []
batch_start_time = 0
batch_end_time = 0
glob_idx = [i for i in range(len(paths_of_images))]
time_logs = []
plot_grid_session_iter_num = 0


# ===========global variables==========================


def calculate_ann_time(batch_start_time, save_to_disk=False):
    global time_logs

    batch_end_time = perf_counter()
    time_elapsed = batch_end_time - batch_start_time
    time_logs.append(time_elapsed)

    if save_to_disk == True:
        if args.is_os_win == 0:
            path = './StatsIO/{}/{}_{}_{}'.format(name_initials, day, month, year)
            file_name = "time_logs.json"
            fp = open(os.path.join(path, file_name), 'w')
            d = {'time_logs': time_logs}
            fp.write(json.dumps(d))
        else:
            path = '.\\StatsIO\\{}\\{}_{}_{}'.format(name_initials, day, month, year)
            file_name = "time_logs.json"
            fp = open(os.path.join(path, file_name), 'w')
            d = {'time_logs': time_logs}
            fp.write(json.dumps(d))


    return batch_end_time, time_elapsed


def check_point():
    '''function reads the last_checkpoint file and returns the current iter_no'''
    with open(f'last_checkpoint_{args.initials}.txt') as lc:
        iter_no = lc.read()
    lc.close()
    iter_no = int(iter_no)
    return iter_no


iter_no = check_point()
sess_start_iter_no = iter_no
print('New Session Resuming from iteration: {}'.format(iter_no))

# =============create folder for a particular session per person per day============================
name_initials = args.initials  # Use this to make folders
today = date.today()
day, month, year = date.today().day, date.today().month, date.today().year

def create_folder(today, name_initials):
    '''Should create a folder if not present and if present print'''
    day, month, year = today.day, today.month, today.year
    if args.is_os_win == 0:
        path = './StatsIO/{}/{}_{}_{}'.format(name_initials, day, month, year)
    else:
        path = '.\\StatsIO\\{}\\{}_{}_{}'.format(name_initials, day, month, year)
    # os.makedirs(path)
    try:
        os.makedirs(path)
        print("Directory created successfully")
    except OSError as error:
        print("Directory already present")


create_folder(today, str(name_initials))


# =============xxxx particular session xxxx============================

# data retrieval for states
def read_json(today):
    global class_of_all_images
    global paths_of_images
    day, month, year = today.day, today.month, today.year

    pool_idx_list = list(range(len(paths_of_images)))
    gn_list, pp_list, gv_list = [], [], []
    pp_dict, gn_dict, gv_dict = {}, {}, {}
    for digit in range(5):
        for i in range(3):
            if i == 0:
                pp_list.extend(pool_idx_list[(i * 200 + (digit * 600)): ((i + 1) * 200 + (digit * 600))])
            elif i == 1:
                gv_list.extend(pool_idx_list[(i * 200 + (digit * 600)): ((i + 1) * 200 + (digit * 600))])
            else:
                gn_list.extend(pool_idx_list[(i * 200 + (digit * 600)): ((i + 1) * 200 + (digit * 600))])

    if args.is_os_win == 0:
        with open(f"StatsIO/{args.initials}/{day}_{month}_{year}/yest_inp_file.json", 'r') as f:  # change here only for the initials from folder @every start of session
            m = json.loads(f.read())
            if args.initials == 'pp':
                for le in pp_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            elif args.initials == 'gv':
                for le in gv_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            elif args.initials == 'gn':
                for le in gn_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            else:
                raise KeyboardInterrupt


    else:
        with open(f"StatsIO\\{args.initials}\\{day}_{month}_{year}\\yest_inp_file.json", 'r') as f:  # change here only for the initials from folder @every start of session
            m = json.loads(f.read())
            # print(m)
            if args.initials == 'pp':
                for le in pp_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            elif args.initials == 'gv':
                for le in gv_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            elif args.initials == 'gn':
                for le in gn_list:
                    class_of_all_images[int(le)] = int(m[f'img_{le}.jpg'])
            else:
                raise KeyboardInterrupt



read_json(today)


# start of session reads from your_file.txt for unseen_idx_set
def start_new_session():
    global unseen_idx_set
    global paths_of_images
    global unseen_idx_list
    global unseen_idx_set_start
    global session_start_time
    global plot_grid_session_iter_num

    plot_grid_session_iter_num = 0

    my_file = open(f"your_file_{args.initials}.txt", "r")
    content = my_file.read()
    if len(list(content)) == 0:
        unseen_idx_set = set([i for i in range(len(paths_of_images))])
        unseen_idx_list = list(unseen_idx_set)
    else:
        ssi = list(content.split('\n'))
        ssi = [int(le) for le in ssi if le != '']
        unseen_idx_set_start = set(ssi)
        unseen_idx_set = set(ssi)
        unseen_idx_list = list(unseen_idx_set)


start_new_session()

def save_functionality():
    '''On Clicking save save (1)recordings into mnist_data.json,
    (2)save unseen idx already calculated in its next call into your_file.txt'''

    global iter_no
    global class_of_all_images
    global unseen_idx_set
    global state_18
    global current_18
    global gl_state_18
    global gl_current_18

    current_18 = list(unseen_idx_set)[:18]

    m = {}

    #   modify class_of_all_images before writing
    for i in range(len(gl_current_18)):
        class_of_all_images[gl_current_18[i]] = gl_state_18[i]

    # ======= saving everything in datastructures i.e MM i.e RAM for now as a session is to be treated as an atomic event =================
    req_dict = {f'img_{i}.jpg': class_of_all_images[i] for i in range(len(class_of_all_images))}

    unseen_idx_set = unseen_idx_set.difference(set(current_18))
    ssil = list(unseen_idx_set)

    print('\nEOSAVE')
    # print(c1)
    return ""




# ======================================================================= dash app ====================================
# ========================================================================================================================================================================


# ============ In card_body() we initialize the placeholder values from the previous_day json-file ================
def card_body(card_id):
    global current_18
    global state_18
    global paths_of_images
    global class_of_all_images

    for i in range(len(current_18)):
        if current_18[i] == card_id:
            break

    #============= CHANGES HERE CHANGES THE HEIGHT AND WIDTH OF INDIVIDUAL IMAGE EMBEDDED IN A CARD.================
    return [
        dbc.CardImg(src=paths_of_images[card_id], top=True, style={"height": "120px", "width": "168px"}),
        dbc.CardBody([
            dcc.RadioItems(
                id={
                    'type': 'label-option',
                    'index': "{}".format(card_id)  # global ids
                },
                options=[

                    {'label': 'Dig_0', 'value': "0"},
                    {'label': 'Dig_1', 'value': "1"},
                    {'label': 'Dig_2', 'value': "2"},
                    {'label': 'Dig_3', 'value': "3"},
                    {'label': 'Dig_4', 'value': "4"}

                ],
                value=str(state_18[i]),
                labelStyle={'display': 'inline-block', "padding": "0px 1px 0px 1px", "margin": "1px"},
                inputStyle={"margin-right": "1px"},
                className=""
            )
        ], style={"padding": "0.05rem"})
    ]

# make changes here to get reflected in each rectangle(card) as a whole
def card(card_id):
    global current_18
    title = "title"
    description = "desc"
    return dbc.Card(card_body(card_id), id={
        'type': 'card',
        'index': "{}".format(card_id)  # global ids
    }, style={"height": "200px", "width": "170px"})


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]

    )


# html.Button("Start", id="start",
#                 style={'backgroundColor': "black", "color": "#ffffff", "margin": "5px", "padding": "5px"}),
#     html.Button("Load", id="load",
#                 style={'backgroundColor': "green", "color": "#ffffff", "margin": "5px", "padding": "5px"}),


# change styling of buttons
app.layout = html.Div([
    html.Button("Next", id="next",
                style={'backgroundColor': "blue", "color": "white", "margin": "5px", "padding": "5px"}),
    # html.Button("Save", n_clicks=0, id="save",
    #             style={'backgroundColor': "SlateBlue", "color": "white", "margin": "5px", "padding": "5px"}),
    html.Button("Export", value='Export', n_clicks=0, id="export",
                style={'backgroundColor': "Tomato", "color": "white", "margin": "5px", "padding": "5px"}),
    html.Div(id="card-deck", style={"margin": "1px", "padding": "1px"}),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "0.5em 0em"})


def gen_cards(current_18):
    global gl_state_18
    global gl_current_18

    gl_current_18 = current_18
    gl_state_18 = state_18


    return [
        dbc.Row([
            dbc.Col([card(i)]) for i in current_18[:6]]),
        dbc.Row([
            dbc.Col([card(i)]) for i in current_18[6:12]]),
        dbc.Row([
            dbc.Col([card(i)]) for i in current_18[12:18]]),
    ]


def predict_next_18_states(next_18):
    '''invoked from next i.e when "next_btn" is clicked and Uses ML to predict
    placeholders for next set of 18 points.....for now placeholders follow the following logic

    Here, I already have 18_predicted states predicted by model "yesterday" in yesterday_file. Just read the file and
    associate |yesterday_set|//3 inp-labels.json and assign to state_18'''
    global state_18
    global class_of_all_images

    state_18 = [class_of_all_images[int(le)] for le in next_18]
    return state_18


def most_confused_18():
    '''invoked from next ; calculates most confused 18 images based on HITL....
    for now random 18 from unseen_idx_set

    Here, I already have 18_predicted indices predicted by model "yesterday" in yesterday_file. Just read the file and
    associate |yesterday_set|//3 inp-labels.json and assign to next_18'''
    global unseen_idx_set
    global class_of_all_images

    # read file here now I have deterministically (|yesterday_set|//3) idxs Of the "new_indices" put batches of 18 here.
    next_18 = list(unseen_idx_set)[:18]

    return next_18

# @app.callback(
#     Output('load', 'children'),
#     Input("load", "n_clicks"),
#     prevent_initial_call=True
#
# )
# def load(c1):
#     '''calculates next set of 18 indices and assigns placeholders to these before loading
#     invoke most_confused_18 and then predict_next_18_states'''
#     global class_of_all_images
#     global unseen_idx_set
#     global current_18
#     global state_18
#     global iter_no
#     global paths_of_images
#     global batch_start_time
#     global batch_end_time
#
#     batch_start_time = perf_counter()
#
#     # Only save the logs in MM not to disk yet
#     # batch_start_time, time_elapsed = calculate_ann_time(batch_start_time, save_to_disk=False)
#
#     # check iter_no
#     if int(iter_no) >= ((1000 // 18) + 1):
#         sys.exit()
#         # resets to start_state
#         file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
#         file1.write('0')
#         file1.close()
#         print("\nlast_checkpoint file reset to 0. Do a reality check\n")
#
#         # before reseting mnist_json/fundus_json files 1st copy the recordings to another file
#         with open("mnist_data.json", "r") as f1, open("previous_recordings.json", "w") as f2:
#             f2.write(f1.read())
#         reset_json_file(args.is_os_win)
#         print("\n\n\n ======= No more images to be parsed. Stop the session to start from scratch again. \n\n\n")
#
#         unseen_idx_set = set([i for i in range(len(paths_of_images))])
#
#         # ======= reading the mnist_data.json file updated the previous day for this person ====================
#         read_json() #modifies class_of_all_images globally
#         # =======================================================================================================
#
#
#         iter_no = check_point()
#         print('New Session Resuming from iteration: {}'.format(iter_no))
#
#         start_new_session()
#
#         current_18 = most_confused_18()
#         # print('inside next current_24: ', current_24)
#         state_18 = predict_next_18_states(current_18)
#
#     else:
#         # no need to read from your_file.txt
#         current_18 = most_confused_18()  # next_24 will be current_24 for next iter
#         # print(f"inside next() with current_24: {current_24}")
#         state_18 = predict_next_18_states(current_18)  # returned state list of these current_24 points
#         # print(f"inside next() with state_24: {state_24}")
#
#     return gen_cards(current_18)




@app.callback(
    Output('card-deck', 'children'),
    Input("next", "n_clicks"),
    prevent_initial_call=True

)
def next(c1):
    '''calculates next set of 18 indices and assigns placeholders to these before loading
    invoke most_confused_18 and then predict_next_18_states'''
    global class_of_all_images
    global unseen_idx_set
    global current_18
    global state_18
    global iter_no
    global paths_of_images
    global batch_start_time
    global batch_end_time
    global plot_grid_session_iter_num

    plot_grid_session_iter_num += 1
    if plot_grid_session_iter_num >= 2:
        save_functionality()
    # Only save the logs in MM not to disk yet
    batch_start_time, time_elapsed = calculate_ann_time(batch_start_time, save_to_disk=True) #inside this fn def per_counter


    if int(iter_no) < ((1000 // 18) + 1):
        current_18 = most_confused_18()  # next_24 will be current_24 for next iter
        state_18 = predict_next_18_states(current_18)  # returned state list of these current_24 points

    return gen_cards(current_18)


# @app.callback(
#     Output(component_id='save', component_property='className'),
#     Input(component_id="save", component_property="n_clicks"),
#     prevent_initial_call=True
# )
# def save(n_clicks):
#     '''On Clicking save save (1)recordings into mnist_data.json,
#     (2)save unseen idx already calculated in its next call into your_file.txt'''

#     global iter_no
#     global class_of_all_images
#     global unseen_idx_set
#     global state_18
#     global current_18
#     global gl_state_18
#     global gl_current_18

#     current_18 = list(unseen_idx_set)[:18]

#     m = {}

#     #   modify class_of_all_images before writing
#     for i in range(len(gl_current_18)):
#         class_of_all_images[gl_current_18[i]] = gl_state_18[i]

#     # ======= saving everything in datastructures i.e MM i.e RAM for now as a session is to be treated as an atomic event =================
#     req_dict = {f'img_{i}.jpg': class_of_all_images[i] for i in range(len(class_of_all_images))}

#     unseen_idx_set = unseen_idx_set.difference(set(current_18))
#     ssil = list(unseen_idx_set)

#     print('\nEOSAVE')
#     # print(c1)
#     return ""


@app.callback(
    Output(component_id='export', component_property='className'),
    Input(component_id="export", component_property="n_clicks"),
    prevent_initial_call = True
)
def stop_session(n_clicks):
    '''Only on clicking i.e n_clicks>=1 export button would work not from starting when the app runs'''
    global batch_start_time

    batch_start_time, time_elapsed = calculate_ann_time(batch_start_time, save_to_disk=True)

    global iter_no
    global class_of_all_images
    global unseen_idx_set
    global state_18
    global current_18
    global gl_state_18
    global gl_current_18
    global name_initials
    global today
    global unseen_idx_set_start

    day, month, year = today.day, today.month, today.year
    current_18 = list(unseen_idx_set)[:18]


    m = {}

    #   modify class_of_all_images before writing
    for i in range(len(gl_current_18)):
        class_of_all_images[gl_current_18[i]] = gl_state_18[i]

    # ======= saving everything in datastructures i.e MM i.e RAM for now as a session is to be treated as an atomic event =================
    req_dict = {f'img_{i}.jpg': class_of_all_images[i] for i in range(len(class_of_all_images))}

    unseen_idx_set = unseen_idx_set.difference(set(current_18))
    ssil = list(unseen_idx_set)

    # =======================================================================================================================================

    # ================ saving everything in secondary memory (Disk) at StatsIO/{initials}/{day_month_year} ==========================================================================
    # create a dict1 and dump here. How to create idx from class_of_all_images and from index we know the img_name?

    # ======== save o/p files ===============================
    if args.is_os_win == 0:
        with open(file='./StatsIO/{}/{}_{}_{}/mnist_uptil_today_out_files.json'.format(name_initials, day, month, year),  mode="w") as f:
            f.write(json.dumps(req_dict))

    else:
        with open(file='.\\StatsIO\\{}\\{}_{}_{}/mnist_uptil_today_out_files.json'.format(name_initials, day, month, year),  mode="w") as f:
            f.write(json.dumps(req_dict))


    # save the idx_unseen uptil now
    with open(f'your_file_{args.initials}.txt', 'w') as f:
        for item in ssil:
            f.write("%s\n" % item)

    # save the last_checkpoint for this user so as to start from correct point the next time
    iter_no += 1
    file1 = open(f"last_checkpoint_{args.initials}.txt", "w")
    file1.write('{}'.format(str(iter_no)))
    file1.close()


    # create the idx-state file annotated_today
    my_file = open(f"your_file_{args.initials}.txt", "r")
    content = my_file.read()
    ssi = list(content.split('\n'))
    ssi = [int(le) for le in ssi if le != '']
    unseen_idx_set_next = set(ssi)
    idx_set_annotated_today = unseen_idx_set_start.difference(unseen_idx_set_next)
    images_annotated_today_dict = {f'img_{int(idx)}.jpg': req_dict[f'img_{int(idx)}.jpg'] for idx in idx_set_annotated_today}
    if args.is_os_win == 0:
        with open(file='./StatsIO/{}/{}_{}_{}/images_annotated_today.json'.format(name_initials, day, month, year),mode="w") as f:
            f.write(json.dumps(images_annotated_today_dict))
    else:
        with open(file='.\\StatsIO\\{}\\{}_{}_{}/images_annotated_today.json'.format(name_initials, day, month,year), mode="w") as f:
            f.write(json.dumps(images_annotated_today_dict))



    print(f'\n{(1 - (len(unseen_idx_set)/1000)) *100:.1f}% of the images parsed\n')
    print('\nSESSION COMPLETE!!\n')
    os.kill(os.getpid(), signal.SIGTERM)
    # ==========================================================================================================================================

    return ""


@app.callback(
    Output({'type': 'label-option', 'index': MATCH}, 'className'),
    Input({'type': 'label-option', 'index': MATCH}, 'value'),
    Input({'type': 'label-option', 'index': MATCH}, 'id')
)
def button_click(value, id):
    '''What are the states of the radio buttons clicked?'''
    # I have the value of the recording here ,
    # Manipulate here and store in a datastructure and fire when save is clicked
    # print(f"val: {value}, id: {id}")

    global class_of_all_images
    global glob_idx
    global current_18
    global state_18
    class_of_all_images[int(id['index'])] = int(value)
    print("class of {} set to {}".format(id['index'], value))
    for i in range(len(current_18)):
        if int(current_18[i]) == int(id['index']):
            break
    state_18[i] = int(value)
    return ""


if __name__ == '__main__':
    port = random.randrange(2000, 7999)
    # during development
    # app.run_server(host='127.0.0.1', port=port, debug=True)


    # for testing
    app.run_server(host='127.0.0.1', port=port, debug=True)


