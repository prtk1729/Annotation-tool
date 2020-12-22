import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import random
from glob import glob
import json
## Global variables 

from create_start_state import reset_json_file

unseen_idx_set = set({})
unseen_idx_list = []

def check_point():
    '''function reads the last_checkpoint file and returns the current iter_no'''
    with open('./last_checkpoint.txt') as lc:
        iter_no = lc.read()
    lc.close()
    iter_no = int(iter_no)
    return iter_no

iter_no = check_point()
print(f'New session Resuming from iteration: {iter_no}')
# here write Load button logic




paths_of_images = glob('static/mnist/*.jpg')
glob_idx = [i for i in range(len(paths_of_images))]


class_of_all_images = [-1]*len(paths_of_images) # stores the class annotations of all the images by initialising -1.
# data retrieval for states
with open('mnist_data.json','r') as f:
# with open('fundus_data.json','r') as f:
    m = json.loads(f.read())
    for i in range(len(paths_of_images)):
        class_of_all_images[i] = m[paths_of_images[i].split("/")[-1]]
# print(class_of_all_images) #states of all the images uptil this point.



req_dict = {le:'-1' for le in paths_of_images} #for this load dict



# start of session reads from your_file.txt for unseen_idx_set
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




def card_body(card_id):
    global current_24
    global state_24
    global paths_of_images

    for i in range(len(current_24)):
        if current_24[i]==card_id:
            break
    # print(f"inside card_body with current_24[i]: {current_24[i]}")
    return [
        dbc.CardImg(src=paths_of_images[card_id], top=True, style={"height":"150px", "width": "180px"}),
        dbc.CardBody([                
            dcc.RadioItems(
                id={
                'type': 'label-option',
                'index': f"{card_id}" #global ids
                },
                options=[

                    {'label': 'Normal', 'value': f"0"},
                    {'label': 'Grade 1', 'value': f"1"},
                    {'label': 'Grade 2', 'value': f"2"},
                    {'label': 'Grade 3', 'value': f"3"},
                    {'label': 'POOR QUALITY', 'value': f"4"}

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
                'index': f"{card_id}" #global ids
                }, style={"height":"222px", "width": "180px"})
   

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Button("Next",    id="next", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Save",    id="save", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Div(id="card-deck"),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "1em 0em"})

def gen_cards(current_24):
    print(f"inside gen_cards with current_24: {current_24}")
    # l = [i+16 for i in range(len(current_24[:8])) ]
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
    placeholders for next set of 24 points.....for now placeholders follow the following logic'''
    global state_24

    state_24 = [le%5 for le in next_24]
    # print(f"inside predict_next_24_states() with state_24: {state_24}")
    return state_24

def most_confused_24():
    '''invoked from next ; calculates most confused 24 images based on HITL....
    for now random 24 from unseen_idx_set'''
    global unseen_idx_set
    next_24 = list(unseen_idx_set)[:24]
    # print(f"inside most_confused() with next_24: {next_24}")
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

    # check iter_no
    if int(iter_no) >= (len(paths_of_images)//24 + 1):
        file1 = open("./last_checkpoint.txt","w") 
        file1.write('0')
        file1.close()

        # before reseting mnist_json/fundus_json files 1st copy the recordings to another file
        with open("mnist_data.json", "r") as f1, open("previous_recordings.json", "w") as f2:
            f2.write(f1.read())
        reset_json_file()
        print("\n\n\n ======= No more images to be parsed. Needs a Hard reset to iterate from scratch again\n\n\n")

        return

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
    current_24 = list(unseen_idx_set)[:24]    

    m = {}
    print(c1)

    # Save recordings
    for i in range(len(class_of_all_images)):
        m[paths_of_images[i].split("/")[-1]] = class_of_all_images[i]
    with open('mnist_data.json', "w") as f:
        f.write(json.dumps(m))

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

    # print(c1)
    return ""

@app.callback(
    Output({'type': 'label-option','index': MATCH}, 'className'),
    Input({'type': 'label-option','index': MATCH}, 'value'),
    Input({'type': 'label-option','index': MATCH}, 'id')
)
def button_click(value, id):
    '''What are the states of the radio buttons clicked?'''
    # I ahve the value of the recording here , 
    # Manipulate here and store in a datastructure and fire when save is clicked
    # print(f"val: {value}, id: {id}")

    global glob_idx
    global req_dict
    class_of_all_images[int(id['index'])] = int(value)
    # req_dict[ int(id['index'] 
    print(f"class of {id['index']} set to {value}")
    # print(class_of_all_images)
    print(id)
    return ""


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=4870 ,debug=True)