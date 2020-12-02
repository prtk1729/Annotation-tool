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


paths_of_images = glob('static/mnist/*.jpg')
glob_idx = [i for i in range(len(paths_of_images))]
# paths_of_images = glob('static/f_le/*.png')
req_dict = {le:'-1' for le in paths_of_images}
# print(type(paths_of_images)) #list
unseen_idx_set = set([i for i in range(len(paths_of_images))])
unseen_idx_list = list(unseen_idx_set)

class_of_all_images = [-1]*len(paths_of_images) # stores the class annotations of all the images by initialising -1.
indices_of_displayed = list(range(24)) # stores indices of currently displayed images

# Here read from file logic for every new session
with open('mnist_data.json','r') as f:
# with open('fundus_data.json','r') as f:
    m = json.loads(f.read())
    for i in range(len(paths_of_images)):
        class_of_all_images[i] = m[paths_of_images[i].split("/")[-1]]

# print(class_of_all_images) #states of all the images uptil this point.

def card_body(card_id):
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
                    {'label': 'Grade 3', 'value': f"3"}
                ],
                value=str(class_of_all_images[card_id]),
                labelStyle={'display': 'inline-block', "padding": "0px 5px 0px 0px", "margin-bottom": "0px"},
                inputStyle = {"margin-right": "2px"},
                className=""
            )  
        ],  style = {"padding": "0.2rem"})
        ]

def card(card_id):
    title = "title"
    description = "desc"
    return dbc.Card(card_body(card_id), id={
                'type': 'card',
                'index': f"{card_id}" #global ids
                }, style={"height":"200px", "width": "180px"})
   

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Button("Next",    id="next", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Save",    id="save", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Div(id="card-deck"),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "1em 0em"})

def gen_cards(batch):
    return [
        dbc.Row([
            dbc.Col([card(i)]) for i in batch[:8]], justify="start"),
        dbc.Row([
            dbc.Col([card(i)]) for i in batch[8:16]], justify="center"),
        dbc.Row([
            dbc.Col([card(i)]) for i in batch[16:24]], justify="end"),
        ]

@app.callback(
    Output('card-deck', 'children'),
    Input("next", "n_clicks")
)
def next(c1):
    # if it's check_point 0
    # else if it's not predict next set of checkpoints and initialize to radiobuttons
    # For this understand working of gen_cards, card, card_body, 
    global class_of_all_images
    print(class_of_all_images)
    global indices_of_displayed
    global unseen_idx_set
    batch = list(unseen_idx_set)[:24]    
    unseen_idx_set = unseen_idx_set.difference(set(batch))
    indices_of_displayed = batch
    return gen_cards(indices_of_displayed)


@app.callback(
    Output('save', 'className'),
    Input("save", "n_clicks")
)
def save(c1):
    global class_of_all_images
    m = {}
    print(c1)
    for i in range(len(class_of_all_images)):
        m[paths_of_images[i].split("/")[-1]] = class_of_all_images[i]
    
    print(f"during save: {class_of_all_images}")
    # with open('fundus_data.json', "w+") as f:
    with open('mnist_data.json', "w") as f:
        f.write(json.dumps(m))
    print(c1)
    return ""

@app.callback(
    Output({'type': 'label-option','index': MATCH}, 'className'),
    Input({'type': 'label-option','index': MATCH}, 'value'),
    Input({'type': 'label-option','index': MATCH}, 'id')
)
def button_click(value, id):
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
    app.run_server(host='127.0.0.1', port=2320 ,debug=True)