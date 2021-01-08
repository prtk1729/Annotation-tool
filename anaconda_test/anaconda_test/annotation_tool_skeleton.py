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


# ===========global variables==========================

plot_grid_session_iter_num = 0


# ===========global variables==========================


def start_new_session():
    plot_grid_session_iter_num = 0 


# app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]

    )

# change styling of buttons
app.layout = html.Div([
    html.Button("Start Annotation", id="start-session", n_clicks=0,
                style={'textAlign':'center','margin':'auto', 'backgroundColor': "Green", "color": "green", "margin": "5px", "padding": "5px", "display":'block'}),
    html.Button("Next", id="next",
                style={'backgroundColor': "blue", "color": "white", "margin": "5px", "padding": "5px"}),
    # html.Button("Save", n_clicks=0, id="save",
    #             style={'backgroundColor': "SlateBlue", "color": "white", "margin": "5px", "padding": "5px"}),
    html.Button("Stop Session", value='Stop Session', n_clicks=0, id="stop-session",
                style={'backgroundColor': "Tomato", "color": "white", "margin": "5px", "padding": "5px"}),
    html.Div(id="card-deck", style={"margin": "1px", "padding": "1px"}),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "0.5em 0em"})


@app.callback(
    [  Output(component_id='next', component_property='style'),
    Output(component_id='start-session', component_property='style')  ],
    Input(component_id="start-session", component_property="n_clicks"),
    prevent_initial_call = False
)
def start_session(n_clicks):
    print('start',n_clicks)
    print('\nInside START Session\n')

    if n_clicks == 0:
        return [{"display":'none'}, {'textAlign':'center','margin':'auto', 'backgroundColor': "green", "color": "white", "display":'block'}]

    if n_clicks == 1:
        return [{'textAlign':'center','margin':'auto', 'backgroundColor': "blue", "color": "white", "padding": "5px",  "display":'none', "display":'block'}, {"display":'none'}]
 


@app.callback(
    Output('card-deck', 'children'),
    Input("next", "n_clicks"),
    prevent_initial_call=True

)
def next(n_clicks):
    print(f'n_clicks: {n_clicks}')

    print('\nInside next\n')

    if n_clicks == 6:
        prevent_initial_call = False

    return prevent_initial_call

@app.callback(
    Output(component_id='stop-session', component_property='className'),
    Input(component_id="stop-session", component_property="n_clicks"),
    prevent_initial_call = True
)
def stop_session(n_clicks):
    print('\nInside Stop Session\n')
    return ""


if __name__ == '__main__':
    port = random.randrange(2000, 7999)
    # during development
    app.run_server(host='127.0.0.1', port=port, debug=True)


    # for testing
    # app.run_server(host='127.0.0.1', port=port, debug=False)
