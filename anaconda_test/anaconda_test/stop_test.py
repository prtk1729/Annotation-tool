import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
import random
import random
import sys
import os

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Button("Start",    id="start", style={'backgroundColor': "black", "color": "#ffffff", "margin": "20px", "padding": "10px"}),
    html.Button("Load",    id="load", style={'backgroundColor': "green", "color": "#ffffff", "margin": "20px", "padding": "10px"}),
    html.Button("Next",    id="next", style={'backgroundColor': "blue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Save",   n_clicks=0, id="save", style={'backgroundColor': "SlateBlue", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Button("Export", value='Export', n_clicks=0,    id="export",  style={'backgroundColor': "Tomato", "color": "white", "margin": "20px", "padding": "10px"}),
    html.Div(id="card-deck"),
    html.H3(id='button-clicks'),

], style={'text-align': "center", "margin": "1em 0em"})


#

if __name__ == '__main__':
    port = random.randrange(2000, 7999)
    app.run_server(host='127.0.0.1', port=port ,debug=True)