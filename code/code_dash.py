import numpy as np
import cv2
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import random

from dash_iconify import DashIconify
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
from datetime import datetime, date
import dash

from itertools import cycle

chemin_plateau = "./plateau.png"

left = np.array([-1,0])
up = np.array([0,1])
right = np.array([1,0])
down = np.array([0,-1])

move = cycle([left, up, right, down])

move_current = next(move)

des_global = 0
#global case_current
#case_current = 0

def affichage_plateau(path, coord_x=[960], coord_y=[65], col="red"):
    im = cv2.imread(path)
    fig = px.imshow(im)
    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, 1024])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, 1024])
    fig.add_trace(go.Scatter(x=coord_x, y=coord_y, mode="markers", marker=dict(color=col, size=10), showlegend=False))
    #fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig

app = Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        dmc.Group([
            dmc.Button("Dés", size="lg", id="btn-des", color="violet", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
        ], position="center")
    ]),
    html.Div(children=[
        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
        html.Center(dcc.Graph(figure=affichage_plateau(chemin_plateau), id="plateau-figure"))
    ], style={"width":"100%", "display": "inline-block", "text-align": "center"}),
    dcc.Store(id='x-variable', data=[960]),
    dcc.Store(id='y-variable', data=[65]),
    dcc.Store(id='cpt', data=0),
    dcc.Store(id='des', data=[0]),
])


@callback([Output(component_id="plateau-figure", component_property="figure"), Output("btn-des", "n_clicks"),
        Output("x-variable", "data"), Output("y-variable", "data"), Output("cpt", "data")],
        [Input("btn-des", "n_clicks"), Input("x-variable", "data"), Input("y-variable", "data"), Input("cpt", "data")],
        prevent_initial_call=True)
def tirage_des(btn_des, x, y, c):
    global move_current
    if btn_des:
        #tirage des dés
        de_1 = random.randint(1, 6)
        de_2 = random.randint(1, 6)
        random_number = de_1 + de_2
        #compteur pour tourner
        #c += random_number
        #vire_de_bord = sum(1 for i in range(1, c) if i % 10 == 0)

        print(random_number)
        
        for j in range(random_number):
            x[0] += move_current[0] * 90
            y[0] += move_current[1] * 90
            c +=1
            print(c, move_current, x[0], y[0])
            if c % 10 == 0:
                move_current = next(move)

        fig = affichage_plateau(chemin_plateau, x, y, "red")
    return fig, 0, x, y, c

if __name__ == '__main__':
    app.run(debug=True)


