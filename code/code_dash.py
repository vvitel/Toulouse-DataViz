import numpy as np
import cv2
import emoji
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import random

from dash_iconify import DashIconify
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
from datetime import datetime, date
import dash

from itertools import cycle

chemin_images = "./image"

left = np.array([-1,0])
up = np.array([0,1])
right = np.array([1,0])
down = np.array([0,-1])

move = cycle([left, up, right, down])
move_current = next(move)

#ouvrir et afficher le plateau
def affichage_plateau(path, coord_x=[1444], coord_y=[166], col="blue"):
    im = cv2.imread(f"{path}/plateau.png")
    im = im[:,:,[2,1,0]]
    #im = cv2.resize(im, (im.shape[0], im.shape[1]))
    im = cv2.flip(im, 0) 
    fig = px.imshow(im, width=im.shape[0]//2, height=im.shape[1]//2)
    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, im.shape[1]])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, im.shape[0]])
    fig.add_trace(go.Scatter(x=coord_x, y=coord_y, mode="markers", marker=dict(color=col, size=10), showlegend=False))
    #fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig

#ouvrir et afficher une carte
def affichage_carte(path, id_carte):
    im = cv2.imread(f"{path}/cartes/carte_{id_carte}.png")
    im = im[:,:,[2,1,0]]
    im = cv2.flip(im, 0)
    fig = px.imshow(im)
    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, im.shape[1]])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, im.shape[0]])
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig
    

app = Dash(__name__)

app.layout = html.Div(children=[
    html.Div([
        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
        dmc.AccordionMultiple(
            children=[
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Joueur 1"),
                        dmc.AccordionPanel([dmc.Button("Dés", size="lg", id="btn-des-j1", color="blue", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
                                            dmc.Text("", id="result-des-j1", size="xl", color="blue", weight=700)],
                                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                    ],
                    value="joueur1",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Joueur 2"),
                        dmc.AccordionPanel(dmc.Button("Dés", size="lg", id="btn-des-j2", color="red", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30))),
                    ],
                    value="joueur2",
                ),
            ],
        ),
    ], style={'width': '20%', 'float': 'left'}),
    html.Div(children=[
        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
        html.Center(dcc.Graph(figure=affichage_plateau(chemin_images), id="plateau-figure"))
    ], style={"width": "60%", "float": "left", "text-align": "center"}),
    html.Div(children=[
        html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0), id="carte-figure"))
    ], id="carte-div", style={"width": "20%", "float": "left", "text-align": "center"}),
    dcc.Store(id='x-variable', data=[1444]),
    dcc.Store(id='y-variable', data=[166]),
    dcc.Store(id='cpt', data=0)
])


#tirage dés - déplacement sur le plateau
@callback([Output(component_id="plateau-figure", component_property="figure"), Output("btn-des-j1", "n_clicks"),
        Output("x-variable", "data"), Output("y-variable", "data"), Output("cpt", "data"), Output("result-des-j1", "children")],
        [Input("btn-des-j1", "n_clicks"), Input("x-variable", "data"), Input("y-variable", "data"), Input("cpt", "data")],
        prevent_initial_call=True)
def tirage_des(btn_des, x, y, c):
    global move_current
    if btn_des:
        #tirage des dés
        de_1 = random.randint(1, 6)
        de_2 = random.randint(1, 6)
        random_number = de_1 + de_2
        
        for j in range(random_number):
            x[0] += move_current[0] * 110
            y[0] += move_current[1] * 110
            c +=1
            if c % 12 == 0:
                move_current = next(move)

        fig = affichage_plateau(chemin_images, x, y, "blue")
    return fig, 0, x, y, c, f"🎲 {de_1} 🎲 {de_2}"

#affichage des cartes en fonction de la case
@callback(Output(component_id="carte-figure", component_property="figure"),
          Input("cpt", "data"),
        prevent_initial_call=True)
def case_carte(c):
    fig = affichage_carte(chemin_images, c % 12)
    return fig


if __name__ == '__main__':
    app.run(debug=True)


