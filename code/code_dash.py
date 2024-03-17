import numpy as np
import cv2
import emoji
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

from dash_iconify import DashIconify
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
from datetime import datetime, date
import dash

from itertools import cycle

chemin_images = "./image"
df_prix = pd.read_csv("./data/prix_discipline.csv", sep=";", encoding='latin1')

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
                        dmc.AccordionPanel(
                            [
                                dmc.Button("Dés", size="lg", id="btn-des-j1", color="blue", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
                                dmc.Text("", id="result-des-j1", size="xl", color="blue", weight=700),
                                dmc.Text("", id="depense-j1", size="xl", color="blue", weight=700)
                            ],
                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}
                        ),
                    ],
                    value="joueur1",
                ),
                dmc.AccordionItem(
                    [
                        dmc.AccordionControl("Joueur 2"),
                        dmc.AccordionPanel(
                            dmc.Button("Dés", size="lg", id="btn-des-j2", color="red", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30))
                        ),
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
    html.Div(children=[
        dmc.Group([
            dmc.Button("A", id="btn-catA", size="lg", color="blue", variant="outline", n_clicks=0),
            dmc.Button("B", id="btn-catB", size="lg", color="blue", variant="outline", n_clicks=0),
            dmc.Button("C", id="btn-catC", size="lg", color="blue", variant="outline", n_clicks=0),
            dmc.Button("D", id="btn-catD", size="lg", color="blue", variant="outline", n_clicks=0)
        ])
    ], id="carte-div", style={"width": "20%", "float": "left", "text-align": "center"}),
    dcc.Store(id="x-variable", data=[1444]),
    dcc.Store(id="y-variable", data=[166]),
    dcc.Store(id="cpt", data=0),
    dcc.Store(id="somme-depense-j1", data=0)
])


#tirage dés - déplacement sur le plateau
@app.callback([Output(component_id="plateau-figure", component_property="figure"), Output("btn-des-j1", "n_clicks"),
               Output("x-variable", "data"), Output("y-variable", "data"),
               Output("cpt", "data"), Output("result-des-j1", "children")],
              [Input("btn-des-j1", "n_clicks"), Input("x-variable", "data"), 
               Input("y-variable", "data"), Input("cpt", "data")],
               prevent_initial_call=True)
def tirage_des(btn_des, x, y, c):
    global move_current
    if btn_des:
        #tirage des dés
        de_1 = random.randint(1, 6)
        de_2 = random.randint(1, 6)
        random_number = de_1 + de_2
        
        for j in range(random_number):
            x[0] += move_current[0] * 105
            y[0] += move_current[1] * 105
            c +=1
            if c % 12 == 0:
                move_current = next(move)

        fig = affichage_plateau(chemin_images, x, y, "blue")
    return fig, 0, x, y, c, f"🎲 {de_1} 🎲 {de_2}"

#affichage des cartes en fonction de la case
@app.callback(Output(component_id="carte-figure", component_property="figure"),
          Input("cpt", "data"),
        prevent_initial_call=True)
def case_carte(c):
    global move_current
    if np.array_equal(move_current, np.array([-1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 0)
    if np.array_equal(move_current, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 12 + 12)
    if np.array_equal(move_current, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 24)
    if np.array_equal(move_current, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 12 + 36)
    return fig

#dépenses places
@app.callback([Output("depense-j1", "children"), Output("somme-depense-j1", "data"),
               Output("btn-catA", "n_clicks"), Output("btn-catB", "n_clicks"),
               Output("btn-catC", "n_clicks"), Output("btn-catD", "n_clicks")],
              [Input("btn-catA", "n_clicks"), Input("btn-catB", "n_clicks"),
               Input("btn-catC", "n_clicks"), Input("btn-catD", "n_clicks"),
               Input("somme-depense-j1", "data"), Input("cpt", "data")],
        prevent_initial_call=True)
def calcul_depense(btnA, btnB, btnC, btnD, depense, c):
    #obtenir valeur de la case
    global move_current
    if np.array_equal(move_current, np.array([-1,0])):
        c = c % 12 + 0
    if np.array_equal(move_current, np.array([0,1])):
        c = c % 12 + 12
    if np.array_equal(move_current, np.array([1,0])):
        c = c % 12 + 24
    if np.array_equal(move_current, np.array([0,-1])):
        c = c % 12 + 36

    #les prix correspondants
    prix_sport = df_prix[["catA", "catB", "catC", "catD"]][df_prix["id_case"] == c].values[0].tolist()
    
    #caluler dépense
    if btnA: depense = depense + prix_sport[0]
    if btnB: depense = depense + prix_sport[1]
    if btnC: depense = depense + prix_sport[2]
    if btnD: depense = depense + prix_sport[3]

    return f"{depense} €", depense, 0, 0, 0, 0


if __name__ == '__main__':
    app.run(debug=True)


