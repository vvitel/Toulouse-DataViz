import numpy as np
import cv2
import emoji
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

from dash_iconify import DashIconify
from dash import Dash, html, dcc, callback, Output, Input, State
import dash

from itertools import cycle

chemin_images = "./image"
df_prix = pd.read_csv("./data/prix_discipline.csv", sep=";", encoding='latin1')

txt1 = "Voilà les règles de notre jeu c'est la fiesta c'est les copains"
txt2 = "Oh c'est génial musique maestro "

left = np.array([-1,0])
up = np.array([0,1])
right = np.array([1,0])
down = np.array([0,-1])

move_j1, move_j2 = cycle([left, up, right, down]), cycle([left, up, right, down])
move_current_j1, move_current_j2 = next(move_j1), next(move_j2)

#ouvrir et afficher le plateau
def affichage_plateau(path, coord_x1=[1504], coord_y1=[105], coord_x2=[1504], coord_y2=[55], col1="blue", col2="red"):
    im = cv2.imread(f"{path}/plateau.jpg")
    im = im[:,:,[2,1,0]]
    #im = cv2.resize(im, (im.shape[0], im.shape[1]))
    im = cv2.flip(im, 0) 
    fig = px.imshow(im, width=im.shape[0]//2, height=im.shape[1]//2)
    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, im.shape[1]])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, im.shape[0]])
    fig.add_trace(go.Scatter(x=coord_x1, y=coord_y1, mode="markers", marker=dict(color=col1, size=20), showlegend=False))
    fig.add_trace(go.Scatter(x=coord_x2, y=coord_y2, mode="markers", marker=dict(color=col2, size=20), showlegend=False))    
    #fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig

#ouvrir et afficher une carte
def affichage_carte(path, id_carte):
    im = cv2.imread(f"{path}/cartes/olympiques/carte_{id_carte}.png")
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
    dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.Tab("Jeux Olympiques", value="tab1"),
                    dmc.Tab("Jeux Paralympiques", value="tab2"),
                ],
                grow=True
            ),
            dmc.TabsPanel(
                [
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
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0), id="carte-figure-j1")),
                                dmc.ButtonGroup(
                                    [
                                        dmc.Button("A", id="btn-catA-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("B", id="btn-catB-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("C", id="btn-catC-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("D", id="btn-catD-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                    ]
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                dmc.Modal(title="Règles du Jeu", id="modal-regles", size="55%", zIndex=10000, children=[dmc.Text(f"{txt1}\n{txt2}")]),
                                html.Center(dmc.Button("Règles du Jeu", id="modal-regles-btn", size="lg", color="violet"))
                            ],
                        ),
                    ], style={'width': '20%', 'float': 'left'}),
                    html.Div(children=[
                        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                        html.Center(dcc.Graph(figure=affichage_plateau(chemin_images), id="plateau-figure"))
                    ], style={"width": "60%", "float": "left", "text-align": "center"}),
                    html.Div([
                        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                        dmc.AccordionMultiple(
                            children=[
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl("Joueur 2"),
                                        dmc.AccordionPanel(
                                            [
                                                dmc.Button("Dés", size="lg", id="btn-des-j2", color="red", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
                                                dmc.Text("", id="result-des-j2", size="xl", color="red", weight=700),
                                                dmc.Text("", id="depense-j2", size="xl", color="red", weight=700)
                                            ],
                                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}
                                        ),
                                    ],
                                    value="joueur2",
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0), id="carte-figure-j2")),
                                dmc.ButtonGroup(
                                    [
                                        dmc.Button("A", id="btn-catA-j2", size="xl", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("B", id="btn-catB-j2", size="xl", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("C", id="btn-catC-j2", size="xl", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("D", id="btn-catD-j2", size="xl", color="red", variant="outline", n_clicks=0),
                                    ]
                                )
                            ],
                        ),
                    ], style={'width': '20%', 'float': 'left'}),
                    dcc.Store(id="x-variable-j1", data=[1504]),
                    dcc.Store(id="x-variable-j2", data=[1504]),
                    dcc.Store(id="y-variable-j1", data=[105]),
                    dcc.Store(id="y-variable-j2", data=[55]),
                    dcc.Store(id="cpt-j1", data=0),
                    dcc.Store(id="cpt-j2", data=0),
                    dcc.Store(id="somme-depense-j1", data=0),
                    dcc.Store(id="somme-depense-j2", data=0)
                ],
                value="tab1"
            ),
            dmc.TabsPanel(
                html.Div([
                    html.Button("Bouton Test", id="bouton-test"),
                ]),
                value="tab2"
            )
        ],
        value="tab1", color="violet"
    )
])

#tirage dés - déplacement sur le plateau - JOUEUR 1
@app.callback([Output(component_id="plateau-figure", component_property="figure"), Output("btn-des-j1", "n_clicks"),
               Output("x-variable-j1", "data"), Output("y-variable-j1", "data"),
               Output("cpt-j1", "data"), Output("result-des-j1", "children"),
               Output("btn-des-j2", "n_clicks"), Output("x-variable-j2", "data"),
               Output("y-variable-j2", "data"), Output("cpt-j2", "data"), Output("result-des-j2", "children")],
              [Input("btn-des-j1", "n_clicks"), Input("x-variable-j1", "data"), 
               Input("y-variable-j1", "data"), Input("cpt-j1", "data"),
               Input("btn-des-j2", "n_clicks"), Input("x-variable-j2", "data"), 
               Input("y-variable-j2", "data"), Input("cpt-j2", "data")],
               prevent_initial_call=True)
def tirage_des(btn_des_j1, x_j1, y_j1, c_j1, btn_des_j2, x_j2, y_j2, c_j2):
    global move_current_j1
    global move_current_j2
    de_1_j1, de_2_j1, de_1_j2, de_2_j2 = 0, 0, 0, 0
    
    if btn_des_j1:
        #tirage des dés
        de_1_j1 = random.randint(1, 6)
        de_2_j1 = random.randint(1, 6)
        random_number_j1 = de_1_j1 + de_2_j1
        
        for j in range(random_number_j1):
            x_j1[0] += move_current_j1[0] * 115
            y_j1[0] += move_current_j1[1] * 115
            c_j1 +=1
            if c_j1 % 12 == 0:
                move_current_j1 = next(move_j1)

    if btn_des_j2:
        #tirage des dés
        de_1_j2 = random.randint(1, 6)
        de_2_j2 = random.randint(1, 6)
        random_number_j2 = de_1_j2 + de_2_j2
        
        for j in range(random_number_j2):
            x_j2[0] += move_current_j2[0] * 120
            y_j2[0] += move_current_j2[1] * 120
            c_j2 +=1
            if c_j2 % 12 == 0:
                move_current_j2 = next(move_j2)
    
    fig = affichage_plateau(chemin_images, x_j1, y_j1, x_j2, y_j2, "blue", "red")

    return fig, 0, x_j1, y_j1, c_j1, f"🎲 {de_1_j1} 🎲 {de_2_j1}", 0, x_j2, y_j2, c_j2, f"🎲 {de_1_j2} 🎲 {de_2_j2}"

###########################################################################
##################################Joueur1##################################
###########################################################################
#affichage des cartes en fonction de la case - JOUEUR 1
@app.callback(Output(component_id="carte-figure-j1", component_property="figure"),
          Input("cpt-j1", "data"),
        prevent_initial_call=True)
def case_carte(c):
    global move_current_j1
    if np.array_equal(move_current_j1, np.array([-1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 0)
    if np.array_equal(move_current_j1, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 12 + 12)
    if np.array_equal(move_current_j1, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 24)
    if np.array_equal(move_current_j1, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 12 + 36)
    return fig

#dépenses places - JOUEUR 1
@app.callback([Output("depense-j1", "children"), Output("somme-depense-j1", "data"),
               Output("btn-catA-j1", "n_clicks"), Output("btn-catB-j1", "n_clicks"),
               Output("btn-catC-j1", "n_clicks"), Output("btn-catD-j1", "n_clicks")],
              [Input("btn-catA-j1", "n_clicks"), Input("btn-catB-j1", "n_clicks"),
               Input("btn-catC-j1", "n_clicks"), Input("btn-catD-j1", "n_clicks"),
               Input("somme-depense-j1", "data"), Input("cpt-j1", "data")],
        prevent_initial_call=True)
def calcul_depense(btnA, btnB, btnC, btnD, depense, c):
    #obtenir valeur de la case
    global move_current_j1
    if c != 0:
        if np.array_equal(move_current_j2, np.array([-1,0])):
            c = c % 12 + 0
        if np.array_equal(move_current_j2, np.array([0,1])):
            c = c % 12 + 12
        if np.array_equal(move_current_j2, np.array([1,0])):
            c = c % 12 + 24
        if np.array_equal(move_current_j2, np.array([0,-1])):
            c = c % 12 + 36

        #les prix correspondants
        prix_sport = df_prix[["catA", "catB", "catC", "catD"]][df_prix["id_case"] == c].values[0].tolist()
        
        #caluler dépense
        if btnA: depense = depense + prix_sport[0]
        if btnB: depense = depense + prix_sport[1]
        if btnC: depense = depense + prix_sport[2]
        if btnD: depense = depense + prix_sport[3]

        return f"{depense} €", depense, 0, 0, 0, 0
    else:
        return f"0 €", 0, 0, 0, 0, 0

###########################################################################
##################################Joueur2##################################
###########################################################################
#affichage des cartes en fonction de la case - JOUEUR 2
@app.callback(Output(component_id="carte-figure-j2", component_property="figure"),
          Input("cpt-j2", "data"),
        prevent_initial_call=True)
def case_carte(c):
    global move_current_j2
    if np.array_equal(move_current_j2, np.array([-1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 0)
    if np.array_equal(move_current_j2, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 12 + 12)
    if np.array_equal(move_current_j2, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 24)
    if np.array_equal(move_current_j2, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 12 + 36)
    return fig

#dépenses places - JOUEUR 2
@app.callback([Output("depense-j2", "children"), Output("somme-depense-j2", "data"),
               Output("btn-catA-j2", "n_clicks"), Output("btn-catB-j2", "n_clicks"),
               Output("btn-catC-j2", "n_clicks"), Output("btn-catD-j2", "n_clicks")],
              [Input("btn-catA-j2", "n_clicks"), Input("btn-catB-j2", "n_clicks"),
               Input("btn-catC-j2", "n_clicks"), Input("btn-catD-j2", "n_clicks"),
               Input("somme-depense-j2", "data"), Input("cpt-j2", "data")],
        prevent_initial_call=True)
def calcul_depense(btnA, btnB, btnC, btnD, depense, c):
    #obtenir valeur de la case
    global move_current_j2
    if c != 0:
        if np.array_equal(move_current_j2, np.array([-1,0])):
            c = c % 12 + 0
        if np.array_equal(move_current_j2, np.array([0,1])):
            c = c % 12 + 12
        if np.array_equal(move_current_j2, np.array([1,0])):
            c = c % 12 + 24
        if np.array_equal(move_current_j2, np.array([0,-1])):
            c = c % 12 + 36

        #les prix correspondants
        prix_sport = df_prix[["catA", "catB", "catC", "catD"]][df_prix["id_case"] == c].values[0].tolist()
        
        #caluler dépense
        if btnA: depense = depense + prix_sport[0]
        if btnB: depense = depense + prix_sport[1]
        if btnC: depense = depense + prix_sport[2]
        if btnD: depense = depense + prix_sport[3]

        return f"{depense} €", depense, 0, 0, 0, 0
    else:
        return f"0 €", 0, 0, 0, 0, 0

#afficher les règles
@app.callback(
    Output("modal-regles", "opened"),
    Input("modal-regles-btn", "n_clicks"),
    State("modal-regles", "opened"),
    prevent_initial_call=True,
)
def ouvrir_regles(n_clicks, opened):
    return not opened


if __name__ == '__main__':
    app.run(debug=True)


