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
df_prix_para = pd.read_csv("./data/prix_discipline_para.csv", sep=";", encoding='latin1')

#ouvrir image - olympique
im = cv2.imread(f"{chemin_images}/plateau.png")
im = im[:,:,[2,1,0]]
im = cv2.flip(im, 0)

#ouvrir image - paralympique
im_para = cv2.imread(f"{chemin_images}/plateau_para.png")
im_para = im_para[:,:,[2,1,0]]
im_para = cv2.flip(im_para, 0)

txt1 = "J.O.P.oly se compose de deux plateaux de jeu, chacun avec de 2 dés, de 2 pions (rouge et bleu ; un par joueur). Sur le plateau Jeux Olympiques (JO), 40 cartes épreuves, 5 cartes événements et 2 cartes CHANCE. Sur le plateau Jeux Paralympiques (JP), 21 cartes épreuves, 5 cartes événements et 13 cartes CHANCE (ou PAS DE CHANCE). Préparez-vous, échauffez-vous et placez-vous sur la ligne, ou plutôt la case départ. Pensez à adapter le zoom en fonction de votre écran. *coup de pistolet* Joueur 1 prenez place et cliquez sur la flèche correspondante. Lancez les dés. Continuez de vous échauffer pendant que les dés font leur travail. Si vous souhaitez acheter des places, cliquez autant de fois que vous le voulez sur la catégorie correspondante. Puis, la main passe au Joueur 2 qui est dans les starting-blocks. A vos marques, prêt ? A votre tour. "

left = np.array([-1,0])
up = np.array([0,1])
right = np.array([1,0])
down = np.array([0,-1])

move_j1, move_j2 = cycle([left, up, right, down]), cycle([left, up, right, down])
move_j1_para, move_j2_para = cycle([left, up, right, down]), cycle([left, up, right, down])

move_current_j1, move_current_j2 = next(move_j1), next(move_j2)
move_current_j1_para, move_current_j2_para = next(move_j1_para), next(move_j2_para)

#ouvrir et afficher le plateau
def affichage_plateau(image, coord_x1=[5000], coord_y1=[350], coord_x2=[5000], coord_y2=[200], col1="blue", col2="red", para=False):
    #plateau jeux olympiques

    fig = px.imshow(image, width=im.shape[0]//7, height=im.shape[1]//7)

    #plateau jeux paralympiques
    if para:
        fig = px.imshow(image, width=im.shape[0]//6, height=im.shape[1]//6)

    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, image.shape[1]])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, domain=[0,1], range=[0, image.shape[0]])
    fig.add_trace(go.Scatter(x=coord_x1, y=coord_y1, mode="markers", marker=dict(color=col1, size=20), showlegend=False))
    fig.add_trace(go.Scatter(x=coord_x2, y=coord_y2, mode="markers", marker=dict(color=col2, size=20), showlegend=False))    
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig

#ouvrir et afficher une carte
def affichage_carte(path, id_carte, o_p):
    im = cv2.imread(f"{path}/cartes/{o_p}/carte_{id_carte}.png")
    im = im[:,:,[2,1,0]]
    im = cv2.flip(im, 0)
    fig = px.imshow(im)
    fig.update_layout(template = None, margin=dict(l=0, r=0, t=0, b=0), autosize=True)
    fig.update_xaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, im.shape[1]])
    fig.update_yaxes(visible=False, showticklabels=False, showgrid=False, showline=False, range=[0, im.shape[0]])
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    return fig
    

app = Dash(__name__)
server = app.server

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
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0, "olympiques"), id="carte-figure-j1")),
                                dmc.ButtonGroup(
                                    [
                                        dmc.Button("A", id="btn-catA-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("B", id="btn-catB-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("C", id="btn-catC-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("D", id="btn-catD-j1", size="xl", color="blue", variant="outline", n_clicks=0),
                                    ]
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                dmc.Modal(title="Règles du Jeu", id="modal-regles", size="55%", zIndex=10000, children=[dmc.Text(txt1)]),
                                html.Center(dmc.Button("Règles du Jeu", id="modal-regles-btn", size="lg", color="violet"))
                            ],
                        ),
                    ], style={'width': '20%', 'float': 'left'}),
                    html.Div(children=[
                        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                        html.Center(dcc.Graph(figure=affichage_plateau(im), id="plateau-figure"))
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
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0, "olympiques"), id="carte-figure-j2")),
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
                    dcc.Store(id="x-variable-j1", data=[5000]),
                    dcc.Store(id="x-variable-j2", data=[5000]),
                    dcc.Store(id="y-variable-j1", data=[350]),
                    dcc.Store(id="y-variable-j2", data=[200]),
                    dcc.Store(id="cpt-j1", data=0),
                    dcc.Store(id="cpt-j2", data=0),
                    dcc.Store(id="somme-depense-j1", data=0),
                    dcc.Store(id="somme-depense-j2", data=0)
                ],
                value="tab1"
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
                                                dmc.Button("Dés", size="lg", id="btn-des-j1-para", color="blue", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
                                                dmc.Text("", id="result-des-j1-para", size="xl", color="blue", weight=700),
                                                dmc.Text("", id="depense-j1-para", size="xl", color="blue", weight=700)
                                            ],
                                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}
                                        ),
                                    ],
                                    value="joueur1",
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0, "olympiques"), id="carte-figure-j1-para")),
                                dmc.ButtonGroup(
                                    [
                                        dmc.Button("A", id="btn-catA-j1-para", size="lg", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("A PFR", id="btn-catB-j1-para", size="lg", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("B", id="btn-catC-j1-para", size="lg", color="blue", variant="outline", n_clicks=0),
                                        dmc.Button("B PFR", id="btn-catD-j1-para", size="lg", color="blue", variant="outline", n_clicks=0),
                                    ]
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                dmc.Modal(title="Règles du Jeu", id="modal-regles-para", size="55%", zIndex=10000, children=[dmc.Text(txt1)]),
                                html.Center(dmc.Button("Règles du Jeu", id="modal-regles-btn-para", size="lg", color="violet"))
                            ],
                        ),
                    ], style={'width': '20%', 'float': 'left'}),
                    html.Div(children=[
                        html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                        html.Center(dcc.Graph(figure=affichage_plateau(im_para, coord_x1=[4270], coord_y1=[405], coord_x2=[4270], coord_y2=[255], para=True), id="plateau-figure-para"))
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
                                                dmc.Button("Dés", size="lg", id="btn-des-j2-para", color="red", n_clicks=0, leftIcon=DashIconify(icon="ion:dice", color="white", width=30)),
                                                dmc.Text("", id="result-des-j2-para", size="xl", color="red", weight=700),
                                                dmc.Text("", id="depense-j2-para", size="xl", color="red", weight=700)
                                            ],
                                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}
                                        ),
                                    ],
                                    value="joueur2",
                                ),
                                html.Hr(style={'border': '1px solid rgba(0, 0, 0, 0)'}),
                                html.Center(dcc.Graph(figure=affichage_carte(chemin_images, 0, "olympiques"), id="carte-figure-j2-para")),
                                dmc.ButtonGroup(
                                    [
                                        dmc.Button("A", id="btn-catA-j2-para", size="lg", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("A PFR", id="btn-catB-j2-para", size="lg", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("B", id="btn-catC-j2-para", size="lg", color="red", variant="outline", n_clicks=0),
                                        dmc.Button("B PFR", id="btn-catD-j2-para", size="lg", color="red", variant="outline", n_clicks=0),
                                    ]
                                )
                            ],
                        ),
                    ], style={'width': '20%', 'float': 'left'}),
                    dcc.Store(id="x-variable-j1-para", data=[4255]),
                    dcc.Store(id="x-variable-j2-para", data=[4255]),
                    dcc.Store(id="y-variable-j1-para", data=[350]),
                    dcc.Store(id="y-variable-j2-para", data=[200]),
                    dcc.Store(id="cpt-j1-para", data=0),
                    dcc.Store(id="cpt-j2-para", data=0),
                    dcc.Store(id="somme-depense-j1-para", data=0),
                    dcc.Store(id="somme-depense-j2-para", data=0)
                ],
                value="tab2"
            ),
        ],
        value="tab1", color="violet"
    )
])
##############################################OLYMPIQUES##############################################
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
            x_j1[0] += move_current_j1[0] * 400
            y_j1[0] += move_current_j1[1] * 400
            c_j1 +=1
            if c_j1 % 12 == 0:
                move_current_j1 = next(move_j1)

    if btn_des_j2:
        #tirage des dés
        de_1_j2 = random.randint(1, 6)
        de_2_j2 = random.randint(1, 6)
        random_number_j2 = de_1_j2 + de_2_j2
        
        for j in range(random_number_j2):
            x_j2[0] += move_current_j2[0] * 400
            y_j2[0] += move_current_j2[1] * 400
            c_j2 +=1
            if c_j2 % 12 == 0:
                move_current_j2 = next(move_j2)
    
    fig = affichage_plateau(im, x_j1, y_j1, x_j2, y_j2, "blue", "red")

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
        fig = affichage_carte(chemin_images, c % 12 + 0, "olympiques")
    if np.array_equal(move_current_j1, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 12 + 12, "olympiques")
    if np.array_equal(move_current_j1, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 24, "olympiques")
    if np.array_equal(move_current_j1, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 12 + 36, "olympiques")
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
        if np.array_equal(move_current_j1, np.array([-1,0])):
            c = c % 12 + 0
        if np.array_equal(move_current_j1, np.array([0,1])):
            c = c % 12 + 12
        if np.array_equal(move_current_j1, np.array([1,0])):
            c = c % 12 + 24
        if np.array_equal(move_current_j1, np.array([0,-1])):
            c = c % 12 + 36

        #les prix correspondants
        if c in df_prix["id_case"].values:
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
        fig = affichage_carte(chemin_images, c % 12 + 0, "olympiques")
    if np.array_equal(move_current_j2, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 12 + 12, "olympiques")
    if np.array_equal(move_current_j2, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 12 + 24, "olympiques")
    if np.array_equal(move_current_j2, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 12 + 36, "olympiques")
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
        if c in df_prix["id_case"].values:
            prix_sport = df_prix[["catA", "catB", "catC", "catD"]][df_prix["id_case"] == c].values[0].tolist()
        
        #caluler dépense
        if btnA: depense = depense + prix_sport[0]
        if btnB: depense = depense + prix_sport[1]
        if btnC: depense = depense + prix_sport[2]
        if btnD: depense = depense + prix_sport[3]

        return f"{depense} €", depense, 0, 0, 0, 0
    else:
        return f"0 €", 0, 0, 0, 0, 0
    
##############################################PARALYMPIQUES##############################################
#tirage dés - déplacement sur le plateau - JOUEUR 1
@app.callback([Output(component_id="plateau-figure-para", component_property="figure"), Output("btn-des-j1-para", "n_clicks"),
               Output("x-variable-j1-para", "data"), Output("y-variable-j1-para", "data"),
               Output("cpt-j1-para", "data"), Output("result-des-j1-para", "children"),
               Output("btn-des-j2-para", "n_clicks"), Output("x-variable-j2-para", "data"),
               Output("y-variable-j2-para", "data"), Output("cpt-j2-para", "data"), Output("result-des-j2-para", "children")],
              [Input("btn-des-j1-para", "n_clicks"), Input("x-variable-j1-para", "data"), 
               Input("y-variable-j1-para", "data"), Input("cpt-j1-para", "data"),
               Input("btn-des-j2-para", "n_clicks"), Input("x-variable-j2-para", "data"), 
               Input("y-variable-j2-para", "data"), Input("cpt-j2-para", "data")],
               prevent_initial_call=True)
def tirage_des(btn_des_j1, x_j1, y_j1, c_j1, btn_des_j2, x_j2, y_j2, c_j2):
    global move_current_j1_para
    global move_current_j2_para
    de_1_j1_para, de_2_j1_para, de_1_j2_para, de_2_j2_para = 0, 0, 0, 0
    
    if btn_des_j1:
        #tirage des dés
        de_1_j1_para = random.randint(1, 6)
        de_2_j1_para = random.randint(1, 6)
        random_number_j1_para = de_1_j1_para + de_2_j1_para
        print(de_1_j1_para, de_2_j1_para,random_number_j1_para)
        
        for j in range(random_number_j1_para):
            x_j1[0] += move_current_j1_para[0] * 400 * ((4555*7)/(5308*6))
            y_j1[0] += move_current_j1_para[1] * 400 * ((4555*7)/(5308*6))
            c_j1 +=1
            if c_j1 % 10 == 0:
                move_current_j1_para = next(move_j1_para)

    if btn_des_j2:
        #tirage des dés
        de_1_j2_para = random.randint(1, 6)
        de_2_j2_para = random.randint(1, 6)
        random_number_j2_para = de_1_j2_para + de_2_j2_para
        print(de_1_j2_para, de_2_j2_para,random_number_j2_para)
        for j in range(random_number_j2_para):
            x_j2[0] += move_current_j2_para[0] * 400 * ((4555*7)/(5308*6))
            y_j2[0] += move_current_j2_para[1] * 400 * ((4555*7)/(5308*6))
            c_j2 +=1
            if c_j2 % 10 == 0:
                move_current_j2_para = next(move_j2_para)
    
    fig_para = affichage_plateau(im_para, x_j1, y_j1, x_j2, y_j2, "blue", "red", para=True)

    return fig_para, 0, x_j1, y_j1, c_j1, f"🎲 {de_1_j1_para} 🎲 {de_2_j1_para}", 0, x_j2, y_j2, c_j2, f"🎲 {de_1_j2_para} 🎲 {de_2_j2_para}"

###########################################################################
##################################Joueur1##################################
###########################################################################
#affichage des cartes en fonction de la case - JOUEUR 1
@app.callback(Output(component_id="carte-figure-j1-para", component_property="figure"),
          Input("cpt-j1-para", "data"),
        prevent_initial_call=True)
def case_carte(c):
    global move_current_j1_para
    if np.array_equal(move_current_j1_para, np.array([-1,0])):
        fig = affichage_carte(chemin_images, c % 10 + 0, "paralympiques")
    if np.array_equal(move_current_j1_para, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 10 + 10, "paralympiques")
    if np.array_equal(move_current_j1_para, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 10 + 20, "paralympiques")
    if np.array_equal(move_current_j1_para, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 10 + 30, "paralympiques")
    return fig

#dépenses places - JOUEUR 1
@app.callback([Output("depense-j1-para", "children"), Output("somme-depense-j1-para", "data"),
               Output("btn-catA-j1-para", "n_clicks"), Output("btn-catB-j1-para", "n_clicks"),
               Output("btn-catC-j1-para", "n_clicks"), Output("btn-catD-j1-para", "n_clicks")],
              [Input("btn-catA-j1-para", "n_clicks"), Input("btn-catB-j1-para", "n_clicks"),
               Input("btn-catC-j1-para", "n_clicks"), Input("btn-catD-j1-para", "n_clicks"),
               Input("somme-depense-j1-para", "data"), Input("cpt-j1-para", "data")],
        prevent_initial_call=True)
def calcul_depense(btnA, btnB, btnC, btnD, depense, c):
    #obtenir valeur de la case
    global move_current_j1_para
    if c != 0:
        if np.array_equal(move_current_j1_para, np.array([-1,0])):
            c = c % 10 + 0
        if np.array_equal(move_current_j1_para, np.array([0,1])):
            c = c % 10 + 10
        if np.array_equal(move_current_j1_para, np.array([1,0])):
            c = c % 10 + 20
        if np.array_equal(move_current_j1_para, np.array([0,-1])):
            c = c % 10 + 30

        #les prix correspondants
        if c in df_prix_para["id_case"].values:
            prix_sport = df_prix_para[["catA", "catA_PFR", "catB", "catB_PFR"]][df_prix_para["id_case"] == c].values[0].tolist()
        
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
@app.callback(Output(component_id="carte-figure-j2-para", component_property="figure"),
          Input("cpt-j2-para", "data"),
        prevent_initial_call=True)
def case_carte(c):
    global move_current_j2_para
    if np.array_equal(move_current_j2_para, np.array([-1,0])):
        fig = affichage_carte(chemin_images, c % 10 + 0, "paralympiques")
    if np.array_equal(move_current_j2_para, np.array([0,1])):
        fig = affichage_carte(chemin_images, c % 10 + 10, "paralympiques")
    if np.array_equal(move_current_j2_para, np.array([1,0])):
        fig = affichage_carte(chemin_images, c % 10 + 20, "paralympiques")
    if np.array_equal(move_current_j2_para, np.array([0,-1])):
        fig = affichage_carte(chemin_images, c % 10 + 30, "paralympiques")
    return fig

#dépenses places - JOUEUR 2
@app.callback([Output("depense-j2-para", "children"), Output("somme-depense-j2-para", "data"),
               Output("btn-catA-j2-para", "n_clicks"), Output("btn-catB-j2-para", "n_clicks"),
               Output("btn-catC-j2-para", "n_clicks"), Output("btn-catD-j2-para", "n_clicks")],
              [Input("btn-catA-j2-para", "n_clicks"), Input("btn-catB-j2-para", "n_clicks"),
               Input("btn-catC-j2-para", "n_clicks"), Input("btn-catD-j2-para", "n_clicks"),
               Input("somme-depense-j2-para", "data"), Input("cpt-j2-para", "data")],
        prevent_initial_call=True)
def calcul_depense(btnA, btnB, btnC, btnD, depense, c):
    #obtenir valeur de la case
    global move_current_j2_para
    if c != 0:
        if np.array_equal(move_current_j2_para, np.array([-1,0])):
            c = c % 10 + 0
        if np.array_equal(move_current_j2_para, np.array([0,1])):
            c = c % 10 + 10
        if np.array_equal(move_current_j2_para, np.array([1,0])):
            c = c % 10 + 20
        if np.array_equal(move_current_j2_para, np.array([0,-1])):
            c = c % 10 + 30

        #les prix correspondants
        if c in df_prix_para["id_case"].values:
            prix_sport = df_prix_para[["catA", "catA_PFR", "catB", "catB_PFR"]][df_prix_para["id_case"] == c].values[0].tolist()
        
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


