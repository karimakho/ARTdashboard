import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import folium
import geopandas as gpd
from branca.colormap import linear
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
import json
from collections import Counter
from itertools import combinations
import networkx as nx
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os


# Base directory where the script is located
base_dir = os.path.dirname(__file__)

# Relative path to the CSV file in the assets directory
file_path = os.path.join(base_dir, '..', 'assets', 'data_predictions.csv')

# Load data
data = pd.read_csv(file_path)


# Convert the 'Date' column to datetime and extract the year
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
if 'Date' in data.columns:
   data['Year'] = data['Date'].dt.year

base_dir = os.path.dirname(__file__)
# New section for psychological aspects of infertility
# Load new data
preprocessed_psychopma_path = os.path.join(base_dir, '..', 'assets', 'preprocessed_psychopma.csv')
df_psycho = pd.read_csv(preprocessed_psychopma_path)


# Ensure the Date column is parsed as datetime
df_psycho['Date'] = pd.to_datetime(df_psycho['Date'], errors='coerce')


# Remove timezone information for period conversion
df_psycho['Date'] = df_psycho['Date'].dt.tz_localize(None)


# Create dropdown options for years and sort them
years_psycho = sorted(df_psycho['Date'].dt.year.unique())
year_options = [{'label': str(year), 'value': year} for year in years_psycho]


# Bootstrap themes URL
url_theme = dbc.themes.FLATLY


# CSS URL for Bootstrap components
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[url_theme, dbc_css], suppress_callback_exceptions=True)
server = app.server


# Navbar
navbar = dbc.Navbar(
   dbc.Container(
       [
           dbc.Row(
               dbc.Col(
                   html.A(
                       html.Div([
                           "The ART journey: Forum post analysis",
                           html.Span(className="fas fa-baby",
                                     style={'font-size': '30px', 'margin-left': '15px', 'color': 'white'}),
                       ], style={'text-align': 'center', 'width': '100%', 'font-size': '32px', 'color': 'white'}),
                       href='/',
                       style={'text-decoration': 'none'}
                   ),
                   width={"size": 12, "offset": 0},  # Center the title
                   style={'text-align': 'center'}
               ),
           ),
       ],
       fluid=True,
   ),
   color="#D83367",  # Darker pink color
   dark=True,
   className="mb-4",
   style={'padding': '15px', 'border-radius': '0px'}  # No rounded borders
)

logo_doctissimo_src = '/assets/Logo_de_Doctissimo.jpg'
logo_fivfr_src = '/assets/logo-fivfr.png'


# Presentation section
# Presentation section
def presentation_page():
   return dbc.Container(
       [
           navbar,
           dbc.Container(
               [
                   html.Hr(),
                   html.H5('Presentation', style={'color': 'black', 'text-align': 'center'}),
                   html.P('This application presents our analysis of discussions from two forums (Fiv.fr and Doctissimo.fr) on the Assisted Reproductive Technology (ART) journey in France.',
                          style={'color': 'black', 'text-align': 'center'}),
                   html.Div(
                       [
                           html.Img(src=logo_doctissimo_src, height='200px', style={'margin-right': '20px'}),
                           html.Img(src=logo_fivfr_src, height='100px')
                       ],
                       style={'text-align': 'center'}
                   ),
                   dbc.Button("Access the analysis results", id="app-button", color="primary", className="mt-3",
                              href='/application',
                              style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                   html.Div(
                       [
                           html.Img(src='/assets/images.png', height='100px', style={'margin-top': '20px'})  # Ajout de la marge ici
                       ],
                       style={'text-align': 'center','margin-bottom': '20px'})
                   ,
                   html.H5('Authors', style={'color': 'black', 'text-align': 'center'}),
                   html.P('BAZINE Nohaila, KANOUN Melinda, KHOUITI Karima, MENZOU Katia and SABRI Ouahiba',
                          style={'color': 'black', 'text-align': 'center'})
               ],
               style={'padding': '20px', 'background-color': 'white', 'border-radius': '10px'}
           )
       ],
       fluid=True,
       className="dbc",
       style={'background-color': 'white'}
   )


# Application section
def application_page():
   return dbc.Container(
       [
           navbar,
           dbc.Tabs(
               [
                   dbc.Tab(label="FIV", tab_id="fiv", label_style={'color': 'black'}),
                   dbc.Tab(label="Doctissimo", tab_id="doctissimo", label_style={'color': 'black'}),
               ],
               id="main-tabs",
               active_tab="fiv",
           ),
           html.Div(id='main-content')
       ],
       fluid=True,
       style={'padding': '20px', 'background-color': 'white', 'border-radius': '10px'}
   )


# Read GeoJSON and data files
data_path = os.path.join(base_dir, '..', 'assets', 'data.csv')
base_dir = os.path.dirname(__file__)
geojson_path = os.path.join(base_dir, '..', 'assets', 'regions-version-simplifiee.geojson')

# Load and use the GeoJSON data as needed
with open(geojson_path) as f:
    geojson_data = f.read()

data_geo = pd.read_csv(data_path, delimiter=';')
gdf = gpd.read_file(geojson_path)


# Rename the column to match for merging
gdf = gdf.rename(columns={'nom': 'Region'})


# Merge GeoDataFrame with DataFrame
merged_gdf = gdf.merge(data_geo, on='Region', how='left')


# Create a Folium map
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)


# Create a colormap for PMA_count
colormap = linear.PuRd_09.scale(merged_gdf['PMA_count'].min(), merged_gdf['PMA_count'].max())


# Specific color for NaN values
na_color = '#d3d3d3'


# Style function to color regions
def style_function(feature):
   pma_count = feature['properties']['PMA_count']
   if pd.isna(pma_count):
       return {
           'fillColor': na_color,
           'color': 'black',
           'weight': 2,
           'fillOpacity': 0.7,
       }
   else:
       return {
           'fillColor': colormap(pma_count),
           'color': 'black',
           'weight': 2,
           'fillOpacity': 0.7,
       }


# Add regions to the map
folium.GeoJson(
   data=merged_gdf,
   style_function=style_function,
   tooltip=folium.GeoJsonTooltip(fields=['Region', 'PMA_count', 'FIV_count'],
                                 aliases=['Region', 'Number of ART occurrences', 'Number of IVF occurrences'],
                                 localize=True)
).add_to(m)


# Add the colormap to the map
colormap.caption = 'Number of ART occurrences'
colormap.add_to(m)


# Save the map as HTML
m.save('regions_pma_fiv_map.html')


# Convert the Folium map to HTML for integration in Dash
folium_map_html = m.get_root().render()


# Load the Doctissimo dataset
dataset = pd.read_csv("C:/Users/33770/Documents/pma_doctissimo_posts.csv")


# Remove rows containing empty strings or NA in critical columns
dataset = dataset.dropna(subset=['title', 'author', 'replies', 'views', 'last_post_date', 'comments'])
dataset = dataset[(dataset['title'] != '') & (dataset['author'] != '') & (dataset['comments'] != '[]')]


# Clean and convert data
dataset['replies'] = dataset['replies'].str.replace('\u00a0', '').astype(int)
dataset['views'] = dataset['views'].str.replace('\u00a0', '').astype(int)
dataset['last_post_date'] = pd.to_datetime(dataset['last_post_date'], errors='coerce')
dataset = dataset.dropna(subset=['last_post_date'])


# Remove rows with dates in 2013
dataset = dataset[dataset['last_post_date'].dt.year != 2013]


# Extract the month and year from the post date
dataset['year'] = dataset['last_post_date'].dt.year.astype(str)  # Convert to string for display in Plotly
dataset['month'] = dataset['last_post_date'].dt.strftime('%b')


# Get the list of years in chronological order
years = dataset['year'].unique()
years.sort()


# Add a 'count' column for the number of posts (each row is a post)
dataset['count'] = 1


# Create a bar chart of posts per month for each year
fig = px.histogram(dataset, x='month', color='year', barmode='group',
                  category_orders={
                      'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                      'year': list(years)},
                  labels={'month': 'Month', 'count': 'Number of Posts', 'year': 'Year'},
                  title='Distribution of posts by month for each year',
                  histfunc='count')


# Update the Y-axis label and axis properties and background
fig.update_layout(
   yaxis_title="Number of posts",
   plot_bgcolor='white',  # White background
   paper_bgcolor='white',  # White background
   font=dict(color='black'),  # Text color black
   xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False),
   yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False)
)


# Comprehensive stop words list
french_stopwords = set(STOPWORDS).union({
   'avoir', 'être', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la', 'les', 'un', 'une', 'des',
   'du', 'de', 'd', 'en', 'et', 'à', 'pour', 'qui', 'que', 'dans', 'ne', 'pas', 'avec', 'sur', 'au', 'aux', 'plus',
   'par', 'se', 'ce', 'ou', 'son', 'sa', 'ses', 'donc', 'car', 'mais', 'ni', 'si', 'comme', 'comment', 'là', 'où',
   'quand', 'tant', 'bien', 'aussi', 'ça', 'tout', 'tous', 'très', 'sans', 'vers', 'ces', 'ça', 'cette', 'si', 'quoi',
   'donc', 'avant', 'après', 'chaque', 'afin', 'ainsi', 'cela', 'cet', 'cette', 'lors', 'afin', 'entre', 'contre',
   'sous', 'ainsi', 'toutes', 'depuis', 'vos', 'ans', 'dautre', 'autre', 'autres', 'est', 'question', 'pma', 'fiv',
   'besoin', 'ivf', 'opk', 'help', 'suis', 'transfert', 'amp', 'fivd', 'svp', 'suit', 'moi'
}).union(sklearn_stopwords)


# Function to generate word cloud
def generate_wordcloud(text):
   # Convert text to lowercase
   text = text.lower()
   # Remove punctuation
   text = re.sub(r'[^\w\s]', '', text)


   # Define a function to generate darker shades of pink
   def dark_pink_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
       return "hsl(330, 100%, {}%)".format(30 + font_size % 30)


   wordcloud = WordCloud(
       width=800,
       height=400,
       background_color='white',
       stopwords=french_stopwords,
       color_func=dark_pink_color_func,
       collocations=False,
       margin=2,
       max_font_size=100,  # Adjust max font size
       relative_scaling=0.5,  # Adjust relative scaling
       min_font_size=10  # Ensure a minimum font size
   ).generate(text)


   img = BytesIO()
   wordcloud.to_image().save(img, format='PNG')
   img.seek(0)
   return base64.b64encode(img.getvalue()).decode()


# Generate the word cloud with all titles
all_titles_text = " ".join(dataset['title'].tolist())
wordcloud_img = generate_wordcloud(all_titles_text)
wordcloud_img_src = 'data:image/png;base64,{}'.format(wordcloud_img)


# Load data for 3D visualization
stopwords_french = set([
   "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux",
   "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc",
   "dos", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font",
   "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant",
   "mais", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où",
   "par", "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles",
   "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur",
   "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre",
   "vous", "vu", "ça", "étaient", "état", "étions", "été", "être"
])


# Keywords dictionary
keywords = {
   'Causes': [
       'infertilité', 'stérilité', 'trouble hormonal', 'SOPK', 'endométriose', 'âge avancé', 'obstruction tubaire',
       'faible qualité du sperme', 'azoospermie', 'anovulation', 'facteur masculin', 'réserve ovarienne diminuée',
   ],
   'Sentiments': [
       'espoir', 'frustration', 'stress', 'anxiété', 'déception', 'joie', 'soulagement', 'désespoir', 'détermination',
       'impatience', 'culpabilité', 'tristesse',
   ],
   'Conséquences': [
       'réussite', 'échec', 'grossesse multiple', 'fausse couche', 'naissance prématurée', 'complications médicales',
       'coût financier', 'traitement prolongé', 'dépendance émotionnelle', 'soutien familial',
       'changements de mode de vie',
       'impact sur le couple', 'saignement',
   ],
   'Grossesse': [
       'embryon', 'implantation', 'stimulation ovarienne', 'ponction folliculaire', 'transfert d\'embryon',
       'suivi médical', 'hormones', 'échographie', 'tests de grossesse', 'protocole', 'nidation',
       'congélation d\'embryons', 'ovaire', 'embryon',
       'ovulation', 'enceinte', 'pma', 'fiv', 'icsi', 'iac', 'ovitrelle', 'clomid', 'follicule', 'duphaston',
       'fivette', 'injection'
   ]
}


# Keywords dictionary
psych_keywords = {
   'Causes': [
       'infertilité', 'stérilité', 'trouble hormonal', 'SOPK', 'endométriose', 'âge avancé', 'obstruction tubaire',
       'faible qualité du sperme', 'azoospermie', 'anovulation', 'facteur masculin', 'réserve ovarienne diminuée',
   ],
   'Sentiments': [
       'espoir', 'frustration', 'stress', 'anxiété', 'déception', 'joie', 'soulagement', 'désespoir', 'détermination',
       'impatience', 'culpabilité', 'tristesse',
   ],
   'Conséquences': [
       'réussite', 'échec', 'grossesse multiple', 'fausse couche', 'naissance prématurée', 'complications médicales',
       'coût financier', 'traitement prolongé', 'dépendance émotionnelle', 'soutien familial',
       'changements de mode de vie', 'impact sur le couple', 'saignement',
   ],
   'Psychological Aspects': [
       'dépression', 'anxiété', 'stress', 'tristesse', 'peur', 'culpabilité', 'désespoir', 'espoir', 'frustration',
       'psychologique', 'psychologie', 'thérapie', 'émotion', 'fatigue', 'épuisement', 'angoisse',
       'nervosité', 'déprime', 'mental', 'bien-être', 'détresse', 'désarroi', 'solitude', 'isolement', 'appétit'
   ]
}


# Flatten the keywords list
psych_keywords_set = set([item for sublist in psych_keywords.values() for item in sublist])
# Flatten the keywords list
keywords_set = set([item for sublist in keywords.values() for item in sublist])


# Function to extract and clean comments
def extract_and_clean_comments(comments_json):
   try:
       comments = json.loads(comments_json.replace("'", '"'))
       text = ' '.join(comment['content'] for comment in comments)
       text = text.lower()
       text = re.sub(r'\d+', '', text)  # Remove digits
       text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
       tokens = text.split()
       tokens = [word for word in tokens if word not in stopwords_french and word in keywords_set]
       return tokens
   except json.JSONDecodeError:
       return []


dataset['clean_comments'] = dataset['comments'].apply(extract_and_clean_comments)


# Co-occurrence analysis
cooccurrences = Counter()
for tokens in dataset['clean_comments']:
   cooccurrences.update(Counter(combinations(tokens, 2)))


# Co-occurrence visualization
G = nx.Graph()


# Add edges with weights based on co-occurrences
for (word1, word2), count in cooccurrences.items():
   if count > 2 and word1 != word2:  # Threshold to filter less frequent co-occurrences
       G.add_edge(word1, word2, weight=count)


# Node positioning in 3D
pos = nx.spring_layout(G, dim=3, k=1.0, iterations=50)


# Colors for categories with lighter shades for green and purple
color_map = {
   'Causes': '#FF6666',
   'Sentiments': '#66FF66',  # Light green
   'Conséquences': '#C299FF',  # Light purple
   'Grossesse': '#FFB6C1'  # Light pink
}


# Assign colors to nodes
node_colors = [color_map[next(category for category, words in keywords.items() if node in words)] for node in G.nodes]


# Node sizes based on co-occurrences
node_sizes = [20 + 5 * G.degree(node) for node in G.nodes]


# Extract node positions
x_nodes = [pos[node][0] for node in G.nodes]
y_nodes = [pos[node][1] for node in G.nodes]
z_nodes = [pos[node][2] for node in G.nodes]


# Extract edge positions and widths
edge_x = []
edge_y = []
edge_z = []
edge_widths = []


for edge in G.edges(data=True):
   x0, y0, z0 = pos[edge[0]]
   x1, y1, z1 = pos[edge[1]]
   edge_x.extend([x0, x1, None])
   edge_y.extend([y0, y1, None])
   edge_z.extend([z0, z1, None])
   edge_widths.append(edge[2]['weight'])


# Normalize edge widths for proper display
max_weight = max(edge_widths)
edge_widths = [2 + 8 * (weight / max_weight) for weight in edge_widths]


# Create the 3D graph with plotly
fig_3d = go.Figure()


# Add edges
for i in range(len(edge_x) // 3):
   fig_3d.add_trace(go.Scatter3d(
       x=edge_x[i * 3:(i + 1) * 3],
       y=edge_y[i * 3:(i + 1) * 3],
       z=edge_z[i * 3:(i + 1) * 3],
       mode='lines',
       line=dict(color='gray', width=edge_widths[i]),  # Gray lines
       hoverinfo='none'
   ))


# Add nodes
fig_3d.add_trace(go.Scatter3d(
   x=x_nodes, y=y_nodes, z=z_nodes,
   mode='markers+text',
   marker=dict(
       size=node_sizes,
       color=node_colors,
       opacity=0.8
   ),
   text=[f'<b>{node}</b>' for node in G.nodes],  # Bold text
   textposition="top center",
   hoverinfo='text'
))


# Add annotations for the legend
annotations = []
for idx, (category, color) in enumerate(color_map.items()):
   annotations.append(
       dict(
           x=0.02,  # Position on the left
           y=0.99 - idx * 0.05,  # Position at the top with vertical spacing between annotations
           xref='paper',
           yref='paper',
           text=f'<b>{category}</b>',
           showarrow=False,
           font=dict(
               size=14,
               color=color
           ),
           align="left"
       )
   )


# Update layout to remove axes and add annotations
fig_3d.update_layout(
   title="Co-occurrences des mots dans les commentaires",
   showlegend=False,
   annotations=annotations,
   scene=dict(
       xaxis=dict(visible=False),
       yaxis=dict(visible=False),
       zaxis=dict(visible=False)
   ),
   margin=dict(l=0, r=0, b=0, t=50),  # Increase top margin for more space
   paper_bgcolor='white'
)


# Layout for FIV sub-sections
fiv_layout = dbc.Container([
   dbc.Tabs(
       [
           dbc.Tab(label="Statistics", tab_id="graphs", label_style={'color': 'black'}),
           dbc.Tab(label="Regional Distribution", tab_id="map", label_style={'color': 'black'}),
           dbc.Tab(label="Psychological Aspect", tab_id="psychological", label_style={'color': 'black'}),
       ],
       id="fiv-tabs",
       active_tab="graphs",
   ),
   html.Div(id='fiv-content')
], fluid=True, style={'background-color': 'white'})


# Layouts for different forum selections
doctissimo_layout = dbc.Container([
   dbc.Tabs(
       [
           dbc.Tab(label="Posts Distribution", tab_id="distribution", label_style={'color': 'black'}),
           dbc.Tab(label="Word Cloud", tab_id="wordcloud", label_style={'color': 'black'}),
           dbc.Tab(label="3D", tab_id="3d-graph", label_style={'color': 'black'})
       ],
       id="doctissimo-tabs",
       active_tab="distribution"
   ),
   html.Div(id='doctissimo-content', style={'height': '90vh'})  # Add style here to increase size
], fluid=True, style={'background-color': 'white'})


# Enhanced main layout
app.layout = html.Div(
   [
       dcc.Location(id='url', refresh=False),
       html.Div(id='page-content')
   ],
   style={'background-color': 'white', 'height': '100vh', 'width': '100vw'}
   # Light pink background color and take full screen
)


@app.callback(
   Output('page-content', 'children'),
   [Input('url', 'pathname')]
)
def display_page(pathname):
   if pathname == '/application':
       return application_page()
   else:
       return presentation_page()


@app.callback(
   Output('main-content', 'children'),
   [Input("main-tabs", "active_tab")]
)
def update_content(active_tab):
   if active_tab == "fiv":
       return fiv_layout
   elif active_tab == "doctissimo":
       return doctissimo_layout
   else:
       return fiv_layout  # Default layout


@app.callback(
   Output('fiv-content', 'children'),
   [Input("fiv-tabs", "active_tab")]
)
def update_fiv_content(active_tab):
   if active_tab == "graphs":
       return dbc.Container([
           dbc.Row([
               dbc.Col(
                   [
                       html.Div(style={'height': '40px'}),  # Space equivalent to dropdown
                       dcc.Graph(id='yearly-trend-graph')
                   ], width=6
               ),
               dbc.Col(
                   [
                       html.P('Select a year:'),
                       dcc.Dropdown(
                           id='year-dropdown',
                           options=[{'label': year, 'value': year} for year in sorted(data['Year'].dropna().unique())],
                           value=sorted(data['Year'].dropna().unique())[0]
                       ),
                       dcc.Graph(id='prediction-graph')
                   ], width=6
               )
           ])
       ], fluid=True)
   elif active_tab == "map":
       return html.Div([
           html.H4("Map of ART and IVF occurrences by region"),
           html.Iframe(
               id='map',
               srcDoc=folium_map_html,
               width='100%',
               height='600'
           ),
           html.Div(id='info')
       ])
   elif active_tab == "psychological":
       return html.Div([
           dbc.Row([
               dbc.Col(
                   [
                       html.H3("Sentiment analysis", style={'text-align': 'center'}),
                       dcc.Graph(id='sentiment-pie-chart')
                   ], width=5
               ),
               dbc.Col(
                   [
                       html.H3("Topic analysis", style={'text-align': 'center'}),
                       dcc.Graph(id='word-cloud-psycho')
                   ], width=7
               )
           ]),
           dbc.Row([
               dbc.Col(
                   [
                       html.H3("Sentiment trends over time"),
                       dcc.Checklist(
                           id='year-checklist',
                           options=[{'label': str(year), 'value': year} for year in
                                    sorted(data['Year'].dropna().unique()) if year < 2014 or year > 2017],
                           value=[year for year in sorted(data['Year'].dropna().unique()) if
                                  year < 2014 or year > 2017],
                           # Select the first available year outside 2014-2018
                           labelStyle={'display': 'inline-block'}
                       ),
                       dcc.Graph(id='sentiment-trends-over-time')
                   ], width=12
               )
           ])
       ], style={'background-color': 'white'})
   else:
       return html.Div()


@app.callback(
   Output('doctissimo-content', 'children'),
   [Input("doctissimo-tabs", "active_tab")]
)
def update_doctissimo_content(active_tab):
   if active_tab == "distribution":
       return html.Div([
           dcc.Graph(id='histogram', figure=fig),
           html.Div(id='main-page', children=[
               html.Div(id='output')
           ]),
           html.Div(id='detail-page', children=[
               dcc.Graph(id='daily-graph'),
               html.Button('Back', id='back-button')
           ], style={'display': 'none'})
       ], style={'background-color': 'white'})
   elif active_tab == "wordcloud":
       return html.Div([
           html.Div(
               [
                   html.Div(
                       [
                           html.P('Legend:', style={'font-weight': 'bold', 'font-size': '24px'}),  # Add font size
                           html.P('Dark pink: Very frequent words', style={'color': 'hsl(330, 100%, 30%)'}),
                           html.P('Light pink: Less frequent words', style={'color': 'hsl(330, 100%, 60%)'}),
                       ],
                       style={'text-align': 'left', 'background-color': 'white', 'padding': '20px', 'width': '20%'}
                   ),
                   html.Div(
                       html.Img(id='wordcloud', src=wordcloud_img_src, style={'width': '75%'}),
                       style={'text-align': 'center', 'background-color': 'white', 'padding': '20px', 'width': '75%'}
                   )
               ],
               style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'justify-content': 'center'}
           )
       ], style={'background-color': 'white'})
   elif active_tab == "3d-graph":
       return html.Div([
           dcc.Graph(id='3d-graph-content', figure=fig_3d, style={'height': '85vh'})  # Adjust size here
       ], style={'background-color': 'white'})
   else:
       return html.Div()


@app.callback(
   Output('prediction-graph', 'figure'),
   Input('year-dropdown', 'value')
)
def update_graph(selected_year):
   filtered_data = data[data['Year'] == selected_year]
   yearly_counts = filtered_data.groupby('prediction').size().reset_index(name='counts')


   fig = go.Figure(data=[
       go.Bar(name='Failure',
              x=yearly_counts[yearly_counts['prediction'] == 'echec']['prediction'],
              y=yearly_counts[yearly_counts['prediction'] == 'echec']['counts'],
              marker_color='rgba(255, 99, 132, 0.6)',  # Light red
              width=0.4),
       go.Bar(name='Success',
              x=yearly_counts[yearly_counts['prediction'] == 'reussite']['prediction'],
              y=yearly_counts[yearly_counts['prediction'] == 'reussite']['counts'],
              marker_color='rgba(54, 162, 235, 0.6)',  # Light blue
              width=0.4)
   ])


   fig.update_layout(
       title='Number of Failures and Successes',
       xaxis_title='Classification',
       yaxis_title='Number',
       barmode='group',
       plot_bgcolor='white',
       paper_bgcolor='white',
       font=dict(color='black'),
       xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False),
       yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False),
       margin=dict(l=0, r=0, b=0, t=50)  # Lower the graph by reducing the top margin
   )


   return fig


@app.callback(
   Output('yearly-trend-graph', 'figure'),
   Input('year-dropdown', 'value')
)
def update_yearly_trend(selected_year):
   yearly_data = data.groupby(['Year', 'prediction']).size().unstack().fillna(0)


   fig = go.Figure()
   fig.add_trace(go.Scatter(x=yearly_data.index, y=yearly_data['echec'], mode='lines+markers', name='Failure',
                            line=dict(color='rgba(255, 182, 193, 0.7)', width=3)))
   fig.add_trace(go.Scatter(x=yearly_data.index, y=yearly_data['reussite'], mode='lines+markers', name='Success',
                            line=dict(color='rgba(135, 206, 235, 0.7)', width=3)))


   fig.update_layout(
       title='Yearly trends of failures and successes',
       xaxis_title='Year',
       yaxis_title='Number',
       plot_bgcolor='white',  # Fond blanc
       paper_bgcolor='white',  # Fond blanc
       font=dict(color='black'),
       xaxis=dict(
           showgrid=True,
           zeroline=True,
           showline=True,
           linewidth=1,
           linecolor='black',
           mirror=False,
           title='Year'  # Étiquette de l'axe des X
       ),
       yaxis=dict(
           showgrid=True,
           zeroline=True,
           showline=True,
           linewidth=1,
           linecolor='black',
           mirror=False,
           title='Number'  # Étiquette de l'axe des Y
       ),
       margin=dict(l=0, r=0, b=0, t=50)  # Réduire la marge supérieure
   )


   return fig


@app.callback(
   Output('sentiment-pie-chart', 'figure'),
   Input('sentiment-pie-chart', 'id')
)
def update_sentiment_pie_chart(_):
   sentiment_counts = df_psycho['Sentiment'].value_counts()
   fig = px.pie(values=sentiment_counts, names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={"Positive": "#CDEAC0", "Negative": "#CA3C66", "Neutral": "#E6CFF7"})
   return fig


@app.callback(
   Output('sentiment-trends-over-time', 'figure'),
   Input('year-checklist', 'value')
)
def update_sentiment_trends_over_time(selected_years):
   filtered_df = df_psycho[df_psycho['Date'].dt.year.isin(selected_years)]
   filtered_df['YearMonth'] = filtered_df['Date'].dt.to_period('M')
   sentiment_trends = filtered_df.groupby(['YearMonth', 'Sentiment']).size().unstack(fill_value=0)
   sentiment_trends_df = sentiment_trends.reset_index()
   sentiment_trends_df['YearMonth'] = sentiment_trends_df['YearMonth'].dt.to_timestamp()


   fig = px.line(sentiment_trends_df, x='YearMonth', y=sentiment_trends.columns,
                 color_discrete_map={"Positive": "#CDEAC0", "Negative": "#CA3C66", "Neutral": "#E6CFF7"})
   fig.update_layout(
       xaxis_title="Date",
       yaxis_title="Number of Posts",
       plot_bgcolor='white',  # Fond blanc
       paper_bgcolor='white',  # Fond blanc
       xaxis=dict(
           showgrid=True,
           zeroline=True,
           showline=True,
           linewidth=1,
           linecolor='black',
           mirror=False,
           title='Date'  # Étiquette de l'axe des X
       ),
       yaxis=dict(
           showgrid=True,
           zeroline=True,
           showline=True,
           linewidth=1,
           linecolor='black',
           mirror=False,
           title='Number of Posts'  # Étiquette de l'axe des Y
       )
   )
   return fig


@app.callback(
   Output('word-cloud-psycho', 'figure'),
   Input('word-cloud-psycho', 'id')
)
def update_word_cloud_psycho(_):
   # Flatten the keywords dictionary into a single list of keywords
   keyword_list = [word for category in psych_keywords.values() for word in category]
   # Filter the content to include only the keywords
   filtered_text = ' '.join(
       [word for word in ' '.join(df_psycho['Processed_Contenu'].dropna().tolist()).split() if word in keyword_list])


   # Define a function to generate darker shades of pink
   def dark_pink_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
       return "hsl(330, 100%, {}%)".format(30 + font_size % 30)


   wordcloud = WordCloud(stopwords=french_stopwords, color_func=dark_pink_color_func,
                         background_color='white').generate(filtered_text)


   # Save word cloud to a file
   img = BytesIO()
   plt.figure(figsize=(20, 10))
   plt.imshow(wordcloud, interpolation='bilinear')
   plt.axis('off')
   plt.savefig(img, format='png')
   plt.close()
   img.seek(0)
   encoded_image = base64.b64encode(img.getvalue()).decode('utf-8')


   fig = {
       "data": [],
       "layout": {
           "images": [
               {
                   "source": f"data:image/png;base64,{encoded_image}",
                   "xref": "paper",
                   "yref": "paper",
                   "x": 0.5,
                   "y": 0.5,
                   "sizex": 1,
                   "sizey": 1,
                   "xanchor": "center",
                   "yanchor": "middle",
                   "opacity": 1,
                   "layer": "below"
               }
           ],
           "xaxis": {"visible": False},
           "yaxis": {"visible": False},
           "plot_bgcolor": 'white',
           "paper_bgcolor": 'white'
       }
   }
   return fig


@app.callback(
   [Output('main-page', 'style'),
    Output('detail-page', 'style'),
    Output('daily-graph', 'figure')],
   [Input('histogram', 'clickData'),
    Input('back-button', 'n_clicks'),
    Input('daily-graph', 'clickData')],
   [State('url', 'pathname'),
    State('daily-graph', 'figure')]
)
def display_page(hist_clickData, n_clicks, daily_clickData, pathname, daily_fig):
   ctx = dash.callback_context


   if not ctx.triggered or (ctx.triggered[0]['prop_id'] == 'back-button.n_clicks' and n_clicks):
       return {'display': 'block'}, {'display': 'none'}, dash.no_update


   if ctx.triggered[0]['prop_id'] == 'histogram.clickData' and hist_clickData:
       # Get click information
       month = hist_clickData['points'][0]['x']
       year_index = hist_clickData['points'][0]['curveNumber']
       selected_year = years[year_index]


       # Filter the dataset for the selected month and year
       filtered_data = dataset[(dataset['year'] == selected_year) & (dataset['month'] == month)]
       print(f"Filtered Data for {month} {selected_year}:\n", filtered_data)  # Debugging print


       # Check for existence of filtered data
       if filtered_data.empty:
           return {'display': 'block'}, {'display': 'none'}, dash.no_update


       # Ensure nbinsx is an integer
       num_days = int(filtered_data['last_post_date'].dt.days_in_month.iloc[0])


       # Get color for selected year
       year_color = px.colors.qualitative.Plotly[year_index % len(px.colors.qualitative.Plotly)]


       # Create a histogram of posts by day for the selected month and year
       daily_fig = px.histogram(filtered_data, x='last_post_date', nbins=num_days,
                                labels={'last_post_date': 'Date', 'count': 'Number of Posts'},
                                title=f'Distribution of posts in {month} {selected_year}',
                                histfunc='count')


       daily_fig.update_traces(marker_color=year_color)
       daily_fig.update_layout(
           yaxis_title="Number of posts",
           plot_bgcolor='white',  # White background
           paper_bgcolor='white',  # White background
           font=dict(color='black'),  # Text color black
           xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False),
           yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False)
       )
       print(f"Daily Graph for {month} {selected_year}:\n", daily_fig)  # Debugging print


       return {'display': 'none'}, {'display': 'block'}, daily_fig


   if ctx.triggered[0]['prop_id'] == 'daily-graph.clickData' and daily_clickData:
       # Get click information
       date_clicked = pd.to_datetime(daily_clickData['points'][0]['x'])


       # Filter the dataset for the selected date
       filtered_data = dataset[dataset['last_post_date'].dt.date == date_clicked.date()]
       print(f"Filtered Data for {date_clicked.date()}:\n", filtered_data)  # Debugging print


       # Check for existence of filtered data
       if filtered_data.empty:
           return {'display': 'none'}, {'display': 'block'}, dash.no_update


       # Extract hour from the post date
       filtered_data['hour'] = filtered_data['last_post_date'].dt.hour


       # Extract and count comments
       filtered_data['num_comments'] = filtered_data['comments'].apply(lambda x: len(eval(x)))


       # Get color for selected year
       year_index = list(years).index(str(date_clicked.year))
       year_color = px.colors.qualitative.Plotly[year_index % len(px.colors.qualitative.Plotly)]


       # Create a histogram of number of comments per post for the selected date
       hourly_fig = px.histogram(filtered_data, x='hour', y='num_comments', nbins=24,
                                 labels={'hour': 'Hour', 'num_comments': 'Number of Comments'},  # Change label here
                                 title=f'Number of comments per post on {date_clicked.date()}',
                                 histfunc='count')
       hourly_fig.update_traces(marker_color=year_color)
       hourly_fig.update_layout(
           xaxis=dict(tickmode='linear', dtick=1, range=[0, 23], showgrid=False, zeroline=False, showline=True,
                      linewidth=1, linecolor='black', mirror=False),
           yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=False,
                      title="Number of Comments"),  # Change label here
           plot_bgcolor='white',  # White background
           paper_bgcolor='white',  # White background
           font=dict(color='black')  # Text color black
       )
       print(f"Hourly Graph for {date_clicked}:\n", hourly_fig)  # Debugging print


       return {'display': 'none'}, {'display': 'block'}, hourly_fig


# Run the app
if __name__ == '__main__':
   app.run_server(debug=True, port=8052)  # Use a different port

