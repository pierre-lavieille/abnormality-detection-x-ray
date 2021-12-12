import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import base64
from io import BytesIO
import cv2
import model_app
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from PIL import Image
import requests
import json

# name of the images
RANDOM_IDDS = ['0b0e106a53f9dc3b28a9b15f94510b7a',
               '03e6ecfa6f6fb33dfeac6ca4f9b459c9', '1138649f25528f0ab76c93ac60279ca3',
               '011ae9520e81f1efe71c9d954ec07d09', '0108949daa13dc94634a7d650a05c0bb',
               '112cf0367dd8b6aa14b4e384439d9eb7',
               '0007d316f756b3fa0baea2ff514ce945', '011244ab511b20130d846f5f8f0c3866']

# dict disseases
label_dict = dict({0: 'Aortic enlargement',
     1: 'Atelectasis',
     2: 'Calcification',
     3: 'Cardiomegaly',
     4: 'Consolidation',
     5: 'ILD',
     6: 'Infiltration',
     7: 'Lung Opacity',
     8: 'Nodule/Mass',
     9: 'Other lesion',
     10: 'Pleural effusion',
     11: 'Pleural thickening',
     12: 'Pneumothorax',
     13: 'Pulmonary fibrosis',
     14: 'No finding'})

# color for the environement    
env_colors = {
    'background': '#ZZZZZZ',
    'text': '#000000'
}

# colors for visualization
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AA0DFE', 
          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
          '#FF97FF', '#FECB52', '#2CA02C', '#FBE426',
          '#1CFFCE', '#B82E2E', '#E2E2E2'] 

# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)


# plotly.py helper functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))

# Start Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

app.layout = html.Div(className='container', style={'backgroundColor': env_colors['background']}, children=[
    Row(html.H1("Dash Detection App", 
                style={
                'textAlign': 'center',
                'color': env_colors['text']
            })),

    Row(html.P("Input Image ID:", style={'color': env_colors['text']})),
    Row([
        Column(width=6, children=[
            dcc.Input(id='input-idd', style={'width': '100%'}, placeholder='Insert ID...'),
        ]),
        Column(html.Button("Run Model", id='button-run', 
                           n_clicks=0, style={'color': env_colors['text']}), width=2),
        Column(html.Button("Random Image", id='button-random',
                           n_clicks=0, style={'color': env_colors['text']}), width=2)
    ]),
    
    html.Hr(),
    Row([
        html.Div(id='prob-classifier', style={'textAlign': 'center', 'width': '100%'}),
    ]),

    Row(children=[
        Column(width=10, 
               children=[dcc.Graph(id='model-output', style={"height": "100vh"})]),

        Column(width=2, children=[
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.P('Image Parameter:', style={'color': env_colors['text']}),
            dcc.RadioItems(
                id='image-parameter',
                options=[
                    {'label': 'Normal Image', 'value': 'normal'},
                    {'label': 'Histogram equalization', 'value': 'hist'},
                    {'label': 'Contrast Limited Adaptive', 'value': 'clahe'},
                    {'label': 'Inverse Color', 'value': 'fix_monochrome'},
                    {'label': 'Graham Preprocessing', 'value': 'graham'}
                ],
                value='normal',
                style={'color': env_colors['text']}
            ),
            
            html.Br(),
            html.Br(),
            
            html.P('Confidence Threshold:', style={'color': env_colors['text']}),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.3, 
                marks={0: '0%', 0.5: '50%', 1: '100%'})
        ])
    ]),
    dcc.Store(id='store-prediction'),
    dcc.Store(id='store-image')
])

# change the images
@app.callback(
    [Output('button-run', 'n_clicks'),
     Output('input-idd', 'value')],
    [Input('button-random', 'n_clicks')],
    [State('button-run', 'n_clicks')])
def randomize(random_n_clicks, run_n_clicks):
    idd = RANDOM_IDDS[random_n_clicks%len(RANDOM_IDDS)]
    return run_n_clicks+1, idd

# return the classifier output
@app.callback(
    [Output('store-prediction', 'data'),
     Output('prob-classifier', 'children')],
    [Input('button-run', 'n_clicks'),
     Input('input-idd', 'n_submit')],
    [State('input-idd', 'value')])
def get_predictions(n_clicks, n_submit, idd):
    path = f'image_dicom/{idd}.dicom'
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if dicom.PhotometricInterpretation == 'MONOCHROME1' :
        data = np.amax(data) - data
    im = Image.fromarray(data)
    records, prob_classifier = model_app.detect(im)
    text_to_return =  u'Abnormality probability : {}%'.format(prob_classifier*100)
    return records.to_json(date_format='iso', orient='split'), text_to_return

# return the detection output
@app.callback(
    Output('model-output', 'figure'),
    [Input('button-run', 'n_clicks'),
     Input('input-idd', 'n_submit'),
     Input('slider-confidence', 'value'),
     Input('image-parameter', 'value'),
     Input('store-prediction', 'data')],
    [State('input-idd', 'value')])
def run_model(n_clicks, n_submit, confidence, im_param, records, idd):
    #apply_nms = 'enabled' in checklist
    try:
        path = f'image_dicom/{idd}.dicom'
        dicom = pydicom.read_file(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        if im_param=='clahe':
            clahe_transformation = cv2.createCLAHE(clipLimit = 2., tileGridSize = (10, 10))
            data = clahe_transformation.apply(data)
        elif im_param=='hist':
            data = cv2.equalizeHist(data)
        elif im_param=='fix_monochrome':
            data = np.amax(data) - data
        elif im_param=='graham':
            sigmaX = 125
            data = cv2.addWeighted (data, 4, cv2.GaussianBlur(data, (0,0), sigmaX) ,-4 ,128)
            
        im = Image.fromarray(data)
        records = pd.read_json(records, orient='split')
    except:
        return go.Figure().update_layout(title=f'Incorrect ID  {idd}')
    
    boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
    scores = records['scores'].values
    class_id = records['class_id'].values
    
    thr = scores > confidence
    boxes = boxes[thr]
    scores = scores[thr]
    class_id = class_id[thr]

    fig = pil_to_fig(im, showlegend=True, title='Model Predictions')
    existing_classes = set()

    for i in range(boxes.shape[0]):
        label = label_dict[class_id[i]]
        confidence = scores[i]
        x0, y0, x1, y1 = boxes[i]

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.3, group=label, name=label, color=COLORS[class_id[i]], 
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label)
        
    fig.update_layout(
    font_color=env_colors['text'],
    #plot_bgcolor=env_colors['background']
    )

    return fig

if __name__ == '__main__':
    app.run_server(
        port=8050,
        host='0.0.0.0'
    )