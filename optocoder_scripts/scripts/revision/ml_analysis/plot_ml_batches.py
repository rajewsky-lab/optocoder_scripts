import numpy as np
import pandas as pd
import os
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from ...helpers.utils import load_yaml
from ...tools.barcode_matcher import BarcodeMatcher

def plot_ml_batch_comparison(puck_paths, save_path):
    matcher = BarcodeMatcher()
    # model, test puck, training puck
    match_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for puck_name, puck_path in puck_paths.items():
        puck_meta = load_yaml(puck_path['config_path'])
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck

        puck_barcodes_path = os.path.join(save_path, 'ml_batch_comparison', 'predictions', puck_name)
    
        illumina = np.genfromtxt(illumina_path, dtype='str')

        for model in ['gb', 'mlp', 'rf', 'rnn']:
            for puck_name_models, puck_path_models in puck_paths.items():
                barcodes_path = os.path.join(puck_barcodes_path, f'predictions_{model}_{puck_name_models}.csv')
                if puck_path['is_solid']:
                    num_matches = matcher.match(barcodes_path, illumina_path, is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'], is_optocoder=True)
                else:
                    num_matches = matcher.match(barcodes_path, illumina_path, is_solid=puck_path['is_solid'], is_optocoder=True)
 
                #num_matches = len(set(barcodes).intersection(illumina))
                match_counts[model][puck_path['name']][puck_path_models['name']] = num_matches

    tick_font_params = dict(tickfont = dict(family="Arial",size=38))
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF']
    for model in ['gb', 'mlp', 'rf', 'rnn']:
        df = pd.DataFrame(match_counts[model])
        #df = pd.melt(df, id_vars=['Training Puck'])
        #print(df)
        fig = px.imshow(df, text_auto=".2s", color_continuous_scale=px.colors.sequential.Reds)
        fig.update_xaxes(side="top")

        fig.update_layout(
        width=1400,
        height=1400
    )
        fig.update_layout(
        yaxis_title="Training Puck",
        xaxis_title="Prediction Puck",
        font=dict(
            family="Arial",
            size=50,
        ),
        xaxis={'type':'category'},  yaxis={'type':'category'}
    )
        fig.update_xaxes(title_font_family="Arial", title_font_size=95)
        fig.update_yaxes(title_font_family="Arial", title_font_size=95)
        fig.write_image(os.path.join(os.path.join(save_path, 'ml_batch_comparison'), f'{model}.svg'))
