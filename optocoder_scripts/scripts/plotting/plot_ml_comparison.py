import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
from ..tools.barcode_matcher import BarcodeMatcher
from ..helpers.utils import load_yaml

def plot_ml_comparison(samples, save_folder, name_save):
    # here we plot the number of matches for different machine learning models

    # folder to save the plots
    save_folder = os.path.join(save_folder, 'ml_comparison_plots')
    os.makedirs(save_folder, exist_ok=True) # create if not exists

    # methods to compare
    methods = ['rf', 'mlp', 'rnn', 'gb']
    method_names = ['Random Forest', 'Multilayer Perceptron', 'Recurrent Neural Networks', 'Gradient Boosting']

    ### CALCULATE THE MATCHES AND SCORES
    match_counts = defaultdict(lambda: defaultdict(int)) # this will keep the number of matches
    matcher = BarcodeMatcher()

    # we iterate through samples
    for name, sample in samples.items():
        sample_data = load_yaml(sample['config_path']) # load sample run info
        optical_bc_path = sample_data['output_path'] # path of the outputs (i.e where the results are)
        illumina_path = sample_data['illumina_path'] # path of the illumina barcodes
    
        # set lig seq and nuc seq if it is a solid seqeuncing sample
        lig_seq = sample_data['lig_seq'] if sample['is_solid'] else None
        nuc_seq = sample_data['nuc_seq'] if sample['is_solid'] else None
        
        # iterate methods
        for name, method in zip(method_names, methods):
            optical_bc_path_method = os.path.join(optical_bc_path, f'predictions_{method}.csv')
            num_matches = matcher.match(optical_bc_path_method, illumina_path, is_solid=sample['is_solid'], lig_seq=lig_seq, nuc_seq=nuc_seq, is_optocoder=True)
            match_counts[name][sample['name']] = num_matches

    # color params for figures
    colors = px.colors.qualitative.T10
    tick_font_params = dict(tickfont = dict(family="Arial",size=55))

    ## PLOT THE MATCHES
    count_frame = pd.DataFrame([(k,k1,v1) for k,v in match_counts.items() for k1,v1 in v.items()], columns = ['Machine Learning Model','Puck','Matches'])

    fig = px.bar(count_frame, y="Matches", x="Puck", color="Machine Learning Model", color_discrete_sequence=colors)
    fig.update_layout(barmode='group')
    fig.update_traces(marker_line_width=2, marker_line_color='rgb(0,0,0)')

    fig.update_layout(
        width=1800,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True)
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True)
    fig.update_layout(
        xaxis_title="Pucks",
        yaxis_title="# of Barcode Matches",
        legend_title="Machine Learning Model",
        font=dict(
            family="Arial",
            size=45,
        )
    )
    fig.update_layout(xaxis = tick_font_params, yaxis=tick_font_params)

    fig.write_image(os.path.join(save_folder, 'matches_' + name_save+ '.svg'))