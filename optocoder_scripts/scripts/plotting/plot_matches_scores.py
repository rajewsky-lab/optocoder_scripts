import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
from ..tools.barcode_matcher import BarcodeMatcher
from ..helpers.utils import load_yaml

def percentage_change(col1,col2):
    # helper method to calculate the percentage of matches
    return ((col2 - col1) / col1) * 100

def plot_matches(samples, save_folder, name_save):
    # here we plot the number of matches and the chastity score distributions

    # folder to save the plots
    save_folder = os.path.join(save_folder, 'match_plots')
    os.makedirs(save_folder, exist_ok=True) # create if not exists

    # the correction methods list
    methods = ['naive', 'only_ct', 'phasing', 'gb']
    method_names = ['Naive', 'Crosstalk Correction', 'Phasing Correction', 'Machine Learning']

    ### CALCULATE THE MATCHES AND SCORES
    match_counts = defaultdict(lambda: defaultdict(int)) # this will keep the matches
    scores = pd.DataFrame() # this will keep the scores

    # create a barcode matcher
    matcher = BarcodeMatcher()

    # we iterate through the samples to check the matches and scores
    for name, sample in samples.items():
        
        # read some info
        sample_data = load_yaml(sample['config_path']) # load sample run info
        optical_bc_path = sample_data['output_path'] # path of the outputs (i.e where the results are)
        illumina_path = sample_data['illumina_path'] # path of the illumina barcodes
    
        # set lig seq and nuc seq if it is a solid sequencing sample
        lig_seq = sample_data['lig_seq'] if sample['is_solid'] else None
        nuc_seq = sample_data['nuc_seq'] if sample['is_solid'] else None
        
        # if we compare the puckcaller, get the matches for it
        if sample['puckcaller_barcodes_path'] != "":
            num_matches_puckcaller = matcher.match(sample['puckcaller_barcodes_path'], illumina_path, is_solid=sample['is_solid'], lig_seq=lig_seq, nuc_seq=nuc_seq, is_optocoder=False)
            match_counts['Puckcaller'][sample['name']] = num_matches_puckcaller

        # we iterate for all the methods and caluclate the mathces
        for name, method in zip(method_names, methods):
            optical_bc_path_method = os.path.join(optical_bc_path, f'predictions_{method}.csv')
            num_matches = matcher.match(optical_bc_path_method, illumina_path, is_solid=sample['is_solid'], lig_seq=lig_seq, nuc_seq=nuc_seq, is_optocoder=True)
            match_counts[name][sample['name']] = num_matches
            # get the scores as well
            score_frame = matcher.get_scores(optical_bc_path_method, illumina_path, name, sample['name'])
            
            # we don't calculate the machine learning scores here because they are not
            # really comparable with the others
            if method != 'gb':
                scores = scores.append([score_frame])

    # calculate some increase percentages to print and see
    match_counts_percentages = pd.DataFrame(match_counts)
    match_counts_percentages['ct_increase'] = percentage_change(match_counts_percentages['Naive'], match_counts_percentages['Crosstalk Correction'])
    match_counts_percentages['phasing_increase'] = percentage_change(match_counts_percentages['Naive'], match_counts_percentages['Phasing Correction'])
    match_counts_percentages['ml_increase'] = percentage_change(match_counts_percentages['Naive'], match_counts_percentages['Machine Learning'])

    print(match_counts_percentages)

    # color params for figures
    colors = ['#7D5A5A', '#E58606', '#5D69B1', '#99C945', '#CC61B0']
    if sample['puckcaller_barcodes_path'] == "": colors = colors[1:]
    tick_font_params = dict(tickfont = dict(family="Arial",size=55))

    #### PLOTTING THINGS
    # a bit ugly for now but works

    ## PLOT THE SCORES 
    fig = go.Figure()
    fig = px.violin(scores, x="Basecalling Type", y="Score", color="Basecalling Type", facet_col="Puck", facet_col_wrap=2, color_discrete_sequence=colors)
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(
        width=1800,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True, tickfont=tick_font_params['tickfont'])
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True, tickfont=tick_font_params['tickfont'])

    fig.write_image(os.path.join(save_folder, 'chastity_' + name_save + '.svg'))

    ## PLOT THE MATCHES
    count_frame = pd.DataFrame([(k,k1,v1) for k,v in match_counts.items() for k1,v1 in v.items()], columns = ['Basecalling Type','Puck','Matches'])

    fig = px.bar(count_frame, y="Matches", x="Puck", color="Basecalling Type", color_discrete_sequence=colors)
    fig.update_layout(barmode='group')
    fig.update_traces(marker_line_width=2, marker_line_color='rgb(0,0,0)')

    fig.update_layout(
        width=1800,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True)
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True)
    fig.update_layout(
        xaxis_title="Pucks",
        yaxis_title="# of Barcode Matches",
        legend_title="Barcode Type",
        font=dict(
            family="Arial",
            size=45,
        )
    )
    fig.update_layout(xaxis = tick_font_params, yaxis=tick_font_params)

    fig.write_image(os.path.join(save_folder, 'matches_' + name_save + '.svg'))