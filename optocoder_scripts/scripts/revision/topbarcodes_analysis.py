import os
import argparse
from re import search
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
import plotly.express as px

from ..helpers.utils import load_yaml
from ..tools.barcode_matcher import BarcodeMatcher

def plot_topbarcodes_analysis(puck_paths, save_path):

    matcher = BarcodeMatcher()
    scores = pd.DataFrame()
    match_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    filtered_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for puck_name, puck_path in puck_paths.items():
        puck_meta = load_yaml(puck_path['config_path'])
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck
        head, tail = os.path.split(illumina_path)
        illumina_path = os.path.join(head, puck_name + '_300k.txt')
        puck_barcodes_path = os.path.join(puck_meta['output_path'])
    
        illumina = np.genfromtxt(illumina_path, dtype='str')
        save_folder = os.path.join(save_path, 'topbarcodes')
        os.makedirs(save_folder, exist_ok=True)

        methods = ['naive', 'only_ct', 'phasing', 'gb']
        method_names = ['Naive', 'Crosstalk Correction', 'Phasing Correction', 'Machine Learning']

        match_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        scores = pd.DataFrame()
        matcher = BarcodeMatcher()
        
        lig_seq = puck_meta['lig_seq'] if puck_path['is_solid'] else None
        nuc_seq = puck_meta['nuc_seq'] if puck_path['is_solid'] else None
        search_range = np.arange(0, 301000, 1000)
        optical_bc_path = puck_meta['output_path']
        if puck_path['puckcaller_barcodes_path'] != "":
            num_matches_puckcaller = matcher.match(puck_path['puckcaller_barcodes_path'], illumina_path, is_solid=puck_path['is_solid'], lig_seq=lig_seq, nuc_seq=nuc_seq, is_optocoder=False, num_illumina_to_use=search_range)
            match_counts[puck_path['name']]['Puckcaller'] = num_matches_puckcaller

        for name, method in zip(method_names, methods):
            optical_bc_path_method = os.path.join(optical_bc_path, f'predictions_{method}.csv')

            num_matches = matcher.match(optical_bc_path_method, illumina_path, is_solid=puck_path['is_solid'], lig_seq=lig_seq, nuc_seq=nuc_seq, is_optocoder=True, num_illumina_to_use=search_range)
            match_counts[puck_path['name']][name] = num_matches

        colors = ['#7D5A5A', '#E58606', '#5D69B1', '#99C945', '#CC61B0']
        if puck_path['puckcaller_barcodes_path'] == "": colors = colors[1:]

        for sample, counts in match_counts.items():
            print(f'Sample -> {sample}')
            counts = pd.DataFrame(counts)
            counts = counts.set_index(search_range)
            print(counts)
            tick_font_params = dict(tickfont = dict(family="Arial",size=55))

            fig = px.line(counts, color_discrete_sequence=colors)
            fig.update_traces(line=dict(width=8))
            fig.update_layout(
                width=1300,
                height=950,
                paper_bgcolor='white',
                plot_bgcolor='white',
            )
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True)
            fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True)
            fig.update_layout(
                xaxis_title="# of Top Illumina Barcodes Used",
                yaxis_title="# of Barcode Matches",
                legend_title="Barcode Type",
                font=dict(
                    family="Arial",
                    size=45,
                )
            )
            fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
            fig.update_layout(xaxis = tick_font_params, yaxis=tick_font_params)
            fig.write_image(os.path.join(save_folder, sample + '.svg'))
