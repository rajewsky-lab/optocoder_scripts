import numpy as np
import pandas as pd
import os
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from ...helpers.utils import load_yaml
from ...tools.barcode_matcher import BarcodeMatcher
from optocoder.machine_learning.ml_basecaller import ML_Basecaller
import pickle

def get_barcode_splits(matching_beads, barcodes):
    # this function basically checks with barcodes are initially matched or not (i.e training/testing sets)

    # read the barcodes
    # get the beads that match orignally or not
    matching_barcodes = barcodes[barcodes['bead_id'].isin(matching_beads)]
    non_matching_barcodes = barcodes[~barcodes['bead_id'].isin(matching_beads)]

    return non_matching_barcodes

def get_cycle_scores(optical_path, num_cycles):
        optical = pd.read_csv(optical_path, sep="\t")

        optical_scores = optical.iloc[:, 4:]
        optical_scores = np.array(optical_scores)
        optical_scores = np.hsplit(optical_scores, num_cycles)

        cycle_scores = []
        for cycle in optical_scores:
                maxes = np.max(cycle, axis=1)
                cycle_scores.append(maxes)

        cycle_scores = np.array(cycle_scores)
        return cycle_scores

def plot_ml_curves(puck_paths, save_path):

    matcher = BarcodeMatcher()
    scores = pd.DataFrame()
    match_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    filtered_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for puck_name, puck_path in puck_paths.items():
        puck_meta = load_yaml(puck_path['config_path'])
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck

        puck_barcodes_path = os.path.join(puck_meta['output_path'])
    
        illumina = np.genfromtxt(illumina_path, dtype='str')
        save_folder = os.path.join(save_path, 'chastity_analysis')
        os.makedirs(save_folder, exist_ok=True)



        with open(os.path.join(puck_meta['output_path'], 'experiment.pkl'), 'rb') as input:
            experiment = pickle.load(input)
        
        # read illumina barcodes
        illumina_barcodes = np.genfromtxt(illumina_path, dtype='str')
        
        # we create a ml basecaller to get the matching barcodes etc. a bit messy but fine for now
        if puck_path['is_solid']:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'])
        else:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=puck_path['is_solid'])

        matching_bead_ids = ml_basecaller.matching_beads['bead_id']

        for model in ['rf', 'mlp', 'rnn', 'gb']:
            barcodes_path = os.path.join(puck_barcodes_path, f'predictions_{model}.csv')
            barcodes_full = pd.read_csv(barcodes_path, sep='\t')
            score_frame = get_cycle_scores(barcodes_path, puck_meta['num_cycles'])
            mean_scores = np.mean(score_frame, axis=0)
            barcode_counts = len(barcodes_full)
            threshold_range = np.arange(0.25, 1.0, 0.05)


            for threshold in threshold_range:
                idx = np.where(mean_scores >= threshold)[0]
                barcodes = barcodes_full.iloc[idx,:]
                original_training = get_barcode_splits(matching_bead_ids, barcodes)
                if puck_path['is_solid']:
                    num_matches_orig = matcher.match(original_training, illumina_path, optical_loaded=True, is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'], is_optocoder=True)
                else:
                    num_matches_orig = matcher.match(original_training, illumina_path, optical_loaded=True,is_solid=puck_path['is_solid'], is_optocoder=True)

                match_counts[puck_path['name']][model]['matches'][threshold] = num_matches_orig
                match_counts[puck_path['name']][model]['mismatches'][threshold] = (barcode_counts - num_matches_orig) / barcode_counts
    
    colors = px.colors.qualitative.T10
    for puck in match_counts:
        fig = go.Figure()
        for i, method in enumerate(['rf', 'mlp', 'rnn', 'gb']):
            data = pd.DataFrame(match_counts[puck][method])
            print(data)
            tick_font_params = dict(tickfont = dict(family="Arial",size=55))
            fig.add_trace(go.Scatter(x=threshold_range, y=data['matches'], name=method, mode='lines', line=dict(color=colors[i], width=10)))


            fig.update_layout(
                width=1750,
                height=930,
                paper_bgcolor='white',
                plot_bgcolor='white',
            )
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridwidth=0.5, gridcolor='rgb(105, 105, 105, 60)', mirror=True)
            fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black', gridcolor='rgb(0, 0, 0, 0)', mirror=True)
            
            fig.update_layout(
                xaxis_title='Prediction Probability Threshold',
                yaxis_title="# of Matches",
                legend_title="Barcode Type",
                font=dict(
                    family="Arial",
                    size=45,
                )
            )
            fig.update_layout(xaxis = tick_font_params, yaxis=tick_font_params)
        fig.write_image(os.path.join(save_folder, f'{puck}_ml.svg'))

