import numpy as np
import pandas as pd
import os
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from ...helpers.utils import load_yaml
from ...tools.barcode_matcher import BarcodeMatcher
import pickle
from optocoder.machine_learning.ml_basecaller import ML_Basecaller

def get_barcode_splits(matching_beads, barcode_path):
    # this function basically checks with barcodes are initially matched or not (i.e training/testing sets)

    # read the barcodes
    barcodes = pd.read_csv(barcode_path, sep='\t')
    # get the beads that match orignally or not
    matching_barcodes = barcodes[barcodes['bead_id'].isin(matching_beads)]
    non_matching_barcodes = barcodes[~barcodes['bead_id'].isin(matching_beads)]

    return matching_barcodes

def plot_specificity(puck_paths, save_path):

    # create a barcode matcher
    matcher = BarcodeMatcher()
    match_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    texts = []
    # iterate through the pucks
    for puck_name, puck_path in puck_paths.items():

        puck_meta = load_yaml(puck_path['config_path'])
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck

        # path for the analysis barcode, these barcodes are the ones that are predicted for all beads!
        puck_barcodes_path = os.path.join(save_path, 'ml_specificity_analysis', 'predictions', puck_name) 
   
        # read the experiment file for the puck to predict things for
        with open(os.path.join(puck_meta['output_path'], 'experiment.pkl'), 'rb') as input:
            experiment = pickle.load(input)
        
        # read illumina barcodes
        illumina_barcodes = np.genfromtxt(illumina_path, dtype='str')
        
        # we create a ml basecaller to get the matching barcodes etc. a bit messy but fine for now
        if puck_path['is_solid']:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'])
        else:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=puck_path['is_solid'])

        # these are the barcodes that were match via the phasing!!!
        matching_bead_ids = ml_basecaller.matching_beads['bead_id']

        # now we check our models
        for model_full_name, model in zip(['GB', 'MLP', 'RF', 'RNN'],['gb', 'mlp', 'rf', 'rnn']):
            # this is the prediction file for the all bead prediction settings
            all_beads_predicted_barcodes_path = os.path.join(puck_barcodes_path, f'predictions_{model}.csv')
            # this is the original setting where we only predict non-matching barcodes
            original_predicted_barcodes_path = os.path.join(puck_meta['output_path'], f'predictions_{model}.csv')

            all_beads_training = get_barcode_splits(matching_bead_ids, all_beads_predicted_barcodes_path)
            original_training = get_barcode_splits(matching_bead_ids, original_predicted_barcodes_path)
            if puck_path['is_solid']:
                num_matches_orig = matcher.match(original_training, illumina_path, optical_loaded=True, is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'], is_optocoder=True)
                num_matches_all = matcher.match(all_beads_training, illumina_path, optical_loaded=True,is_solid=puck_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'], is_optocoder=True)

            else:
                num_matches_orig = matcher.match(original_training, illumina_path, optical_loaded=True,is_solid=puck_path['is_solid'], is_optocoder=True)
                num_matches_all = matcher.match(all_beads_training, illumina_path,optical_loaded=True, is_solid=puck_path['is_solid'], is_optocoder=True)
            match_counts['Original'][puck_name][model_full_name] = num_matches_orig
            match_counts['Tested'][puck_name][model_full_name] = num_matches_all
            mismatch_percentage = "{:.1f}".format((100*(num_matches_orig - num_matches_all))/len(matching_bead_ids))
            texts.append(f'{num_matches_orig - num_matches_all} ({mismatch_percentage}%)')
            match_counts['diff'][puck_name][model_full_name] = num_matches_orig - num_matches_all

    df = pd.DataFrame(match_counts['diff'])
    fig = px.imshow(df, color_continuous_scale=px.colors.sequential.Reds)
    fig.update_xaxes(side="top")
    z_text = np.array(texts).reshape(df.shape).T

    fig.update_traces(text=z_text, texttemplate="%{text}")
    fig.update_layout(
        width=1400,
        height=1400
    )
    fig.update_layout(
        yaxis_title="Pucks",
        xaxis_title="Machine Learning Method",
        font=dict(
            family="Arial",
            size=50,
        ),
        xaxis={'type':'category'},  yaxis={'type':'category'}
    )
    fig.update_xaxes(title_font_family="Arial", title_font_size=95)
    fig.update_yaxes(title_font_family="Arial", title_font_size=95)
    fig.write_image(os.path.join(save_path, 'ml_specificity_analysis', 'diff.svg'))