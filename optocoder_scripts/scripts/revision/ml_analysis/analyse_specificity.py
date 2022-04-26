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
from joblib import dump, load
import pickle
from tensorflow import keras
from .reload_rnn import RNNHyperModel
from kerastuner.tuners import RandomSearch
from optocoder.machine_learning.ml_basecaller import ML_Basecaller
from optocoder.evaluation.evaluate_basecalls import output_prediction_data
from ...helpers.utils import load_yaml

def analyse_ml_specificity(samples, main_save_path):

    for puck_name, sample_path in samples.items():
        print('===========================')
        print(f'Analysing ml specificity for puck {puck_name}')

        save_path = os.path.join(main_save_path, 'ml_specificity_analysis', 'predictions', puck_name)
        os.makedirs(save_path, exist_ok=True)

        # load puck metadata for the run to be plotted
        puck_meta = load_yaml(sample_path['config_path'])
        puck_path = puck_meta['output_path'] # output path of the puck (i.e the location of the optocoder output)
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck
        puck_ml_path = os.path.join(puck_path, 'ml', 'intermediate')
            # read the experiment file for the puck to predict things for
        with open(os.path.join(puck_path, 'experiment.pkl'), 'rb') as input:
            experiment = pickle.load(input)
        illumina_barcodes = np.genfromtxt(illumina_path, dtype=str)
        if sample_path['is_solid']:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=sample_path['is_solid'], lig_seq=puck_meta['lig_seq'], nuc_seq=puck_meta['nuc_seq'])
        else:
            ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=sample_path['is_solid'])

        # iterate through the trained models
        for model in ['gb', 'mlp', 'rf']:
            model_path = os.path.join(puck_ml_path, f'{model}.pickle')
            print(f'Loading model from: {model_path}')
            loaded_model = load(model_path)
            ml_basecaller.load_model(model, loaded_model)
            ml_basecaller.predict_all(model)
            output_prediction_data(experiment.beads, experiment.num_cycles, save_path, model)

                
        rnn_model = RNNHyperModel(puck_meta['num_cycles'])
        tuner = RandomSearch(
                    rnn_model,
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=1,
                    directory=puck_ml_path,
                    project_name=puck_name)
        tuner.reload()
        loaded_model = tuner.get_best_models(num_models=1)[0]
        ml_basecaller.load_model('rnn', loaded_model)
        ml_basecaller.predict_all('rnn')
        output_prediction_data(experiment.beads, experiment.num_cycles, save_path, 'rnn')

