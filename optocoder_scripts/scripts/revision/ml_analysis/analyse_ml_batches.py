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

def analyse_ml_batches(samples, save_path):
    # This scripts tests the matching accuracy when a puck is predicted with a model 
    # that is trained with another puck
    main_save_path = save_path
    # We itereate through pucks (i.e the pucks to be predicted)
    for puck_name, sample_path in samples.items():
        print('============================================')
        print(f'Predicting for puck {puck_name}')

        # load puck metadata for the run to be plotted
        puck_meta = load_yaml(sample_path['config_path'])
        puck_path = puck_meta['output_path'] # output path of the puck (i.e the location of the optocoder output)
        illumina_path = puck_meta['illumina_path'] # illumina barcode path of the puck
    
        # read the experiment file for the puck to predict things for
        with open(os.path.join(puck_path, 'experiment.pkl'), 'rb') as input:
            experiment = pickle.load(input)

        # read the illumina barcodes as well
        illumina_barcodes = np.genfromtxt(illumina_path, dtype=str)
        
        save_path = os.path.join(main_save_path, 'ml_batch_comparison', 'predictions', puck_name)
        os.makedirs(save_path, exist_ok=True)
        
        is_solid=sample_path['is_solid'] if 'is_solid' else False
        lig_seq=puck_meta['lig_seq'] if 'lig_seq' in puck_meta else []
        nuc_seq=puck_meta['nuc_seq'] if 'nuc_seq' in puck_meta else []

        # create a ml basecaller
        ml_basecaller = ML_Basecaller(experiment, illumina_barcodes, save_path, is_solid=is_solid, lig_seq=lig_seq, nuc_seq=nuc_seq)

        # now we iterate through other batches' trained models
        for puck_name_models, puck_info_models in samples.items():
            puck_path_models_meta = load_yaml(puck_info_models['config_path'])
            puck_path_models = puck_path_models_meta['output_path']
            puck_ml_path = os.path.join(puck_path_models, 'ml', 'intermediate')

            # iterate through the trained models
            for model in ['gb', 'mlp', 'rf']:
                model_path = os.path.join(puck_ml_path, f'{model}.pickle')
                print(f'Loading model from: {model_path} of {puck_name_models}')
                loaded_model = load(model_path)
                ml_basecaller.load_model(model + "_" + puck_name_models, loaded_model)
                ml_basecaller.predict(model + "_" + puck_name_models)
                output_prediction_data(experiment.beads, experiment.num_cycles, save_path, model + "_" + puck_name_models)
                
            rnn_model = RNNHyperModel(puck_meta['num_cycles'])
            tuner = RandomSearch(
                    rnn_model,
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=1,
                    directory=puck_ml_path,
                    project_name=puck_name_models)
            tuner.reload()
            loaded_model = tuner.get_best_models(num_models=1)[0]
            ml_basecaller.load_model('rnn' + "_" + puck_name_models, loaded_model)
            ml_basecaller.predict('rnn' + "_" + puck_name_models)
            output_prediction_data(experiment.beads, experiment.num_cycles, save_path, 'rnn' + "_" + puck_name_models)

