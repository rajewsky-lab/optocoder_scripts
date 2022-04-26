import numpy as np
import pickle
import os
from joblib import dump, load
from tensorflow import keras
from reload_rnn import RNNHyperModel
from kerastuner.tuners import RandomSearch

inhouse_folder = "/home/asenel/Projects/deployment/outputs/optocoder_v0.1.1/inhouse"
slideseq_folder = "/home/asenel/Projects/deployment/outputs/optocoder_v0.1.1/slideseq"

inhouse_pucks = [ item for item in os.listdir(inhouse_folder) if os.path.isdir(os.path.join(inhouse_folder, item)) ]
slideseq_pucks = [ item for item in os.listdir(slideseq_folder) if os.path.isdir(os.path.join(slideseq_folder, item)) ]

for puck in inhouse_pucks:
    puck_ml_path = os.path.join(inhouse_folder, puck, 'ml', 'intermediate')
    print(puck)
    print('---------------------------------------')
    for model in ['gb', 'mlp', 'rf']:
        model_path = os.path.join(puck_ml_path, f'{model}.pickle')
        model = load(model_path)
        print(model)

    rnn_model = RNNHyperModel(12)
    tuner = RandomSearch(
            rnn_model,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory=puck_ml_path,
            project_name=puck)
    tuner.reload()
    model = tuner.get_best_models(num_models=1)[0]
    print(model.get_config())
    print(model.optimizer.get_config())
    print('---------------------------------------')
