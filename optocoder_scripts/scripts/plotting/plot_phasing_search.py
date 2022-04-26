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
from ..helpers.utils import load_yaml

def plot_phasing_heatmaps(samples, save_folder):
    # here we plot the heatmaps for the phasing search grid

    # folder to save the plots
    save_folder = os.path.join(save_folder, 'phasing_plots')
    os.makedirs(save_folder, exist_ok=True) # create if not exists

    # some plotting params
    tick_font_params = dict(tickfont = dict(family="Arial",size=38))

    # iterate through the samples
    for name, sample in samples.items():
        sample_data = load_yaml(sample['config_path'])
        optical_bc_path = sample_data['output_path']

        # read the phasing grid path and load
        phasing_grid_path = os.path.join(optical_bc_path, 'phasing_grid.npy')
        phasing_grid = np.load(phasing_grid_path, allow_pickle=True)[()]

        ser = pd.Series(list(phasing_grid.values()),
                    index=pd.MultiIndex.from_tuples(phasing_grid.keys()))
        df = ser.unstack().fillna(0)
        fig = px.imshow(df, text_auto=".2s", color_continuous_scale=px.colors.sequential.Reds)
        fig.update_layout(
        width=1400,
        height=1400
        )
        fig.update_layout(
        yaxis_title="Phasing Probability",
        xaxis_title="Pre-phasing Probability",
        font=dict(
            family="Arial",
            size=42,
        ),
        xaxis={'type':'category'},  yaxis={'type':'category'}
        )
        fig.update_xaxes(title_font_family="Arial", title_font_size=65)
        fig.update_yaxes(title_font_family="Arial", title_font_size=65)

        fig.write_image(os.path.join(save_folder, name + '.png'))

