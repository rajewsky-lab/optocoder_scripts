import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from ..helpers.utils import load_yaml

def plot_ct(samples, save_folder):
    ## This function is used to plot pairwise crosstalk correction results
    # TODO: a bit ugly and repetitive, cleanup would be nice

    # some plotting setup
    sns.set_context("paper", rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":18})   
    plt.rcParams["font.family"] = "Arial"
    
    # folder to save the plots
    save_folder = os.path.join(save_folder, 'ct_plots')
    os.makedirs(save_folder, exist_ok=True) # create if not exists

    # we iterate through the samples to check the matches and scores
    for name, sample in samples.items():
        puck_save_folder = os.path.join(save_folder, name)
        os.makedirs(puck_save_folder, exist_ok=True) # create if not exists

        # read some info
        sample_data = load_yaml(sample['config_path']) # load sample run info
        optical_bc_path = sample_data['output_path'] # path of the outputs (i.e where the results are)
        illumina_path = sample_data['illumina_path'] # path of the illumina barcodes
    
        # read crosstalk corrected intensities
        ct_intensities_path = os.path.join(optical_bc_path, 'intensity_files', 'only_ct_intensities.csv')
        cycle = 0
        n = cycle * 4
        ct_intensities = pd.read_csv(ct_intensities_path, sep='\t').iloc[:, [3+n,4+n,5+n,6+n]]
        ct_intensities = ct_intensities.rename(columns={"cycle_1_ch_1": "G", "cycle_1_ch_2": "T", "cycle_1_ch_3": "A", "cycle_1_ch_4": "C"})

        # set some colors
        mapper = {'G': (31/255.0, 119/255.0, 180/255.0), 'T': (255/255.0, 127/255.0, 14/255.0), 'A': (44/255.0, 160/255.0, 44/255.0), 'C': (214/255.0, 39/255.0,40/255.0 )}
        pairs = list(itertools.combinations(['G','T', 'A', 'C'],2)) # we will plot it for all the combinations

        # iterate the pairs and plot
        for i, pair in enumerate(pairs):
            ints = ct_intensities[[pair[0], pair[1]]]
            ints['max'] = ints.idxmax(axis=1)
            hue_order = [pair[0], pair[1]]
            g = sns.PairGrid(ints, x_vars=pair[0], y_vars=pair[1], hue='max', hue_order=hue_order, palette =[mapper[pair[0]], mapper[pair[1]]])
            g.map_diag(sns.histplot)
            g.axes[0,0].set_ylim(np.min(ct_intensities.to_numpy()),np.max(ct_intensities.to_numpy()))
            g.axes[0,0].set_xlim(np.min(ct_intensities.to_numpy()),np.max(ct_intensities.to_numpy()))

            g.map_offdiag(sns.scatterplot, s=4)
            for ax in g.diag_axes: ax.set_visible(False)
            if i != 0:
                g.set(yticklabels=[])
            plt.savefig(os.path.join(puck_save_folder, f'crosstalk_ct_{pair[0]}_{pair[1]}.png'), bbox_inches='tight', dpi=350)

        # now we do everything above
        raw_intensities_path = os.path.join(optical_bc_path, 'intensity_files', 'bc_intensities.csv')
        cycle = 0
        n = cycle * 4
        raw_intensities = pd.read_csv(raw_intensities_path, sep='\t').iloc[:, [3+n,4+n,5+n,6+n]]
        raw_intensities = raw_intensities.rename(columns={"cycle_1_ch_1": "G", "cycle_1_ch_2": "T", "cycle_1_ch_3": "A", "cycle_1_ch_4": "C"})

        mapper = {'G': (31/255.0, 119/255.0, 180/255.0), 'T': (255/255.0, 127/255.0, 14/255.0), 'A': (44/255.0, 160/255.0, 44/255.0), 'C': (214/255.0, 39/255.0,40/255.0 )}
        pairs = list(itertools.combinations(['G','T', 'A', 'C'],2))
        for i, pair in enumerate(pairs):

            ints = raw_intensities[[pair[0], pair[1]]]
            ints['max'] = ints.idxmax(axis=1)
            hue_order = [pair[0], pair[1]]
            g = sns.PairGrid(ints, x_vars=pair[0], y_vars=pair[1], hue='max', hue_order=hue_order, palette =[mapper[pair[0]], mapper[pair[1]]])
            g.map_diag(sns.histplot)
            g.axes[0,0].set_ylim(np.min(raw_intensities.to_numpy()),np.max(raw_intensities.to_numpy()))
            g.axes[0,0].set_xlim(np.min(raw_intensities.to_numpy()),np.max(raw_intensities.to_numpy()))

            g.map_offdiag(sns.scatterplot, s=4)
            for ax in g.diag_axes: ax.set_visible(False)
            if i != 0:
                g.set(yticklabels=[])
            plt.savefig(os.path.join(puck_save_folder, f'crosstalk_bc_{pair[0]}_{pair[1]}.png'), bbox_inches='tight', dpi=350)
            

