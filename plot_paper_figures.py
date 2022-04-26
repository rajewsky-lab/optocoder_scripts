import argparse
from optocoder_scripts.scripts.helpers.utils import load_yaml
import os
import sys
import copy 

# we add the optocoder package to use some piece of code 
# such as from the machine learning module
sys.path.append(os.path.abspath("packages/optocoder"))

## Read metadata for the run to be plotted
parser = argparse.ArgumentParser(description="Plot figures of the results.")
parser.add_argument('-sample_list', type=str, help='Sample list path')
args = parser.parse_args()

figures_plot_run = load_yaml(args.sample_list)  # load the yaml file for the samples
samples = figures_plot_run['samples'] # samples 
save_folder = figures_plot_run['metadata']['save_path'] # path to save the figs
name = figures_plot_run['metadata']['name'] # name of the data type
save_folder = os.path.join(save_folder, name)
os.makedirs(save_folder, exist_ok=True) # create the figure save directories

## here we set what to plot
PLOT_MATCHES = True # bar/violin plots of matches and scores
PLOT_ML_COMPARISON = True # plot that compares machine learning modules
PLOT_PHASING_SEARCH = True # plots for the phasing search heatmaps
PLOT_CROSSTALK = True # scatter plots for crosstalk correction (i.e before/after)
PLOT_ML_BATCH_COMPARISON = True # heatmap for the ml batch comparison
PLOT_ML_SPECIFICITY_ANALYSIS = True # heatmap for the ml specificity analysis
PLOT_CHASTITY_ANALYSIS = True # line plots for the chastity score thresholding analysis
PLOT_TOP_BARCODES_ANALYSIS = True # cumulative plots for the top barcodes analysis

if PLOT_MATCHES:
    from optocoder_scripts.scripts.plotting.plot_matches_scores import plot_matches
    plot_matches(samples, save_folder, name)

if PLOT_CROSSTALK:
    from optocoder_scripts.scripts.plotting.plot_ct import plot_ct
    plot_ct(samples, save_folder)

if PLOT_ML_COMPARISON:
    from optocoder_scripts.scripts.plotting.plot_ml_comparison import plot_ml_comparison
    plot_ml_comparison(samples, save_folder, name)

if PLOT_PHASING_SEARCH:
    from optocoder_scripts.scripts.plotting.plot_phasing_search import plot_phasing_heatmaps
    plot_phasing_heatmaps(samples, save_folder)

if PLOT_CHASTITY_ANALYSIS:
    from optocoder_scripts.scripts.revision.chastity_analysis.plot_chastity_analysis import plot_chastity_curves
    from optocoder_scripts.scripts.revision.chastity_analysis.plot_ml_analysis import plot_ml_curves
    plot_chastity_curves(samples, save_folder)
    plot_ml_curves(samples, save_folder)

if PLOT_ML_SPECIFICITY_ANALYSIS:
    from optocoder_scripts.scripts.revision.ml_analysis.analyse_specificity import analyse_ml_specificity
    from optocoder_scripts.scripts.revision.ml_analysis.plot_specificity import plot_specificity
    analyse_ml_specificity(samples, save_folder)
    plot_specificity(samples, save_folder)

if PLOT_ML_BATCH_COMPARISON:
    from optocoder_scripts.scripts.revision.ml_analysis.analyse_ml_batches import analyse_ml_batches
    from optocoder_scripts.scripts.revision.ml_analysis.plot_ml_batches import plot_ml_batch_comparison
    samples_for_ml_comparison = copy.deepcopy(samples)
    if figures_plot_run['metadata']['type'] == 'slideseq':
        samples_for_ml_comparison.pop('v2_200115_08')
    analyse_ml_batches(samples_for_ml_comparison, save_folder)
    plot_ml_batch_comparison(samples_for_ml_comparison, save_folder)

if PLOT_TOP_BARCODES_ANALYSIS:
    from optocoder_scripts.scripts.revision.topbarcodes_analysis import plot_topbarcodes_analysis
    plot_topbarcodes_analysis(samples, save_folder)