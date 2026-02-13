from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pdb
import os

########### Here are some parameters that you will need to change ##############################
PROJ_ROOT = Path('/home/timsit/riskychoice-paper-code/')  # Path to folder containing train.py, plots.py etc.
INTERIM_DATA_DIR = '/media/timsit/X9Pro1/'  # Path to folder containing widefield+behaviour .mat files
# and also (optionally) the allenDorsalMap.mat file
# Path to folder containing the allenDorsalMap.mat file
ALLEN_DATA_DIR = '/mnt/ogma/manuscripts/alm-risky-choice-2026/widefield/decoding/'


# Optional parameters 
PROCESSED_DATA_DIR = PROJ_ROOT / 'processed_data'  # Path to save decoding results, note that by default
# they will be saved in PROJ_ROOT / 'processed_data' / 'decoding'

# Path to save the figures
FIGURES_DIR = PROJ_ROOT / "figures"

# Optional parameters if pixel_ave.mat files are not there
NPY_MATLAB_DIR = '/home/timsit/npy-matlab/npy-matlab/'

####################################################################################################



# Analysis and plotting parameter files
TRAIN_YAML = PROJ_ROOT / 'train.YAML'
PLOTS_YAML = PROJ_ROOT / 'plots.YAML'
DATASET_YAML = PROJ_ROOT / 'dataset.YAML'

# Risky choice anaylsis
RISKYCHOICE_SUBJECTS = ['IAA-1125880', 'OLI-M-0021', 'OLI-M-0022', 'OLI-M-0041', 'OLI-M-0042', 'OLI-M-0057']
# NOTE: 'IAA-1125879' excluded
RC_WF_PROP = {
    'fs': 30,
    'pre_stim': 1, # 2,  # 2025-08-11 changed from 2 to 1 for new decon data 
    'post_stim': 5,
}

# Convert figure names used during analysis to figure name used in the paper
FIGURE_NAMES = {
    'all_subjects_LotteryChoice_decoding_using_pixel_ave' : 'figure_2i',
    'all_subjects_LeftChoice_decoding_using_pixel_ave': 'figure_2k',
    'all_subjects_AirPuffLeft_decoding_using_pixel_ave': 'figure_S5e',
    'all_subjects_LotteryChoice_cross_window_decoding_using_pixel_ave': 'figure_S5g',
    'all_subjects_AirPuffLeft_cross_window_decoding_using_pixel_ave': 'figure_S5h',
    'all_subjects_LeftChoice_cross_window_decoding_using_pixel_ave': 'figure_S5i',
    'riskychoice_single_region_decoding': 'figure_s5f',
}