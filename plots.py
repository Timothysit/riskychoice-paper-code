import pdb
from pathlib import Path
import os
import glob
import util as mp_util

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# CLI
import typer
from typing import List

# Plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import sciplotlib.style as splstyle
import sciplotlib.util as splutil

# saving metadata
import inspect

# stats 
import scipy.stats as sstats

import sys 


# from matchingp.modeling.regression import recover_Y_aligned

import skimage.measure as skimeasure
from config import FIGURES_DIR, FIGURE_NAMES, PROCESSED_DATA_DIR, PLOTS_YAML, INTERIM_DATA_DIR, ALLEN_DATA_DIR
import scipy.io as spio

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    process_name: str = 'window-decoding'
    # -----------------------------------------
    # plots_to_make: List[],
):

    params = mp_util.read_params(PLOTS_YAML, process_name=process_name)
    mp_util.print_params(params)

    plots_to_make = params['plots_to_make']
    data_type = params['data_type']

    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    for plots_tm in plots_to_make:

        if plots_tm == 'windowed_decoding':

            if not os.path.isdir(params['fig_folder']):
                os.makedirs(params['fig_folder'])

            for y_target in params['y_target']:
                for feature_used in params['feature_used']:
                    if params['plot_ave_across_mice']:
                        all_decoding_results = dict()
                        # TODO: currently this assumes two states...
                        accuracy_per_window_1_store_list = []
                        accuracy_per_window_2_store_list = []

                        accuracy_per_window_1_null_store_list = [] 
                        accuracy_per_window_2_null_store_list = [] 

                        state_1_weights_corr_list = []
                        state_2_weights_corr_list = []
                        across_state_corr_list = []

                        across_window_matrix_1_list = []
                        across_window_matrix_2_list = []

                    for subject_idx, subject in enumerate(params['subject']):


                        if params['model_name'] is not None:
                            try:
                                decoding_result_fpath = glob.glob(os.path.join(PROCESSED_DATA_DIR,
                                                                               'decoding',
                                '{subject}_*_locaNMF_decoding_{y_target}_using_*_{model_name}.npz'.format(
                                subject=subject, y_target=y_target,
                                model_name=params['model_name'])))[0]
                            except:
                                # just in case the user specified directly the folder with the decoding results
                                decoding_result_fpath = glob.glob(os.path.join(PROCESSED_DATA_DIR,
                                                                               '{subject}_*_locaNMF_decoding_{y_target}_using_*_{model_name}.npz'.format(
                                                                                   subject=subject, y_target=y_target,
                                                                                   model_name=params['model_name'])))[0]

                        else:
                            decoding_result_fpath = os.path.join(PROCESSED_DATA_DIR, 'decoding',
                                '{subject}_all_None_locaNMF_decoding_{y_target}_using_{feature_used}_resample_states_{resample_states}_balance_states_trialcounts_{balance_states_trialcounts}.npz'.format(
                                subject=subject, y_target=y_target, feature_used=feature_used, resample_states=params['resample_states'],
                                balance_states_trialcounts=params['balance_states_trialcounts']))

                        decoding_result = np.load(decoding_result_fpath)

                        min_trials_in_contigency_table = np.min(
                            np.concatenate([decoding_result['contigency_table_1'],
                                            decoding_result['contigency_table_2']])
                        )

                        state_1_weights_mean = np.nanmean(decoding_result['weights_per_window_1'][0, :, :, :], axis=2)
                        state_2_weights_mean = np.nanmean(decoding_result['weights_per_window_2'][0, :, :, :], axis=2)
                        state_1_weights_corr = pd.DataFrame(state_1_weights_mean.T).corr()
                        state_2_weights_corr = pd.DataFrame(state_2_weights_mean.T).corr()
                        across_state_corr = correlate_feature_matrices(state_1_weights_mean.T,
                                                                                   state_2_weights_mean.T)

                        if min_trials_in_contigency_table < params['min_trials']:
                            logger.info('Skipping {subject} from the grand average'.format(subject=subject))
                        else:
                            accuracy_per_window_1_store_list.append(decoding_result['accuracy_per_window_1'])
                            accuracy_per_window_2_store_list.append(decoding_result['accuracy_per_window_2'])
                            # mean across the shuffles here
                            accuracy_per_window_1_null_store_list.append(np.mean(decoding_result['accuracy_per_window_null_1'], axis=1))
                            accuracy_per_window_2_null_store_list.append(np.mean(decoding_result['accuracy_per_window_null_2'], axis=1))

                            state_1_weights_corr_list.append(state_1_weights_corr)
                            state_2_weights_corr_list.append(state_2_weights_corr)
                            across_state_corr_list.append(across_state_corr)

                            if 'across_window_matrix_1' in decoding_result.files:
                                across_window_matrix_1_list.append(decoding_result['across_window_matrix_1'])
                                across_window_matrix_2_list.append(decoding_result['across_window_matrix_2'])


                        if (subject_idx == 0) & params['plot_ave_across_mice']:
                            all_decoding_results['decoding_time_bins'] = decoding_result['decoding_time_bins']


                        if decoding_result['decoding_bin_width'] is not None:
                            decoding_end_bins = decoding_result['decoding_time_bins'] + decoding_result['decoding_bin_width']
                        else:
                            decoding_end_bins = decoding_result['decoding_time_bins']

                        fig, axs = plot_windowed_decoding_accuracy_mult_states(decoding_end_bins,
                                                                    decoding_result['accuracy_per_window_1'], decoding_result['accuracy_per_window_null_1'],
                                                                    decoding_result['accuracy_per_window_2'], decoding_result['accuracy_per_window_null_2'],
                                                                    choice_reward_matrix_1=decoding_result['contigency_table_1'],
                                                                    choice_reward_matrix_2=decoding_result['contigency_table_2'],
                                                                    state_names=params['state_names'], xlabel=params['xlabel'],
                                                                    ylabel=params['ylabel_dict'][y_target],
                                                                    row_names=params['row_names'],
                                                                    col_names=params['col_names'],
                                                                    vert_line_times=params['vert_line_times'],
                                                                    vert_line_labels=params['vert_line_labels'],
                                                                    vert_line_styles=params['vert_line_styles'],
                                                                    fig=None, axs=None, 
                                                                    linecolor=[params['engaged_state_color'], params['disengaged_state_color']],
                                                                    custom_xticks=params['xtick_locs'],
                                                                    custom_yticks=params['ytick_locs'][y_target],
                                                                    custom_ylim=params['ylimits'][y_target])
                        fig.suptitle('{subject} decoding using {feature_used}'.format(subject=subject, feature_used=feature_used),
                                     y=1.14)
                        fig_name = '{subject}_{y_target}_decoding_using_{feature_used}'.format(subject=subject, y_target=y_target, feature_used=feature_used)
                        splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

                        # Plot weight similarity across time
                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                            fig.set_size_inches(9, 3)

                            time_start = decoding_end_bins[0]
                            time_end = decoding_end_bins[-1]

                            axs[0].imshow(state_1_weights_corr, vmin=-1, vmax=1, cmap='bwr',
                                          extent=[time_start, time_end, time_end, time_start])
                            axs[1].imshow(pd.DataFrame(state_2_weights_mean.T).corr(), vmin=-1, vmax=1, cmap='bwr',
                                          extent=[time_start, time_end, time_end, time_start])


                            im2 = axs[2].imshow(across_state_corr, vmin=-1, vmax=1, cmap='bwr',
                                                extent=[time_start, time_end, time_end, time_start])

                            line_times = params['vert_line_times']

                            for l_t in line_times:
                                [ax.axvline(l_t, linestyle='--', color='black', lw=1) for ax in axs]
                                [ax.axhline(l_t, linestyle='--', color='black', lw=1) for ax in axs]

                            axs[0].set_xlabel('Time from trial start (s)', size=11)
                            axs[0].set_ylabel('Time from trial start (s)', size=11)

                            axs[0].set_title('%s state' % params['state_names'][0], size=11)
                            axs[1].set_title('%s state' % params['state_names'][1], size=11)
                            axs[2].set_title('Across states', size=11)

                            # colorbar
                            cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                            cbar = fig.colorbar(im2, cbar_ax)
                            cbar.set_label('Weight correlation', size=11)

                            fig.suptitle('{subject} {y_target} decoding using {feature_used}'.format(subject=subject,
                                                                                                     y_target=y_target,
                                                                                                     feature_used=feature_used),
                                         y=1.04)
                            fig_name = '{subject}_{y_target}_weights_sim_using_{feature_used}'.format(subject=subject,
                                                                                                   y_target=y_target,
                                                                                                   feature_used=feature_used)
                            splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

                        # Plot cross-window decoding accuracy
                        if 'across_window_matrix_1' in decoding_result.files:
                            with plt.style.context(splstyle.get_style('nature-reviews')):
                                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                                fig.set_size_inches(6, 3)

                                time_start = decoding_end_bins[0]
                                time_end = decoding_end_bins[-1]
                                cmap = 'bwr'
                                axs[0].imshow(decoding_result['across_window_matrix_1'], vmin=0, vmax=1, cmap=cmap,
                                            extent=[time_start, time_end, time_end, time_start])
                                im1 = axs[1].imshow(decoding_result['across_window_matrix_2'], vmin=0, vmax=1, cmap=cmap,
                                            extent=[time_start, time_end, time_end, time_start])
                                

                                line_times = params['vert_line_times']

                                for l_t in line_times:
                                    [ax.axvline(l_t, linestyle='--', color='black', lw=1) for ax in axs]
                                    [ax.axhline(l_t, linestyle='--', color='black', lw=1) for ax in axs]

                                axs[0].set_xlabel('Time from trial start (s)', size=11)
                                axs[0].set_ylabel('Time from trial start (s)', size=11)

                                axs[0].set_title('%s state' % params['state_names'][0], size=11)
                                axs[1].set_title('%s state' % params['state_names'][1], size=11)
                                # axs[2].set_title('Across states', size=11)

                                # colorbar
                                cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                                cbar = fig.colorbar(im1, cbar_ax)
                                cbar.set_label('Decoding accuracy', size=11)

                                fig.suptitle('{subject} {y_target} decoding using {feature_used}'.format(subject=subject,
                                                                                                         y_target=y_target,
                                                                                            feature_used=feature_used),
                                            y=1.04)
                                fig_name = '{subject}_{y_target}_cross_window_decoding_using_{feature_used}'.format(subject=subject,
                                                                                                    y_target=y_target,
                                                                                                    feature_used=feature_used)
                                splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))


                    if params['plot_ave_across_mice']:

                        if len(accuracy_per_window_1_store_list) == 1: 
                            accuracy_per_window_1_store = np.array(accuracy_per_window_1_store_list)
                            accuracy_per_window_2_store = np.array(accuracy_per_window_2_store_list)
                            accuracy_per_window_1_null_store = np.array(accuracy_per_window_1_null_store_list)
                            accuracy_per_window_2_null_store = np.array(accuracy_per_window_2_null_store_list)
                        else:
                            accuracy_per_window_1_store = np.stack(accuracy_per_window_1_store_list)
                            accuracy_per_window_2_store = np.stack(accuracy_per_window_2_store_list)
                            accuracy_per_window_1_null_store = np.stack(accuracy_per_window_1_null_store_list)
                            accuracy_per_window_2_null_store = np.stack(accuracy_per_window_2_null_store_list)

                        
                        num_windows = np.shape(accuracy_per_window_1_store)[1] 
                        engaged_vs_disengaged_p_vals = np.zeros((num_windows, ))
                        engaged_vs_null_p_vals = np.zeros((num_windows, ))
                        disengaged_vs_null_p_vals = np.zeros((num_windows, ))                        
                        
                        for window_idx in np.arange(num_windows):
                            _, engaged_vs_disengaged_p_vals[window_idx] = sstats.wilcoxon(
                                accuracy_per_window_1_store[:, window_idx], 
                                accuracy_per_window_2_store[:, window_idx], 
                            )

                            _, engaged_vs_null_p_vals[window_idx] = sstats.wilcoxon(
                                accuracy_per_window_1_store[:, window_idx], 
                                accuracy_per_window_1_null_store[:, window_idx], 
                            )

                            _, disengaged_vs_null_p_vals[window_idx] = sstats.wilcoxon(
                                accuracy_per_window_2_store[:, window_idx], 
                                accuracy_per_window_2_null_store[:, window_idx], 
                            )

                        p_val_threshold = 0.05

                        engaged_vs_disengaged_sig = (engaged_vs_disengaged_p_vals < p_val_threshold).astype(float)
                        engaged_vs_null_sig = (engaged_vs_null_p_vals < p_val_threshold).astype(float)
                        disengaged_vs_null_sig = (disengaged_vs_null_p_vals < p_val_threshold).astype(float)

                        # if both engaged and disengaged are not above null, then significant difference is ignored
                        engaged_vs_disengaged_sig[(1 - engaged_vs_null_sig).astype(bool) & (1 - disengaged_vs_null_sig).astype(bool)] = np.nan

                        # subset only values within a minimum window count 
                        # the convolution slieds a window of length k and count how many 1s are in that window
                        min_window_num = 5

                        if min_window_num > 1:
                            engaged_vs_disengaged_sig_cov = np.convolve(engaged_vs_disengaged_sig, np.ones(min_window_num, dtype=int), 'same')
                            engaged_vs_null_sig_cov = np.convolve(engaged_vs_null_sig, np.ones(min_window_num, dtype=int), 'same')
                            disengaged_vs_null_sig_cov = np.convolve(disengaged_vs_null_sig, np.ones(min_window_num, dtype=int), 'same')

                            engaged_vs_disengaged_sig = np.where(engaged_vs_disengaged_sig_cov >= min_window_num, 1.0, 0.0)
                            engaged_vs_null_sig = np.where(engaged_vs_null_sig_cov >= min_window_num, 1.0, 0.0)
                            disengaged_vs_null_sig = np.where(disengaged_vs_null_sig_cov >= min_window_num, 1.0, 0.0)

                        engaged_vs_disengaged_sig[engaged_vs_disengaged_sig == 0] = np.nan 
                        engaged_vs_null_sig[engaged_vs_null_sig == 0] = np.nan 
                        disengaged_vs_null_sig[disengaged_vs_null_sig == 0] = np.nan
                        

                        num_mice = np.shape(accuracy_per_window_1_store)[0]

                        if decoding_result['decoding_bin_width'] is not None:
                            decoding_end_bins = decoding_result['decoding_time_bins'] + decoding_result['decoding_bin_width']
                        else:
                            decoding_end_bins = decoding_result['decoding_time_bins']

                        fig, axs = plot_windowed_decoding_accuracy_mult_states(decoding_end_bins,
                                                                    np.mean(accuracy_per_window_1_store, axis=0),
                                                                    decoding_result['accuracy_per_window_null_1'],
                                                                    np.mean(accuracy_per_window_2_store, axis=0),
                                                                    decoding_result['accuracy_per_window_null_2'],
                                                                    accuracy_1_spread=np.std(accuracy_per_window_1_store, axis=0)/np.sqrt(num_mice),
                                                                    accuracy_2_spread=np.std(accuracy_per_window_2_store, axis=0)/np.sqrt(num_mice),
                                                                    choice_reward_matrix_1=None,
                                                                    choice_reward_matrix_2=None,
                                                                    state_names=params['state_names'], xlabel=params['xlabel'],
                                                                    ylabel=params['ylabel_dict'][y_target],
                                                                    row_names=params['row_names'],
                                                                    col_names=params['col_names'],
                                                                    vert_line_times=params['vert_line_times'],
                                                                    vert_line_labels=params['vert_line_labels'],
                                                                    vert_line_styles=params['vert_line_styles'],
                                                                    fig=None, axs=None, linecolor=[params['engaged_state_color'], params['disengaged_state_color']],
                                                                    custom_xticks=params['xtick_locs'],
                                                                    custom_yticks=params['ytick_locs'][y_target],
                                                                    custom_ylim=params['ylimits'][y_target], 
                                                                    include_scatter=False,
                                                                    single_plot=True)
                        
                        # Plot significance lines 
                        axs.plot(decoding_end_bins[1:], 0.015 + params['ylimits'][y_target][1] * engaged_vs_null_sig, lw=3, color=params['engaged_state_color'], clip_on=False)
                        axs.plot(decoding_end_bins[1:], 0.025 + params['ylimits'][y_target][1] * disengaged_vs_null_sig, lw=3, color=params['disengaged_state_color'], clip_on=False)
                        axs.plot(decoding_end_bins[1:], 0.035 + params['ylimits'][y_target][1] * engaged_vs_disengaged_sig, lw=3, color='black', clip_on=False)


                        fig.text(1, 0.1, r'$n = %.f$ mice' % num_mice)
                        fig.suptitle('all subjects decoding using {feature_used}'.format(subject=subject, feature_used=feature_used),
                                     y=1.04)
                        fig_name = 'all_subjects_{y_target}_decoding_using_{feature_used}'.format(y_target=y_target, feature_used=feature_used)
                        splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

                        # Save an extra copy with the associated figure name in the paper
                        if fig_name in FIGURE_NAMES.keys():
                            splutil.savefig(fig, os.path.join(FIGURES_DIR, FIGURE_NAMES[fig_name]))

                        # Plot weight similarity across time
                        state_1_corr_mean = np.mean(np.stack(state_1_weights_corr_list), axis=0)
                        state_2_corr_mean = np.mean(np.stack(state_1_weights_corr_list), axis=0)
                        across_state_corr_mean = np.mean(np.stack(across_state_corr_list), axis=0)

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                            fig.set_size_inches(9, 3)

                            time_start = decoding_end_bins[0]
                            time_end = decoding_end_bins[-1]

                            axs[0].imshow(state_1_corr_mean, vmin=-1, vmax=1, cmap='bwr',
                                          extent=[time_start, time_end, time_end, time_start])
                            axs[1].imshow(state_2_corr_mean, vmin=-1, vmax=1, cmap='bwr',
                                          extent=[time_start, time_end, time_end, time_start])

                            im2 = axs[2].imshow(across_state_corr_mean, vmin=-1, vmax=1, cmap='bwr',
                                                extent=[time_start, time_end, time_end, time_start])

                            line_times = [0, 1, 2]

                            for l_t in line_times:
                                [ax.axvline(l_t, linestyle='--', color='black', lw=1) for ax in axs]
                                [ax.axhline(l_t, linestyle='--', color='black', lw=1) for ax in axs]

                            axs[0].set_xlabel('Time from trial start (s)', size=11)
                            axs[0].set_ylabel('Time from trial start (s)', size=11)

                            axs[0].set_title('Engaged state', size=11)
                            axs[1].set_title('Disengaged state', size=11)
                            axs[2].set_title('Across states', size=11)

                            # colorbar
                            cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                            cbar = fig.colorbar(im2, cbar_ax)
                            cbar.set_label('Weight correlation', size=11)

                            fig.suptitle('all subjects  decoding using {feature_used}'.format(
                                                                                          feature_used=feature_used),
                                         y=1.14)
                            fig_name = 'all_subjects_{y_target}_weights_sim_using_{feature_used}'.format(
                                                                                                      y_target=y_target,
                                                                                                      feature_used=feature_used)
                            splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

                            # Save an extra copy with the associated figure name in the paper
                            if fig_name in FIGURE_NAMES.keys():
                                splutil.savefig(fig, os.path.join(FIGURES_DIR, FIGURE_NAMES[fig_name]))
                        
                        # Plot cross window decoding: average across mice
                        if len(across_window_matrix_1_list) > 0:
                            across_window_matrix_1_mean = np.mean(np.stack(across_window_matrix_1_list), axis=0)
                            across_window_matrix_2_mean = np.mean(np.stack(across_window_matrix_2_list), axis=0)
                            with plt.style.context(splstyle.get_style('nature-reviews')):
                                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                                fig.set_size_inches(6, 3)

                                cmap = 'bwr'

                                time_start = decoding_end_bins[0]
                                time_end = decoding_end_bins[-1]
                                axs[0].imshow(across_window_matrix_1_mean, vmin=0, vmax=1, cmap=cmap,
                                            extent=[time_start, time_end, time_end, time_start])
                                im1 = axs[1].imshow(across_window_matrix_2_mean, vmin=0, vmax=1, cmap=cmap,
                                            extent=[time_start, time_end, time_end, time_start])

                                line_times = params['vert_line_times']

                                for l_t in line_times:
                                    [ax.axvline(l_t, linestyle='--', color='black', lw=1) for ax in axs]
                                    [ax.axhline(l_t, linestyle='--', color='black', lw=1) for ax in axs]

                                axs[0].set_xlabel('Time from trial start (s)', size=11)
                                axs[0].set_ylabel('Time from trial start (s)', size=11)

                                axs[0].set_title('%s state' % params['state_names'][0], size=11)
                                axs[1].set_title('%s state' % params['state_names'][1], size=11)
                                # axs[2].set_title('Across states', size=11)

                                # colorbar
                                cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                                cbar = fig.colorbar(im1, cbar_ax)
                                cbar.set_label('Decoding accuracy', size=11)

                                fig.suptitle('all_subjects_{y_target} decoding using {feature_used}'.format(
                                                                                                            y_target=y_target,
                                                                                            feature_used=feature_used),
                                            y=1.04)
                                fig_name = 'all_subjects_{y_target}_cross_window_decoding_using_{feature_used}'.format(
                                                                                                    y_target=y_target,
                                                                                                    feature_used=feature_used)
                                splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

                                # Save an extra copy with the associated figure name in the paper
                                if fig_name in FIGURE_NAMES.keys():
                                    splutil.savefig(fig, os.path.join(FIGURES_DIR, FIGURE_NAMES[fig_name]))

        elif plots_tm == 'riskychoice_single_region_decoding':

            
            fig, axs = plot_riskychoice_single_vs_all_region_decoding(model_name=params['model_name'], 
                                                                      include_brainmap=params['include_brainmap'], 
                                                                      fig=None, axs=None)
            fig_name = 'riskychoice_single_region_decoding'
            splutil.savefig(fig, os.path.join(FIGURES_DIR, fig_name))

            # Save an extra copy with the associated figure name in the paper
            if fig_name in FIGURE_NAMES.keys():
                splutil.savefig(fig, os.path.join(FIGURES_DIR, FIGURE_NAMES[fig_name]))
        
    logger.success("Plot generation complete.")


    # -----------------------------------------


def plot_riskychoice_single_vs_all_region_decoding(model_name='20250715', include_brainmap=True, fig=None, axs=None):
    """Plots single versus multiple region decoding performance

    Parameters
    ----------
    model_name : str, optional
        name of the model that you want the results to be plotted, by default '20250715'
    include_brainmap : bool, optional
        whether to include the map of the brain with colors corresponding to decoding result lines, by default True
    fig : matplotlib figure object, optional
        matplotlib figure object, by default None
    axs : matplotlib axes object, optional
        matplotlib axes object, by default None

    Returns
    -------
    fig : matplotlib figure object
    axs : matplotlib axes object
    """

    decoding_targets = ['LotteryChoice', 'AirPuffLeft', 'LeftChoice']
    decoding_titles = ['Lottery', 'Airpuff', 'Choice']
    subject_to_exclude = 'IAA-1125879'


    # Make brain region colors 
    # Solarized accent colors (hex)
    solarized_colors = [
        '#b58900',  # yellow
        '#cb4b16',  # orange
        '#dc322f',  # red
        '#d33682',  # magenta
        '#6c71c4',  # violet
        '#268bd2',  # blue
        '#2aa198',  # cyan
        '#859900',  # green
    ]

    


    # Do plot

    with plt.style.context(splstyle.get_style('nature-reviews')):

        fig, axs = plt.subplots(2, len(decoding_targets), sharex=True, sharey=True)
        fig.set_size_inches(9, 6)

        for d_i, d_target in enumerate(decoding_targets): 
            decoding_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, 'decoding', '*{d_target}_using_pixel_ave_{model_name}.npz'.format(d_target=d_target, model_name=model_name)))

            if len(decoding_files) == 0:
                decoding_files = glob.glob(os.path.join(PROCESSED_DATA_DIR,
                                                        '*{d_target}_using_pixel_ave_{model_name}.npz'.format(
                                                            d_target=d_target, model_name=model_name)))

            decoding_files = [np.load(x)  for x in decoding_files if subject_to_exclude not in x]
            engaged_accuracy_combined = np.stack([x['accuracy_per_window_1'] for x in decoding_files])
            engaged_accuracy_combined_mean = np.mean(engaged_accuracy_combined, axis=0).T

            disengaged_accuracy_combined = np.stack([x['accuracy_per_window_2'] for x in decoding_files])
            disengaged_accuracy_combined_mean = np.mean(disengaged_accuracy_combined, axis=0).T

            if d_i == 0:
                decoding_end_bins = decoding_files[0]['decoding_time_bins'][1:] + decoding_files[0]['decoding_bin_width']
                num_brain_regions = np.shape(engaged_accuracy_combined)[1] - 1
                # Create a continuous colormap by interpolating
                import matplotlib.colors as mcolors
                solarized_cmap = mcolors.LinearSegmentedColormap.from_list("solarized_continuous", solarized_colors)
                brain_region_colors = [solarized_cmap(x)[:-1] for x in np.arange(num_brain_regions) / num_brain_regions]

            for _brain_region_idx in np.arange(num_brain_regions): 

                axs[0, d_i].plot(decoding_end_bins, engaged_accuracy_combined_mean[:, _brain_region_idx], alpha=0.3, lw=1, 
                                 color=brain_region_colors[_brain_region_idx])
                axs[0, d_i].plot(decoding_end_bins, engaged_accuracy_combined_mean[:, -1], alpha=0.3, lw=1, color='black')

                axs[1, d_i].plot(decoding_end_bins, disengaged_accuracy_combined_mean[:, _brain_region_idx], alpha=0.3, lw=1, 
                                 color=brain_region_colors[_brain_region_idx])
                axs[1, d_i].plot(decoding_end_bins, disengaged_accuracy_combined_mean[:, -1], alpha=0.3, lw=1, color='black')

        for _title_idx in np.arange(len(decoding_titles)): 
            axs[0, _title_idx].set_title(decoding_titles[_title_idx], size=11)
        


        axs[0, 0].text(-1, 1, 'Engaged', size=11)
        axs[1, 0].text(-1, 1, 'Disengaged', size=11)

        axs[1, 0].set_xlabel('Time from sound onset (s)', size=11, color='gray')
        axs[1, 0].set_ylabel('Decoding accuracy', size=11, color='gray')

        for _y_loc in np.arange(0.5, 0.91, 0.1):
            axs[0, 0].axhline(_y_loc, xmax=1.2, clip_on=False, linestyle='--', color='gray', lw=0.5)
            axs[0, 1].axhline(_y_loc, xmax=1.2, clip_on=False, linestyle='--', color='gray', lw=0.5)
            axs[0, 2].axhline(_y_loc, xmax=1, clip_on=False, linestyle='--', color='gray', lw=0.5)

            axs[1, 0].axhline(_y_loc, xmax=1.2, clip_on=False, linestyle='--', color='gray', lw=0.5)
            axs[1, 1].axhline(_y_loc, xmax=1.2, clip_on=False, linestyle='--', color='gray', lw=0.5)
            axs[1, 2].axhline(_y_loc, xmax=1, clip_on=False, linestyle='--', color='gray', lw=0.5)

        [ax.set_xticks([0, 1, 2, 3, 4]) for ax in axs.flatten()]
        [ax.tick_params(axis='both', length=0., labelcolor='gray') for ax in axs.flatten()]

        [ax.spines['left'].set_visible(False) for ax in axs.flatten()]
        [ax.spines['bottom'].set_visible(False) for ax in axs.flatten()]

        for _x_loc in [0, 1, 2]: 
            [ax.axvline(_x_loc, linestyle='--', color='gray', zorder=-1, lw=0.5) for ax in axs[0, :]]
            [ax.axvline(_x_loc, linestyle='--', color='gray', zorder=-1, lw=0.5, clip_on=False, ymax=1.2) for ax in axs[1, :]]

        if include_brainmap:
            try:
                big_roi_mask = spio.loadmat(os.path.join(INTERIM_DATA_DIR, 'allenDorsalMap.mat'))['dorsalMaps']
            except:
                big_roi_mask = spio.loadmat(os.path.join(ALLEN_DATA_DIR, 'allenDorsalMap.mat'))['dorsalMaps']

            edge_outline = big_roi_mask[0, 0]['edgeOutline'].flatten()

            # from matplotlib.colors import ListedColormap

            brainmap_ax = fig.add_axes([-0.2, 0.35, 0.3, 0.3])
            # add brain map with labels
            for b_idx in np.arange(len(brain_region_colors)): 

                alpha = 1
                b_color = list(brain_region_colors[b_idx]) + [alpha]
                # cmap = ListedColormap(["none", b_color])
                # _ax.imshow(roi_mask[:, :, b_idx], cmap=cmap, interpolation='none')
                brainmap_ax.fill(edge_outline[b_idx][:, 1], -edge_outline[b_idx][:, 0], 
                                 color=brain_region_colors[b_idx], lw=1.75)

            brainmap_ax.set_xticks([]) 
            brainmap_ax.set_yticks([])
            brainmap_ax.spines['left'].set_visible(False)
            brainmap_ax.spines['bottom'].set_visible(False)
            brainmap_ax.set_aspect('equal')

    return fig, axs

def plot_windowed_decoding_accuracy_mult_states(decoding_time_bins,
                                                accuracy_per_window_1, accuracy_per_window_null_1,
                                                accuracy_per_window_2, accuracy_per_window_null_2,
                                                choice_reward_matrix_1=None, choice_reward_matrix_2=None,
                                                accuracy_1_spread=None, accuracy_2_spread=None,
                                                state_names=['Stochastic state', 'Other states'],
                                                ylabel='Reward decoding accuracy',
                                                xlabel='Time from first lick (s)',
                                                row_names=['L', 'R'],
                                                col_names=['Urw', 'Rw'],
                                                vert_line_times=None,
                                                vert_line_labels=None,
                                                vert_line_styles=['--', ':'],
                                                include_scatter=True, single_plot=False,
                                                fig=None, axs=None, linecolor=['black', 'red'], 
                                                plot_null=True, include_legend=True,
                                                custom_xticks=None, custom_yticks=None, custom_ylim=None):
    """Plots the decoding accuracy per window, for 2 different states (engaged / disengaged states)

    Parameters
    ----------
    decoding_time_bins : numpy ndarray
        1D array corresponding the the time point of each decoding time bin
    accuracy_per_window_1 : numpy ndarray
        _description_
    accuracy_per_window_null_1 : numpy ndarray
        _description_
    accuracy_per_window_2 : numpy ndarray
        _description_
    accuracy_per_window_null_2 : numpy ndarray
        _description_
    choice_reward_matrix_1 : numpy ndarray, optional
        _description_, by default None
    choice_reward_matrix_2 : numpy ndarray, optional
        _description_, by default None
    accuracy_1_spread : numpy ndarray, optional
        some quantification of the distribution of accuracy values for state 1, eg. the standard error of the mean, 
        by default None
    accuracy_2_spread : numpy ndarray, optional
        some quantification of the distribution of accuracy values for state 2, eg. the standard error of the mean, 
        by default No
    state_names : list, optional
        names corresponding to state 1 and state 2, by default ['Stochastic state', 'Other states']
    ylabel : str, optional
        label on the y-axis, by default 'Reward decoding accuracy'
    xlabel : str, optional
        label on the x-axis, by default 'Time from first lick (s)'
    row_names : list, optional
        row names for the contingency table, by default ['L', 'R']
    col_names : list, optional
        column names for the contigency table, by default ['Urw', 'Rw']
    vert_line_times : list, optional
        time (s) to plot vertical lines to indicate time of events, by default None
    vert_line_labels : list, optional
        labels corresponding to each vertical line specified in vert_line_times, by default None
    vert_line_styles : list, optional
        line style corresponding to each vertical line specified in ver_line_times, by default ['--', ':']
    include_scatter : bool, optional
        whether to include scatter dots on top of the decoding performance line, by default True
    single_plot : bool, optional
        whether to plot the accuracy of the two states in a single plot, by default False
    fig : matplotlib figure object, optional
        by default None, which means one will be generated
    axs : matplotlib axes object, optional
        by default None, which means one will be generated
    linecolor : list, optional
        state 1 and 2 line color and shading color, by default ['black', 'red']
    Returns
    -------
    fig : matplotlib figure object
    axs : matplotlib axes object
    """

    with plt.style.context(splstyle.get_style('nature-reviews')):
        if (fig is None) and (axs is None):
            if single_plot:
                fig, axs = plt.subplots()
                fig.set_size_inches(3, 3)
            else:
                fig, axs = plt.subplots(1, 2, sharey=True)
                fig.set_size_inches(6, 3)

        if choice_reward_matrix_1 is not None:
            axs[0].set_title('{state_name} \n Trials: {num_trials}'.format(state_name=state_names[0],
                                                                               num_trials=int(np.nansum(choice_reward_matrix_1))), size=11)
            axs[1].set_title('{state_name} \n Trials: {num_trials}'.format(state_name=state_names[1],
                                                                           num_trials=int(np.nansum(choice_reward_matrix_2))),
                             size=11)

        if single_plot:
            axs.set_ylim([0.4, 1])
            axs.plot(decoding_time_bins[1:], accuracy_per_window_1, color=linecolor[0], label=state_names[0], zorder=3)

            if plot_null:
                axs.plot(decoding_time_bins[1:], np.mean(accuracy_per_window_null_1, axis=1), color='gray', zorder=1)
            axs.plot(decoding_time_bins[1:], accuracy_per_window_2, color=linecolor[1], label=state_names[1], zorder=2)

            if accuracy_1_spread is not None:
                axs.fill_between(decoding_time_bins[1:], accuracy_per_window_1 - accuracy_1_spread,
                                 accuracy_per_window_1 + accuracy_1_spread, alpha=0.5, color=linecolor[0], lw=0,
                                 zorder=3)
            if accuracy_2_spread is not None:
                axs.fill_between(decoding_time_bins[1:], accuracy_per_window_2 - accuracy_2_spread,
                                 accuracy_per_window_2 + accuracy_2_spread, alpha=0.5, color=linecolor[1], lw=0,
                                 zorder=2)

        else:
            axs[0].set_ylim([0.4, 1])
            axs[0].plot(decoding_time_bins[1:], accuracy_per_window_1, color=linecolor[0])
            axs[1].plot(decoding_time_bins[1:], accuracy_per_window_2, color=linecolor[0])

            if plot_null:
                axs[0].plot(decoding_time_bins[1:], np.mean(accuracy_per_window_null_1, axis=1), color='gray')
                axs[1].plot(decoding_time_bins[1:], np.mean(accuracy_per_window_null_2, axis=1), color='gray')

        for axis_index, choice_reward_matrix in zip([0, 1], [choice_reward_matrix_1, choice_reward_matrix_2]):

            if choice_reward_matrix is not None:
                ax_inset = axs[axis_index].inset_axes([0.3, 0.7, 0.3, 0.3], zorder=-3)
                _, ax_inset = plot_contigency_table(choice_reward_matrix, row_names, col_names, colors=None,
                                           fig=None, ax=ax_inset)

        if vert_line_times is not None:
            if single_plot:
                for vline_idx, vline_t in enumerate(vert_line_times):
                    axs.axvline(vline_t, linestyle=vert_line_styles[vline_idx],
                                   label=vert_line_labels[vline_idx], lw=1, color='gray', zorder=0)
                if include_legend:
                    axs.legend(bbox_to_anchor=(1.04, 0.5))
            else:
                for vline_idx, vline_t in enumerate(vert_line_times):
                    [ax.axvline(vline_t, linestyle=vert_line_styles[vline_idx],
                                   label=vert_line_labels[vline_idx], lw=1, color='gray', zorder=0) for ax in axs.flatten()]
                if include_legend:
                    axs[0].legend()

        if include_scatter:
            axs[0].scatter(decoding_time_bins[1:], accuracy_per_window_1, color=linecolor[0])
            axs[0].scatter(decoding_time_bins[1:], np.mean(accuracy_per_window_null_1, axis=1), color='gray')
            axs[1].scatter(decoding_time_bins[1:], accuracy_per_window_2, color=linecolor[1])
            axs[1].scatter(decoding_time_bins[1:], np.mean(accuracy_per_window_null_2, axis=1), color='gray')

        if single_plot:
            axs.set_ylabel(ylabel, size=11)
            axs.set_xlabel(xlabel, size=11)
        else:
            axs[0].set_ylabel(ylabel, size=11)
            axs[0].set_xlabel(xlabel, size=11)
        
        if single_plot:
            if custom_xticks is not None:
                axs.set_xticks(custom_xticks)
            if custom_ylim is not None:
                axs.set_ylim(custom_ylim)
            if custom_yticks is not None:
                axs.set_yticks(custom_yticks)


    return fig, axs


def plot_contigency_table(contigency_table, row_names, column_names, colors=None,
                          fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()

    num_rows = np.shape(contigency_table)[0]
    num_columns = np.shape(contigency_table)[1]

    counter = 0
    for row_i in np.arange(num_rows):
        for column_j in np.arange(num_columns):
            if colors is not None:
                color = colors[counter]
                counter = counter + 1
            else:
                color = 'black'

            if ~np.isnan(contigency_table[column_j, row_i]):
                entry_val = int(contigency_table[column_j, row_i])
            else:
                entry_val = np.nan

            ax.text(row_i, column_j, entry_val, ha='center', va='center',
                    color=color)

    ax.set_xticks(np.arange(num_columns))
    ax.set_yticks(np.arange(num_rows))
    ax.set_xticklabels(column_names)
    ax.set_yticklabels(row_names)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])

    return fig, ax

def correlate_feature_matrices(A, B):
    num_features, num_timepoints = np.shape(A)

    corr_matrix = np.zeros((num_timepoints, num_timepoints))

    for t_i in np.arange(num_timepoints):
        for t_j in np.arange(num_timepoints):
            corr_val = np.corrcoef(A[:, t_i], B[:, t_j])
            corr_matrix[t_i, t_j] = corr_val[0, 1]

    return corr_matrix


if __name__ == "__main__":
    app()


