import collections
import pdb
from pathlib import Path
import yaml
import inspect
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
import os
import numpy as np
import dataset as mp_dataset
from config import PROCESSED_DATA_DIR, TRAIN_YAML, INTERIM_DATA_DIR
import util as mp_util

import warnings

# Parallelisation and checking it is faster
import joblib
import time

# Decoding
from sklearn import svm
from sklearn.base import clone
import sklearn.model_selection as sklselection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sklpreprocessing
import sklearn.svm as sklsvm
import sklearn.linear_model as sklinear
# import shap # SHapley Additive exPlanations

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    process_name: str = 'window-decoding'
    # -----------------------------------------
):
    # ---- Reading in parameters from YAML file ----
    logger.info("Training some model...")

    params = mp_util.read_params(TRAIN_YAML, process_name=process_name)
    # params['RAW_DATA_DIR'] = str(RAW_DATA_DIR)
    params['INTERIM_DATA_DIR'] = str(INTERIM_DATA_DIR)
    params['PROCESSED_DATA_DIR'] = str(PROCESSED_DATA_DIR)
    mp_util.print_params(params)

    dataset_to_fit = params['dataset_to_fit']  # 'mouse' or 'monkey'
    model_type = params['model_type'] # psytrack or hmmglm or locaNMFdecoder
    model_id = params['process_name']

    if dataset_to_fit == 'mouse':

        subjects_to_fit = params['subjects_to_fit']
        target_outputs = 'chooseLR'
        datatype = 'mouse'

    for subject in subjects_to_fit:

        if dataset_to_fit == 'mouse':

            if model_type == 'WFdecoder':
                include_miss_trials = False
                protocol = params['protocol']
                if type(protocol) is not list:
                    protocol = [protocol]

            if 'RiskyChoice' in protocol:
                # data_type = 'riskychoice-decoding-behaviour_df'
                protocol = None  # set to None because I don't apply a filter for that data

                if params['activity_to_use'] == 'pixel_ave':
                    data_type = 'riskychoice-decoding'
                elif params['activity_to_use'] == 'mot_svd':
                    data_type = 'riskychoice-decoding-from-face'
                elif params['activity_to_use'] == 'sleap':
                    data_type = 'riskychoice-decoding-from-sleap'
                else:
                    print('No valid data type specified')
                    pdb.set_trace()
                    
                custom_folder_fstring = None
                try:
                    locanmf_data = mp_dataset.load_data(data_type=data_type, subject=subject,
                                               include_miss_trials=include_miss_trials,
                                               protocol=protocol)
                except:
                    print('Cannot load riskychoice data for some reason, check data source is correct')
                    pdb.set_trace()

            # This loads the behaviour
            mp_data = mp_dataset.load_data(data_type=data_type, subject=subject,
                                           include_miss_trials=include_miss_trials,
                                           protocol=protocol)

            datatype = 'mouse'


        logger.info("Fitting model to %s..." % subject)
        if model_type == 'WFdecoder':

            if protocol is not None:

                if type(protocol) is list:
                    mp_data = mp_data.loc[
                        mp_data['protocol'].isin(protocol)
                    ]
                else:
                    mp_data = mp_data.loc[
                        mp_data['protocol'] == protocol
                    ]

            save_folder = os.path.join(PROCESSED_DATA_DIR, 'decoding')
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

            if 'combine_all_sessions' in params.keys():
                if params['combine_all_sessions']:
                    mp_data['date'] = 'all'

            for date in np.unique(mp_data['date']):
                y_targets = params['y_targets']
                if type(y_targets) is not list:
                    y_targets = [y_targets]

                for y_target in y_targets:

                    if params['model_name'] is not None:
                        save_name = '{subject}_{date}_{protocol}_locaNMF_decoding_{y_target}_using_{activity}_{model_name}.npz'.format(
                            subject=subject, date=date, protocol=protocol, y_target=y_target, activity=params['activity_to_use'],
                            model_name=params['model_name'],
                        )
                    else:
                        save_name = '{subject}_{date}_{protocol}_locaNMF_decoding_{y_target}_using_{activity}_resample_states_{resample_states}_balance_states_trialcounts_{balance_states_trialcounts}.npz'.format(
                            subject=subject, date=date, protocol=protocol, y_target=y_target, activity=params['activity_to_use'],
                            resample_states=params['resample_states'], balance_states_trialcounts=params['balance_states_trialcounts'],
                        )
                    if not os.path.exists(save_name):
                        logger.info('Doing decoding for {subject} {date}'.format(subject=subject, date=date))


                    decoder_result = fit_locanmf_decoder(subject=subject,
                                                         date=date, protocol=protocol,
                                                         locaNMF_data=locanmf_data,
                                                         balancing_method=params['balancing_method'],
                                                         activity_to_use=params['activity_to_use'],
                                                         balance_targets=params['balance_targets'],
                                                         states_to_use=params['states_to_use'],
                                                         weight_type=params['weight_types'],
                                                         y_target=y_target,
                                                         resample_states=params['resample_states'],
                                                         run_in_parallel=params['run_in_parallel'],
                                                         decoding_time_start=params['decoding_time_start'],
                                                         decoding_time_end=params['decoding_time_end'],
                                                         decoding_time_step=params['decoding_time_step'],
                                                         decoding_bin_width=params['decoding_bin_width'],
                                                         num_cv_repeats=params['num_cv_repeats'],
                                                         balance_states_trialcounts=params['balance_states_trialcounts'],
                                                         fit_each_feature=params['fit_each_feature'], 
                                                         do_across_window_decoding=params['do_across_window_decoding'],
                                                         state_p_threshold=params['state_p_threshold'])

                    if decoder_result is not None:
                        decoder_result_save_path = os.path.join(save_folder, save_name)
                        np.savez(decoder_result_save_path, **decoder_result)


        logger.success("Modeling training complete.")


def add_behaviour_features(behaviour_df, feature_names=['prevChoice'],
                           state_p_threshold=0.8):

    for f_name in feature_names:

        if f_name == 'prevChoice':
            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                date_df['prevChoice'] = np.concatenate([np.array([np.nan]), date_df['choice'].values[0:-1]])
                df_store.append(date_df)
            behaviour_df = pd.concat(df_store)
        elif f_name == 'prevReward':
            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                date_df['prevReward'] = np.concatenate([np.array([np.nan]), date_df['reward'].values[0:-1]])
                df_store.append(date_df)
            behaviour_df = pd.concat(df_store)
        elif f_name == 'prevChoiceR':
            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                prevChoice_R = (date_df['choice'].values[0:-1] == 'R')
                date_df['prevChoiceR'] = np.concatenate([np.array([np.nan]), prevChoice_R])
                df_store.append(date_df)
            behaviour_df = pd.concat(df_store)
        elif f_name == 'prevReward_X_prevChoiceR':
            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                prevChoice_R = (date_df['choice'].values[0:-1] == 'R')
                prevReward = date_df['reward'].values[0:-1].copy()
                prevChoice_R[prevChoice_R == 0] = -1
                prevReward[prevReward == 0] =  -1
                date_df['prevReward_X_prevChoiceR'] = np.concatenate([np.array([np.nan]), prevChoice_R * prevReward])
                df_store.append(date_df)
            behaviour_df = pd.concat(df_store)
        elif f_name == 'session_progress':
            session_progress_list = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[behaviour_df['date'] == date].copy()
                session_progress = np.arange(1, len(date_df) + 1) / len(date_df)
                session_progress_list.extend(session_progress)
            behaviour_df['session_progress'] = session_progress_list

        elif f_name == 'switch':

            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                switch = date_df['choice'].values[1:] != date_df['choice'].values[0:-1]
                # put NaN if previous choice is M
                prev_choice_M = date_df['choice'].values[0:-1] == 'M'
                switch[prev_choice_M] = np.nan
                date_df['switch'] = np.concatenate([np.array([np.nan]), switch])
                df_store.append(date_df)
            behaviour_df = pd.concat(df_store)

        elif f_name == 'prevSwitch':

            # TODO: this is not coded up yet
            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                switch = date_df['choice'].values[1:] != date_df['choice'].values[0:-1]
                # put NaN if previous choice is M
                prev_choice_M = date_df['choice'].values[0:-1] == 'M'
                switch[prev_choice_M] = np.nan
                date_df['switch'] = np.concatenate([np.array([np.nan]), switch])
                df_store.append(date_df)

            behaviour_df = pd.concat(df_store)

        elif f_name == 'seq_length':

            def count_consecutive(sequence):
                counts = []
                current_count = 1
                counts.append(current_count)

                for i in range(1, len(sequence)):
                    if sequence[i] == sequence[i - 1]:
                        current_count += 1
                    else:
                        current_count = 1
                    counts.append(current_count)

                return counts

            df_store = []
            for date in np.unique(behaviour_df['date']):
                date_df = behaviour_df.loc[
                    behaviour_df['date'] == date
                    ].copy()
                seq_length = count_consecutive(date_df['choice'].values)
                date_df['seq_length'] = np.concatenate([np.array([np.nan]), seq_length[0:-1]])
                df_store.append(date_df)

            behaviour_df = pd.concat(df_store)
        
        elif f_name == 'biasState':
            behaviour_df['biasState'] = \
                ((behaviour_df['p_leftbias'].values >= state_p_threshold) |
                  (behaviour_df['p_rightbias'].values >= state_p_threshold))

        elif f_name == 'stochasticState':
            
            behaviour_df['stochasticState'] = (behaviour_df['p_stochastic'].values >= state_p_threshold)


    return behaviour_df


def make_X_y(y_target='reward', X_features=['locaNMFcomponents'],
             behaviour_df=None, locaNMF_data=None, balance_targets=None, aligned_to='lick',
             balancing_method='subsample_each_condition',
             exclude_miss_trials=True, return_df=False, protocol='MP',
             locanmf_fieldname='components_per_trial_lick_aligned',
             ave_window={'locaNMFcomponents': [-1.5, 0]},
             extra_columns_to_keep=None):
    """
    Make X and y matrices for regression or decoding
    Parameters
    ----------
    y_target : str
        what you want to decode, eg. 'LotteryChoice', 'AirPuffLeft', 'LeftChoice'
    X_features : str
        what features you want to include to do the decoding
    behaviour_df : pandas dataframe
        dataframe containing trial by trial data such as choices and trial conditions
    locaNMF_data : dict
        dict or dict-like object with widefield pixel averaged data or locaNMF data
    balance_targets : list
        list of variables you want to balance, eg. putting ['LotteryChoice'] will ensure that
        there is 50% lottery choice and 50% surebet choice in the subset of data used for decoding
    aligned_to : str
        event to align the neural data to, or event that the neural data was already aligned to
    balancing_method: str
        how you want to do the trial-type balancing
        options: 'subsample_each_condition', 'min_condition'
    exclude_miss_trials : bool
        whether or not to exclude miss trials (trials where the mice did not make a choice)
    return_df : bool
        whether or not to return the subset of the original dataframe used for making X and y
    protocol : str
        what protocol you are running, the code assumes certain available fields in the dataframe depending
        on which protocol you are using
    locanmf_fieldname : str
        key name within locaNMF_data used to obtain the neural data
    ave_window : dict
        if using neural activity as features, the time window to average the neural activity
    extra_columns_to_keep : list
        list of extra columns you want to keep after performing balancing on the pandas dataframe
    Returns 
    -------
    X : numpy ndarray 
    y : numpy ndarray 
    subset_df : pandas dataframe 
    """

    if balance_targets is None:
        balance_targets = []


    if not type(protocol) is list:
        protocol = [protocol]

    if exclude_miss_trials:
        if bool(set(protocol) & set(['MP', 'MatchingPennies_WF'])):  # check if list of protocol contains any matching
            if y_target == 'prevReward':
                subset_trials = np.where((behaviour_df['choice'].values != 'M') &
                                         ~np.isnan(behaviour_df['prevReward'].values))[0]
                behaviour_df = behaviour_df.iloc[subset_trials]
            elif (y_target == 'prevChoice') or ('prevChoice' in balance_targets):
                subset_trials = np.where((behaviour_df['choice'].values != 'M') &
                                         (behaviour_df['prevChoice'].values != 'M'))[0]
                behaviour_df = behaviour_df.iloc[subset_trials]
            else:
                subset_trials = np.where(behaviour_df['choice'].values != 'M')[0]
                behaviour_df = behaviour_df.iloc[subset_trials]
        elif bool(set(protocol) & set(['riskychoice'])):
            # This is Oli's data
            subset_trials = np.where(behaviour_df['MissTrial'].values == 0)[0]
            behaviour_df = behaviour_df.iloc[subset_trials]

        if locaNMF_data is not None:
            components_per_trial = locaNMF_data[locanmf_fieldname].copy()[:, subset_trials, :]
            """
            if aligned_to == 'lick':
                components_per_trial = locaNMF_data['components_per_trial_lick_aligned'].copy()[:, subset_trials, :]
            else:
                components_per_trial = locaNMF_data['components_per_trial'].copy()[:, subset_trials, :]
            """
    else:
        if locaNMF_data is not None:
            if aligned_to == 'lick':
                components_per_trial = locaNMF_data['components_per_trial_lick_aligned'].copy()
            else:
                components_per_trial = locaNMF_data['components_per_trial'].copy()

    # create trial variable to keep track of which trials to recover from feature matrix
    behaviour_df['Trial'] = np.arange(len(behaviour_df))


    if len(balance_targets) > 0:

        columns_to_keep = balance_targets.copy()

        # also keep the behaviour features we are going to use for the classification
        for x_feat in X_features:
            if (x_feat not in columns_to_keep) & (x_feat in behaviour_df.columns):
                columns_to_keep.append(x_feat)

        if y_target not in columns_to_keep:
            columns_to_keep.append(y_target)

        columns_to_keep.append('Trial')

        if extra_columns_to_keep is not None:
            columns_to_keep.extend(extra_columns_to_keep)

        if balancing_method == 'min_condition':
            grouped = behaviour_df.groupby(balance_targets)[columns_to_keep]
            min_size = grouped.size().min()
            group_columns = balance_targets.copy()

            if y_target not in group_columns:
                group_columns.append(y_target)

            # Undersample to match the smallest group
            behaviour_df = grouped.apply(lambda x: x.sample(n=min_size, random_state=None))  # .reset_index(drop=True)
            # balance_subset_idx =
        elif balancing_method == 'subsample_each_condition':
            df_list = []
            # for each target (eg. reward), balance left/right choice (balance_target)
            # currently only supports one balancing target
            """
            for target_val in np.unique(behaviour_df[y_target]):

                target_cond_df = behaviour_df.loc[
                    behaviour_df[y_target] == target_val
                ]
                unique_balance_vars, unique_balance_counts = np.unique(target_cond_df[balance_targets[0]], return_counts=True)
                min_balance_val = np.min(unique_balance_counts)

                for balance_var in unique_balance_vars:
                    df_list.append(target_cond_df.loc[
                        target_cond_df[balance_targets[0]] == balance_var
                                   ].sample(n=min_balance_val, random_state=None))
            """
            # TEMP CODE TO CHECK BUG
            reward_df = behaviour_df.loc[
                behaviour_df['reward'] == 1
            ]

            choice_vars, choice_counts = np.unique(reward_df['choice'], return_counts=True)
            if len(choice_counts) <= 1:
                reward_min_choice = 0 # only one choice
            else:
                reward_min_choice = min(choice_counts)

            reward_choose_left_df = reward_df.loc[
                reward_df['choice'] == 'L'
            ]

            reward_choose_right_df = reward_df.loc[
                reward_df['choice'] == 'R'
            ]

            reward_choose_left_df_subset = reward_choose_left_df.iloc[
                np.random.choice(np.arange(len(reward_choose_left_df)), reward_min_choice,
                                 replace=False)
            ]

            reward_choose_right_df_subset = reward_choose_right_df.iloc[
                np.random.choice(np.arange(len(reward_choose_right_df)), reward_min_choice,
                                 replace=False)
            ]

            df_list.append(reward_choose_left_df_subset)
            df_list.append(reward_choose_right_df_subset)

            # No reward condition

            no_reward_df = behaviour_df.loc[
                behaviour_df['reward'] == 0
            ]
            choice_vars, choice_counts = np.unique(no_reward_df['choice'], return_counts=True)
            if len(choice_counts) <= 1:
                noreward_min_choice = 0
            else:
                noreward_min_choice = min(choice_counts)

            noreward_choose_left_df = no_reward_df.loc[
                no_reward_df['choice'] == 'L'
            ]

            noreward_choose_right_df = no_reward_df.loc[
                no_reward_df['choice'] == 'R'
                ]

            noreward_choose_left_df_subset = noreward_choose_left_df.iloc[
                np.random.choice(np.arange(len(noreward_choose_left_df)), noreward_min_choice,
                                 replace=False)
            ]

            noreward_choose_right_df_subset = noreward_choose_right_df.iloc[
                np.random.choice(np.arange(len(noreward_choose_right_df)), noreward_min_choice,
                                 replace=False)
            ]


            df_list.append(noreward_choose_left_df_subset)
            df_list.append(noreward_choose_right_df_subset)

            behaviour_df = pd.concat(df_list)



    # Make y target variable
    y_values = behaviour_df[y_target].values
    if np.ndim(behaviour_df[y_target].values) > 1:
        y = behaviour_df[y_target].values[:, 0]
    else:
        y = y_values


    # Make feature matrix
    X = []
    for x_feat in X_features:
        # NOTE: currently you can't use locaNMFcomponents together with behavioural features
        # because locaNMF components also have a time dimension...
        if x_feat == 'locaNMFcomponents':
            X.append(components_per_trial)
        elif x_feat == 'prevChoiceR':
            X.append(behaviour_df['prevChoiceR'].values.reshape(-1, 1))
        elif x_feat == 'prevReward':
            X.append(behaviour_df['prevReward'].values.reshape(-1, 1))
        elif x_feat == 'prevReward_X_prevChoiceR':
            interaction_term = behaviour_df['prevReward'] * behaviour_df['prevChoiceR']
            X.append(interaction_term.values.reshape(-1, 1))
        elif x_feat == 'session_progress':
            X.append(behaviour_df['session_progress'].values.reshape(-1, 1))
        elif x_feat == 'smoothed_reward_rate':
            X.append(behaviour_df['smoothed_reward_rate'].values.reshape(-1, 1))
        elif x_feat == 'p_stochastic':
            X.append(behaviour_df['p_stochastic'].values.reshape(-1, 1))
        elif x_feat == 'locaNMFcomponents_timeAve':

            if aligned_to == 'lick':
                time_bin_to_subset = np.where(
                    (locaNMF_data['lick_time_bins'] >= ave_window['locaNMFcomponents'][0]) &
                    (locaNMF_data['lick_time_bins'] <= ave_window['locaNMFcomponents'][1])
                )[0]

            components_per_trial_mean = np.mean(components_per_trial[:, :, time_bin_to_subset], axis=2).T
            components_per_trial_mean = components_per_trial_mean[behaviour_df['Trial'].values, :]
            X.append(components_per_trial_mean)

    if len(X) == 1:
        X = np.array(X)[0, :, :]
    else:
        X = np.squeeze(np.concatenate(X, axis=1))

    if np.ndim(X) == 3:  # only applies to locaNMFcomponents features, but not to behaviour features
        # susbet based on behaviour_df['Trial'] to get the subsampled (balanced) trials
        X = X[:, behaviour_df['Trial'].values, :]  # features x trial x time


        # Drop columns with NaNs
        features_without_nans = np.sum(np.isnan(X), axis=(1, 2)) == 0
        X = X[features_without_nans, :, :]
    else:
        trials_without_nan = (np.sum(np.isnan(X), axis=1) == 0) & (~np.isnan(y))
        X = X[trials_without_nan, :]
        y = y[trials_without_nan]

    if return_df:  # drop duplicate columns
        subset_df = behaviour_df.loc[:, ~behaviour_df.columns.duplicated()].copy()
        return X, y, subset_df
    else:
        return X, y


def decode_task_var(X, y, clf=svm.SVC(kernel='linear'), num_cv=5, num_shuffle=100, return_weights=False,
                    weight_type='coefs', trained_model=None, num_cv_repeats=1):
    """Performs decoding of task variable from neural or behavioural features.

    Parameters
    ----------
    X : numpy ndarray
        feature matrix
    y : numpy ndarray
        target variable
    clf : scikit-learn or similar classifier object, optional
        scikit-learn classifier object, by default svm.SVC(kernel='linear')
    num_cv : int, optional
        number of cross-validation folds, by default 5
    num_shuffle : int, optional
        number of shuffles to get the null distribution of decoding accuracy, by default 100
    return_weights : bool, optional
        whether to return weights of the classifier, by default False
    weight_type : str, optional
        type of weight to return, by default 'coefs'
    trained_model : sciki-learn or similar classifier object, optional
        pre-trained model to perform decoding anlaysis, by default None
    num_cv_repeats : int, optional
        number of repeats to do for n-fold cross-validation, by default 1

    Returns
    -------
    accuracy : numpy ndarray
        accuracy (ie. P(hit)) across time and trials
    accuracy_null : numpy ndarray
        accuracy from null distribution (obtained by shuffling trials)
    weights : numpy ndarray
        weights obtained from the classifier object
    weights_null : numpy ndarray
        weights obtained from the classifier object trained on shuffled data
    clf : sciki-learn or similar classifier object, optional
        decoder object to perform decoding analysis
    train_test_indices
    """

    cv_splitter = RepeatedStratifiedKFold(n_splits=num_cv, n_repeats=num_cv_repeats)

    accuracy_per_cv = np.zeros((num_cv * num_cv_repeats,)) + np.nan

    if return_weights:

        # initialise the weight array
        num_features = np.shape(X)[1]
        if type(weight_type) is list:
            num_weight_types = len(weight_type)
            weights = np.zeros((num_weight_types * 2, num_features, num_cv*num_cv_repeats)) + np.nan
        else:
            weights = np.zeros((num_features, num_cv*num_cv_repeats)) + np.nan
        
        weights_null = np.zeros((num_features, num_shuffle)) + np.nan
        
        num_trials = np.shape(X)[0]
        train_test_indices = np.zeros((2, num_trials, num_cv*num_cv_repeats))
    

    for cv_i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        if trained_model is None:
            clf.fit(X[train_idx, :], y[train_idx])
        else:
            clf = trained_model

        y_hat = clf.predict(X[test_idx, :])
        accuracy = np.sum(y_hat == y[test_idx]) / len(test_idx)
        accuracy_per_cv[cv_i] = accuracy

        train_test_indices[0, train_idx, cv_i] = 1
        train_test_indices[1, test_idx, cv_i] = 1

        if return_weights:

            if type(weight_type) is list:

                for weight_idx in np.arange(len(weight_type)):

                    if weight_type[weight_idx] == 'coefs':
                        weights[weight_idx, :, cv_i] = clf.named_steps['svm'].coef_

                    elif weight_type[weight_idx] == 'shap':
                        X_train_zscored = (X[train_idx, :] - np.mean(X[train_idx, :], axis=0)) / np.std(X[train_idx, :],
                                                                                                        axis=0)
                        # X_test_zscored = (X[test_idx, :] - np.mean(X[train_idx, :], axis=0)) / np.std(
                        #     X[train_idx, :], axis=0)
                        # masker = shap.maskers.Independent(data=X_train_zscored)
                        masker = shap.maskers.Impute(data=X_train_zscored)
                        explainer = shap.LinearExplainer(clf.named_steps['svm'], masker=masker)
                        # Take the mean SHAP value across all trained samples
                        shap_values = explainer.shap_values(X_train_zscored)
                        weights[weight_idx, :, cv_i] = np.mean(shap_values, axis=0)

                    elif weight_type[weight_idx] == 'shap_scaled':

                        X_train_zscored = (X[train_idx, :] - np.mean(X[train_idx, :], axis=0)) / np.std(
                            X[train_idx, :], axis=0)
                        # masker = shap.maskers.Independent(data=X_train_zscored)
                        masker = shap.maskers.Impute(data=X_train_zscored)
                        explainer = shap.LinearExplainer(clf.named_steps['svm'], masker=masker)
                        # Take the mean SHAP value across all trained samples
                        shap_values = explainer.shap_values(X_train_zscored)
                        weights[weight_idx, :, cv_i] = np.mean(shap_values, axis=0) / np.std(shap_values, axis=0)

                    elif weight_type[weight_idx] == 'shap_abs':

                        X_train_zscored = (X[train_idx, :] - np.mean(X[train_idx, :], axis=0)) / np.std(
                            X[train_idx, :], axis=0)
                        # masker = shap.maskers.Independent(data=X_train_zscored)
                        masker = shap.maskers.Impute(data=X_train_zscored)
                        explainer = shap.LinearExplainer(clf.named_steps['svm'], masker=masker)
                        # Take the mean SHAP value across all trained samples
                        shap_values = explainer.shap_values(X_train_zscored)
                        weights[weight_idx, :, cv_i] = np.mean(np.abs(shap_values), axis=0)

                    elif weight_type[weight_idx] == 'huaffe_transform':
                        weights = clf.named_steps['svm'].coef_
                        y_pred = X[train_idx, :] @ weights.T
                        y_pred = y_pred.flatten()
                        # Haufe transform: covariance of X_train with y_pred
                        weights[weight_idx, :, cv_i] = np.cov(X[train_idx, :].T, y_pred, bias=True)[-1, :-1] / np.var(y_pred)


            elif isinstance(clf, Pipeline):

                if weight_type == 'coefs':
                    weights[weight_idx, :, cv_i] = clf.named_steps['svm'].coef_
                elif weight_type == 'huaffe_transform':
                    # TEMP: do a Hauffe transformation
                    # X_cov = np.cov(X.T)
                    # weights_at_window = np.matmul(weights_at_window, X_cov)

                    # Project X_train into decoder space
                    w = clf.named_steps['svm'].coef_
                    y_pred = X[train_idx, :] @ w.T
                    y_pred = y_pred.flatten()
                    # Haufe transform: covariance of X_train with y_pred
                    weights[weight_idx, :, cv_i] = np.cov(X[train_idx, :].T, y_pred, bias=True)[-1, :-1] / np.var(y_pred)

                elif weight_type == 'shap':

                    explainer = shap.Explainer(clf.named_steps['svm'])
                    weights[weight_idx, :, cv_i] = explainer(X[train_idx, :])

                # weights_per_window[decoding_bin_idx, :, cv_i] = weights_at_window
            elif 'coef_' in dir(clf):
                weights[weight_idx, :, cv_i] = clf.coef_
        # if cv_i == 0:
            # model_per_window.append(clf)

    # cv_score = sklselection.cross_val_score(clf, X, y, cv=cv_splitter)
    accuracy = np.mean(accuracy_per_cv)

    accuracy_per_shuffle = np.zeros((num_shuffle, ))

    for shuffle_idx in np.arange(num_shuffle):
        y_shuffled = y.copy()
        np.random.shuffle(y_shuffled)
        cv_splitter = RepeatedStratifiedKFold(n_splits=num_cv, n_repeats=1)
        accuracy_per_cv_shuffle = np.zeros((num_cv,)) + np.nan
        weights_at_window_per_cv = []
        for cv_i, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y_shuffled)):
            clf.fit(X[train_idx, :], y_shuffled[train_idx])
            y_hat = clf.predict(X[test_idx, :])
            accuracy_shuffled = np.sum(y_hat == y_shuffled[test_idx]) / len(test_idx)
            accuracy_per_cv_shuffle[cv_i] = accuracy_shuffled
            if return_weights:
                if isinstance(clf, Pipeline):
                    weights_at_window = clf.named_steps['svm'].coef_
                weights_at_window_per_cv.append(weights_at_window)

        weights_at_window_per_cv = np.array(weights_at_window_per_cv)
        shuffled_weights_per_window = np.mean(weights_at_window_per_cv, axis=0)
        # cv_score = sklselection.cross_val_score(clf, X, y_shuffled, cv=cv_splitter)
        cv_score = np.mean(accuracy_per_cv_shuffle)
        accuracy_per_shuffle[shuffle_idx] = np.mean(cv_score)
        weights_null[:, shuffle_idx] = weights_at_window_per_cv[0]

    accuracy_null = np.mean(accuracy_per_shuffle)

    return accuracy, accuracy_null, weights, weights_null, clf, train_test_indices


def decode_task_var_windowed(X_t, y, time_bins, decoding_time_bins=np.linspace(-1.5, 2, 11),
                             decoding_bin_width=None,
                             clf=svm.SVC(kernel='linear'), num_cv=5, num_shuffle=100, return_weights=False,
                             weight_type='coefs', trained_model_per_window=None, run_in_parallel=False,
                             num_cv_repeats=1):
    """
    Perform time-resolved decoding of a task variable using a sliding window approach.

    This function averages the data within specified time windows and trains a classifier
    (e.g., linear SVM) to decode the task variable `y`. Optionally, it computes feature
    weights (e.g., classifier coefficients, SHAP values, or Haufe-transformed weights)
    and compares decoding accuracy against a null distribution generated from label shuffling.

    Parameters
    ----------
    X_t : ndarray of shape (n_features, n_trials, n_time_bins)
        Neural or feature data across time.

    y : array-like of shape (n_trials,)
        Labels or target variable to decode.

    time_bins : array-like of shape (n_time_bins,)
        Time vector corresponding to the last axis of `X_t`.

    decoding_time_bins : array-like of shape (n_bins + 1,), optional
        The edges of time windows for decoding. Default is `np.linspace(-1.5, 2, 11)`.

    clf : sklearn-compatible classifier or pipeline, optional
        The classifier to use for decoding. Should implement `fit` and `predict`.
        Defaults to a linear SVM.

    num_cv : int, optional
        Number of cross-validation folds. Default is 5.

    num_shuffle : int, optional
        Number of shuffles to generate the null distribution. Default is 100.

    return_weights : bool, optional
        If True, compute and return feature weights for each decoding window. Default is False.

    weight_type : str or list of str, optional
        Method(s) used to compute feature weights. Options include:
        - 'coefs': Classifier coefficients
        - 'shap': SHAP values
        - 'shap_scaled': SHAP values normalized by their standard deviation
        - 'shap_abs': Absolute SHAP values
        - 'huaffe_transform': Haufe-transformed coefficients
        Can be a single string or list of strings. Default is 'coefs'.

    trained_model_per_window : list of classifiers or None, optional
        Pre-trained models to use per decoding window. If provided, should match the number
        of decoding windows. Default is None.

    Returns
    -------
    accuracy_per_window : ndarray of shape (n_windows,)
        Mean decoding accuracy across cross-validation splits for each time window.

    accuracy_per_window_null : ndarray of shape (n_windows, num_shuffle)
        Decoding accuracy from shuffled labels (null distribution) per time window.

    weights_per_window : ndarray, optional
        Feature weights per window, returned if `return_weights` is True.
        Shape is (n_weight_types * 2, n_windows, n_features, n_cv), where the second
        set of weights are scaled by decoding accuracy.

    weights_per_window_null : ndarray, optional
        Feature weights per window computed on shuffled data, returned if `return_weights` is True.
        Shape is (n_windows, n_features, num_shuffle).

    weight_names : list of str, optional
        Names of weight types, including accuracy-scaled variants, returned if `return_weights` is True.

    model_per_window : list of classifiers, optional
        The trained classifier for the first cross-validation fold of each window,
        returned if `return_weights` is True.

    run_in_parallel : bool
        Whether to run windowed decoding accuracy in parallel
        joblib is used for this

    num_cv_repeats : int
        number of times to re-run cross-validation
        increase this to get smoother results
    Notes
    -----
    - SHAP values are computed using `shap.LinearExplainer` with `Impute` maskers.
    - Haufe-transformed weights are derived using the covariance between input and decoder output.
    """


    accuracy_per_window = np.zeros((len(decoding_time_bins) - 1), ) + np.nan
    accuracy_per_window_null = np.zeros((len(decoding_time_bins) - 1, num_shuffle)) + np.nan

    if return_weights:
        num_features = np.shape(X_t)[0]

        if type(weight_type) is list:
            num_weight_types = len(weight_type)
            weights_per_window = np.zeros((num_weight_types*2, len(decoding_time_bins) - 1, num_features, num_cv*num_cv_repeats)) + np.nan
        else:
            weights_per_window = np.zeros((len(decoding_time_bins) - 1, num_features, num_cv*num_cv_repeats)) + np.nan
        weights_per_window_null =  np.zeros((len(decoding_time_bins) - 1, num_features, num_shuffle)) + np.nan
        model_per_window = []

        # binary matrix to show whether each trial is in the train or test set
        n_trials = np.shape(X_t)[1]
        train_test_indices_per_window = np.zeros((2, n_trials, len(decoding_time_bins)-1, num_cv*num_cv_repeats)) + np.nan

    if run_in_parallel:

        time_bins_list = []
        for decoding_bin_idx in np.arange(len(decoding_time_bins) - 1):

            if decoding_bin_width is not None:
                time_bins_to_get = np.where(
                    (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[decoding_bin_idx] + decoding_bin_width)
                )[0]
            else:
                time_bins_to_get = np.where(
                    (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[decoding_bin_idx + 1])
                )[0]
            time_bins_list.append(time_bins_to_get)

        parallel_results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(decode_task_var)(
                np.mean(X_t[:, :, time_bins_to_get], axis=2).T, y, clf=clone(clf),
                num_cv=num_cv, num_shuffle=num_shuffle, return_weights=return_weights,
                weight_type=weight_type, trained_model=None, num_cv_repeats=num_cv_repeats,
            ) for time_bins_to_get in time_bins_list
        )

        for decoding_bin_idx in np.arange(len(decoding_time_bins) - 1):
            accuracy, accuracy_null, weights, weights_null, model_per_window, train_test_indices = parallel_results[decoding_bin_idx]
            accuracy_per_window[decoding_bin_idx] = np.mean(accuracy)
            accuracy_per_window_null[decoding_bin_idx, :] = accuracy_null
            weights_per_window[:, decoding_bin_idx, :, :] = weights
            weights_per_window_null[decoding_bin_idx, :, :] = weights_null

            # train_test_indices has shape: train_test_indices = np.zeros((2, num_trials, num_cv*num_cv_repeats))
            train_test_indices_per_window[:, :, decoding_bin_idx, :] = train_test_indices

    else:
        for decoding_bin_idx in np.arange(len(decoding_time_bins) - 1):

            if decoding_bin_width is not None:
                time_bins_to_get = np.where(
                    (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[decoding_bin_idx] + decoding_bin_width)
                )[0]
            else:
                time_bins_to_get = np.where(
                    (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[decoding_bin_idx + 1])
                )[0]

            X = np.mean(X_t[:, :, time_bins_to_get], axis=2).T

            if trained_model_per_window is not None:
                trained_model_at_window = trained_model_per_window[decoding_bin_idx]
            else:
                trained_model_at_window = None


            accuracy, accuracy_null, weights, weights_null, clf, train_test_indices = decode_task_var(X, y, clf=clf,
                            num_cv=num_cv, num_shuffle=num_shuffle, return_weights=return_weights,
                            weight_type=weight_type, trained_model=trained_model_at_window,
                            num_cv_repeats=num_cv_repeats)

            accuracy_per_window[decoding_bin_idx] = np.mean(accuracy)
            accuracy_per_window_null[decoding_bin_idx, :] = accuracy_null
            weights_per_window[:, decoding_bin_idx, :, :] = weights
            weights_per_window_null[decoding_bin_idx, :, :] = weights_null
            train_test_indices_per_window[:, :, decoding_bin_idx, :] = train_test_indices
            model_per_window.append(clf)

    # Get decoding relative to null
    mean_null_accuracy_per_window = np.mean(accuracy_per_window_null, axis=1)
    accuracy_rel_null = (accuracy_per_window - mean_null_accuracy_per_window) / (1 - mean_null_accuracy_per_window)

    # Calculate weights per window scaled by decoding accuracy
    weight_names = weight_type.copy()
    if type(weight_type) is not list:
        weight_type = [weight_type]

    for w_idx in np.arange(len(weight_type)):
        w_name = weight_type[w_idx] + '_accuracy_scaled'
        weight_names.append(w_name)
        weights_per_window[len(weight_type)+w_idx, :, :, :] = weights_per_window[w_idx, :, :, :] * np.expand_dims(np.expand_dims(accuracy_rel_null, -1), -1)


    if return_weights:
        return accuracy_per_window, accuracy_per_window_null, weights_per_window, weights_per_window_null, weight_names, model_per_window, train_test_indices_per_window
    else:
        return accuracy_per_window, accuracy_per_window_null
    

def decode_task_var_across_window(X_t, y, time_bins, decoding_time_bins=np.linspace(-1.5, 2, 11),
                             decoding_bin_width=None,
                             clf=svm.SVC(kernel='linear'), num_cv=5, num_cv_repeats=1):
    

    accuracy_cross_window_matrix = np.zeros((len(decoding_time_bins) - 1, len(decoding_time_bins) - 1))
    

    time_bins_list = []
    for decoding_bin_idx in np.arange(len(decoding_time_bins) - 1):

        if decoding_bin_width is not None:
            time_bins_to_get = np.where(
                (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                (time_bins < decoding_time_bins[decoding_bin_idx] + decoding_bin_width)
            )[0]
        else:
            time_bins_to_get = np.where(
                (time_bins >= decoding_time_bins[decoding_bin_idx]) &
                (time_bins < decoding_time_bins[decoding_bin_idx + 1])
            )[0]
        time_bins_list.append(time_bins_to_get)

    
    cv_splitter = RepeatedStratifiedKFold(n_splits=num_cv, n_repeats=num_cv_repeats)


    for train_decoding_bin_idx in np.arange(len(decoding_time_bins) - 1): 

        if decoding_bin_width is not None:
            time_bins_to_get_1 = np.where(
                (time_bins >= decoding_time_bins[train_decoding_bin_idx]) &
                (time_bins < decoding_time_bins[train_decoding_bin_idx] + decoding_bin_width)
            )[0]
        else:
            time_bins_to_get_1 = np.where(
                (time_bins >= decoding_time_bins[train_decoding_bin_idx]) &
                (time_bins < decoding_time_bins[train_decoding_bin_idx + 1])
            )[0]

        X_bin_1 = np.mean(X_t[:, :, time_bins_to_get_1], axis=2).T

        train_idx_list = [] 
        test_idx_list = [] 
        clf_list = []

        for train_idx, test_idx in cv_splitter.split(X_bin_1, y):

            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
            clf_clone = clone(clf)
            clf_clone.fit(X_bin_1[train_idx, :], y[train_idx])
            clf_list.append(clf_clone)


        for test_decoding_bin_idx in np.arange(len(decoding_time_bins) - 1):

            if decoding_bin_width is not None:
                time_bins_to_get_2 = np.where(
                    (time_bins >= decoding_time_bins[test_decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[test_decoding_bin_idx] + decoding_bin_width)
                )[0]
            else:
                time_bins_to_get_2 = np.where(
                    (time_bins >= decoding_time_bins[test_decoding_bin_idx]) &
                    (time_bins < decoding_time_bins[test_decoding_bin_idx + 1])
                )[0]
            
            X_bin_2 = np.mean(X_t[:, :, time_bins_to_get_2], axis=2).T

            accuracy_per_cv = np.zeros((num_cv * num_cv_repeats,)) + np.nan

            for cv_i in np.arange(num_cv * num_cv_repeats):
                test_idx = test_idx_list[cv_i]
                y_hat = clf_list[cv_i].predict(X_bin_2[test_idx, :])
                accuracy = np.sum(y_hat == y[test_idx]) / len(test_idx)
                accuracy_per_cv[cv_i] = accuracy

           
            """
            for cv_i, (train_idx, test_idx) in enumerate(cv_splitter.split(X_bin_1, y)):
                
                clf.fit(X_bin_1[train_idx, :], y[train_idx])

                y_hat = clf.predict(X_bin_2[test_idx, :])
                accuracy = np.sum(y_hat == y[test_idx]) / len(test_idx)
                accuracy_per_cv[cv_i] = accuracy
            """

            accuracy_cross_window_matrix[train_decoding_bin_idx, test_decoding_bin_idx] = np.mean(accuracy_per_cv)


    return accuracy_cross_window_matrix


def resample_states_from_each_day(behaviour_data, state_column='p_stochastic', p_threshold=0.8,
                                  min_trials=25):

    behaviour_data['index'] = np.arange(0, len(behaviour_data))
    behaviour_data_subset_list = []
    for date in np.unique(behaviour_data['date'].values):

        # NOTE: Currently hard-coded to only support a list of two elements
        if type(state_column) is list:
            date_df = behaviour_data.loc[
                (behaviour_data['date'] == date) &
                (behaviour_data[state_column[0]] >= p_threshold) |
                (behaviour_data[state_column[1]] >= p_threshold)
            ]
        else:
            date_df = behaviour_data.loc[
                (behaviour_data['date'] == date) &
                (behaviour_data[state_column] >= p_threshold)
            ]

        num_trials = len(date_df)
        if num_trials >= min_trials:
            behaviour_data_subset_list.append(
                date_df.iloc[np.random.choice(num_trials, min_trials, replace=False)]
            )

    behaviour_data_subset = pd.concat(behaviour_data_subset_list)
    subset_index = behaviour_data_subset['index'].values

    return behaviour_data_subset, subset_index


def do_state_trialcount_balancing(X_1, y_1, df_1, X_2, y_2, df_2, balance_targets):
    """_summary_

    Parameters
    ----------
    X_1 : numpy ndarray 
        _description_
    y_1 : _type_
        _description_
    df_1 : pandas dataframe
        pandas dataframe for state 1
    X_2 : _type_
        _description_
    y_2 : _type_
        _description_
    df_2 : _type_
        _description_
    balance_targets : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # Get the minimum trial count for each balancing condition across the two states
    df_1_min_count = np.min(df_1.groupby(balance_targets).agg('count'))
    df_2_min_count = np.min(df_2.groupby(balance_targets).agg('count'))
    states_min_count = np.min([df_1_min_count, df_2_min_count])

    if df_1_min_count > df_2_min_count:
        df_1['index'] = np.arange(len(df_1))
        grouped = df_1.groupby(balance_targets)
        df_1_resampled = grouped.apply(lambda x: x.sample(n=states_min_count, random_state=None))
        X_1 = X_1[:, df_1_resampled['index'].values, :]
        y_1 = y_1[df_1_resampled['index'].values]
        df_1 = df_1_resampled['Trial'].reset_index()

    elif df_1_min_count < df_2_min_count:
        df_2['index'] = np.arange(len(df_2))
        grouped = df_2.groupby(balance_targets)
        df_2_resampled = grouped.apply(lambda x: x.sample(n=states_min_count, random_state=None))
        X_2 = X_2[:, df_2_resampled['index'].values, :]
        y_2 = y_2[df_2_resampled['index'].values]
        df_2 = df_2_resampled['Trial'].reset_index()


    return X_1, y_1, df_1, X_2, y_2, df_2

def process_decoding_time_bins(decoding_time_bins, time_bins, decoding_bin_width=None):
    """
    Make sure specified decoding_time_bins is valid to be used
    :param decoding_time_bins:
    :param time_bins:
    :param decoding_bin_width:
    :return:
    """

    if decoding_bin_width is not None:
        num_decoding_bins = len(decoding_time_bins)
        # TODO: write couple of checks here to ensure the specified time bins works, and also ensure
        # the decoding_time_bin is not to narrow, and that there are enough datapoints for the specified number of bins

        # NOTE: Currently assume uniform time bins
        decoding_time_bins_step = np.mean(np.diff(decoding_time_bins))
        time_bins_step = np.mean(np.diff(time_bins))

        if decoding_time_bins_step < time_bins_step:
            logger.info('Specified decoding time bin step size is smaller than available, increasing it to the '
                        'smallest available step size')
            decoding_time_bins = np.arange(decoding_time_bins[0], decoding_time_bins[-1], time_bins_step)

        decoding_bin_overrun = np.where(decoding_time_bins + decoding_bin_width > time_bins[-1])[0]
        if len(decoding_bin_overrun) > 0:
            logger.info('Some specified decoding time bins is beyond the available time bins, dropping them')
            decoding_time_bins = decoding_time_bins[decoding_time_bins + decoding_bin_width <= time_bins[-1]]


    return decoding_time_bins, decoding_bin_width

def fit_locanmf_decoder(subject=None, date=None, protocol=None, locaNMF_data=None,
                        y_target='reward', activity_to_use='locaNMF',
                        balancing_method='subsample_each_condition', balance_targets=['choice'],
                        states_to_use=['stochastic', 'other'], weight_type='coefs', session_progress_range=[0, 1],
                        resample_states=False,
                        decoding_time_start=-1.5,
                        decoding_time_end=1.5,
                        decoding_time_step=0.2,
                        decoding_bin_width=None, run_in_parallel=False,
                        num_cv_repeats=1, balance_states_trialcounts=False,
                        fit_each_feature=False, do_across_window_decoding=False,
                        state_p_threshold=0.8, subset_conditions=None,
                        extra_predictors=None, remove_neural=False):
    """
    Fits a decoder to widefield data (either locaNMF components or mean activity per region)
    Outline of the steps: 
    1: load the widefield and behaviour data
    2. Balance the states (eg. stochastic and bias)
    3. Balance the specified conditions (eg. reward and choose left/right)
    4. Go through each window and for each window, do X

    Parameters
    -----------
    subject : str or None
        name of the subjects to fit
    date : str or None 
        which dates to fit, if None, then fit to all dates
    protocol : str or None
        which protocol to fit the data
    balancing_method : str
        'subsample_each_condition' : for each reward condition, subsample L/R choice
        'min_condition' :
    project_to_decoding_axis : bool 
        whether to project the trial by trial population activity to the decoding axis
        this process is also cross-validated; we project the test data onto the trained weights
        
    Outputs 
    --------
    decoder_results : dict with the following fields...
        accuracy_per_window_1, accuracy_per_window_2, accuracy_per_window_3:
            decoding accuracy per time window in states numbered 1, 2, 3
        weights_per_window_1 :
            MP Task
            weights during the stochastic state
            Riskychoice Task
            weights during the engaged state
        weights_per_window_2 :
            MP Task
            weights_during the biased state
            Riskychoice Task
            weights during the disengaged state
    """

    if type(weight_type) is not list:
        weight_type= [weight_type]

    if locaNMF_data is None:

        if activity_to_use == 'pixel_ave':
            include_pixel_data = True
        else:
            include_pixel_data = False

        locaNMF_data = mp_dataset.load_data(subject=subject, date=date, protocol=protocol,
                                        data_type='locaNMF', include_miss_trials=False,
                                        include_pixel_data=include_pixel_data, Vc_Uc_datatype='npz')
        behaviour_data = locaNMF_data['behaviour_df']
        # behaviour_data = mp_dataset.load_data(subject=subject, date=date, protocol=protocol,
         #                                      data_type='mouseMP', include_miss_trials=False)
    else:
        behaviour_data = locaNMF_data['behaviour_df']

    if activity_to_use == 'locaNMF':
        # NOTE: Currently this assumes MP protocol
        feature_name = 'components_per_trial_lick_aligned'
        locanmf_fieldname = 'components_per_trial_lick_aligned'
        # decoding_time_bins = np.linspace(-1.5, 1.5, 15)
        decoding_time_bins = np.arange(decoding_time_start, decoding_time_end, decoding_time_step)
    elif activity_to_use == 'pixel_ave':

        if 'pixel_ave_lick_aligned' in locaNMF_data.keys():
            feature_name = 'pixel_ave_lick_aligned'
            locanmf_fieldname = 'pixel_ave_lick_aligned'
            decoding_time_bins = np.linspace(-1.5, 1.5, 15)
        else:
            feature_name = 'pixel_ave'
            locanmf_fieldname = 'pixel_ave'
            # decoding_time_bins = np.linspace(-1, 4, 20)
            decoding_time_bins = np.arange(decoding_time_start, decoding_time_end, decoding_time_step)
            protocol = 'riskychoice'

    elif activity_to_use == 'mot_svd':

        # currently only for risky choice
        feature_name = 'mot_svd'
        locanmf_fieldname = 'mot_svd'
        decoding_time_bins = np.linspace(-1, 4, 20)
        protocol = 'riskychoice'
    
    elif activity_to_use == 'sleap':

        feature_name = 'sleap'
        locanmf_fieldname = 'sleap'
        protocol = 'riskychoice'
        decoding_time_bins = np.arange(decoding_time_start, decoding_time_end, decoding_time_step)



    if protocol == 'riskychoice':
        # Calculate session progress, prevChoice, prevReward (if necessary)
        behaviour_data = add_behaviour_features(behaviour_data, feature_names=['session_progress'])
    else:
        behaviour_data = add_behaviour_features(behaviour_data,
                                                feature_names=['prevChoice', 'prevChoiceR',
                                                            'prevReward', 'session_progress',
                                                            'switch', 'prevReward_X_prevChoiceR'])
        
    if locaNMF_data is None:
        return None

    if states_to_use[0] == 'stochastic':

        if resample_states:
            _, state_1_index = resample_states_from_each_day(
                behaviour_data, state_column='p_stochastic', p_threshold=state_p_threshold,
                min_trials=50)
        else:
            state_1_index = np.where(
                (behaviour_data['p_stochastic'].values >= state_p_threshold) &
                (behaviour_data['session_progress'] >= session_progress_range[0]) &
                (behaviour_data['session_progress'] <= session_progress_range[1])
            )[0]

        if states_to_use[1] == 'other':
            other_states_index = np.where(behaviour_data['p_stochastic'].values < state_p_threshold)[0]
        elif states_to_use[1] == 'bias':

            if resample_states:
                _, other_states_index = resample_states_from_each_day(
                    behaviour_data, state_column=['p_leftbias', 'p_rightbias'], p_threshold=state_p_threshold,
                    min_trials=50)
            else:
                other_states_index = np.where(
                    ((behaviour_data['p_leftbias'].values >= state_p_threshold) |
                    (behaviour_data['p_rightbias'].values >= state_p_threshold))  &
                    (behaviour_data['session_progress'] >= session_progress_range[0]) &
                    (behaviour_data['session_progress'] <= session_progress_range[1])
                )[0]
        elif states_to_use[1] == 'all':
            other_states_index = np.arange(len(behaviour_data))


    elif states_to_use[0] == 'engaged':

        state_1_index = np.where(
            (behaviour_data['p_engaged_state'].values >= state_p_threshold) &
            (behaviour_data['session_progress'] >= session_progress_range[0]) &
            (behaviour_data['session_progress'] <= session_progress_range[1])
        )[0]

        if states_to_use[1] == 'other':
            other_states_index = np.where(behaviour_data['p_engaged_state'].values < state_p_threshold)[0]
        elif states_to_use[1] == 'disengaged':
            other_states_index = np.where(
                ((behaviour_data['p_disengaged1_state'].values >= state_p_threshold) |
                 (behaviour_data['p_disengaged2_state'].values >= state_p_threshold)) &
                (behaviour_data['session_progress'] >= session_progress_range[0]) &
                (behaviour_data['session_progress'] <= session_progress_range[1])
            )[0]


    # balancing_method = None  # 'min_condition'  # 'min_condition', 'subsample_each_condition', None
    behaviour_data_1 = behaviour_data.iloc[state_1_index]
    locaNMF_data_1 = locaNMF_data.copy()
    locaNMF_data_1[feature_name] = locaNMF_data[feature_name][:, state_1_index, :]

    behaviour_data_2 = behaviour_data.iloc[other_states_index]
    locaNMF_data_2 = locaNMF_data.copy()
    locaNMF_data_2[feature_name] = locaNMF_data[feature_name][:, other_states_index, :]

    if extra_predictors is not None:
        num_time_points = np.shape(locaNMF_data[feature_name])[2]
        state_1_predictors = np.zeros((len(extra_predictors), len(state_1_index), num_time_points))
        state_2_predictors = np.zeros((len(extra_predictors), len(other_states_index), num_time_points))

        for feature_idx, f_name in enumerate(extra_predictors):
            # repeat along the time dimension
            if f_name == 'prevChoice':
                feature_vals_1 = (behaviour_data_1[f_name].values == 'R').reshape(-1, 1).astype(float)
                feature_vals_2 = (behaviour_data_2[f_name].values == 'R').reshape(-1, 1).astype(float)
            else:
                feature_vals_1 = behaviour_data_1[f_name].values.reshape(-1, 1)
                feature_vals_2 = behaviour_data_2[f_name].values.reshape(-1, 1)
            
            state_1_predictors[feature_idx, :, :] = np.tile(feature_vals_1, (1, num_time_points))
            state_2_predictors[feature_idx, :, :] = np.tile(feature_vals_2, (1, num_time_points))
        
        if remove_neural == 1:
            locaNMF_data_1[feature_name] = state_1_predictors
            locaNMF_data_2[feature_name] = state_2_predictors
        else:
            print('TODO: concatenate behaviour with neural')
            pdb.set_trace()
            locaNMF_data_1[feature_name] = np.concatenate([locaNMF_data_1[feature_name], state_1_predictors], axis=0)
            locaNMF_data_2[feature_name] = np.concatenate([locaNMF_data_2[feature_name], state_2_predictors], axis=0)

    X_1, y_1, df_1 = make_X_y(y_target=y_target, X_features=['locaNMFcomponents'],
                              locanmf_fieldname=locanmf_fieldname,
                             behaviour_df=behaviour_data_1, locaNMF_data=locaNMF_data_1,
                              balance_targets=balance_targets,
                            exclude_miss_trials=True, return_df=True, balancing_method=balancing_method,
                              protocol=protocol, extra_columns_to_keep=None)


    X_2, y_2, df_2 = make_X_y(y_target=y_target, X_features=['locaNMFcomponents'],
                              locanmf_fieldname=locanmf_fieldname,
                                          behaviour_df=behaviour_data_2, locaNMF_data=locaNMF_data_2,
                                          balance_targets=balance_targets,
                                          exclude_miss_trials=True, return_df=True,
                              balancing_method=balancing_method,
                              protocol=protocol, extra_columns_to_keep=None)

    # Make the choice L/R reward/unreward matrix
    choice_reward_matrix_1 = np.zeros((2, 2))

    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    df_1 = df_1.loc[:, ~df_1.columns.duplicated()].copy()
    df_2 = df_2.loc[:, ~df_2.columns.duplicated()].copy()

    if balance_states_trialcounts:
        X_1, y_1, df_1, X_2, y_2, df_2 = do_state_trialcount_balancing(X_1, y_1, df_1, X_2, y_2, df_2, balance_targets)

    df_1['globalTrial'] = state_1_index[df_1['Trial'].values]
    df_2['globalTrial'] = other_states_index[df_2['Trial'].values]


    # TODO: make this work for more than 2 balance targets
    # (not high priority, as this will actually require quite a bit of work)
    if len(balance_targets) == 2:

        row_name, col_name = balance_targets

        row_vals = np.unique(df_1[row_name].values)
        col_vals = np.unique(df_1[col_name].values)
        contigency_table_1 = make_contigency_table(df_1, row_name, col_name, row_vals=row_vals, col_vals=col_vals)
        contigency_table_2 = make_contigency_table(df_2, row_name, col_name, row_vals=row_vals, col_vals=col_vals)

    else:
        contigency_table_1 = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        contigency_table_2 = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    y_1_unique_vals, y_1_unique_counts = np.unique(y_1, return_counts=True)
    y_2_unique_vals, y_2_unique_counts = np.unique(y_2, return_counts=True)
    unique_con_met = (len(y_1_unique_vals) == 2) and (len(y_2_unique_vals) == 2)
    if unique_con_met:
        min_con_met = (np.min(y_1_unique_counts) >= 2) & (np.min(y_2_unique_counts) >= 2)
    else:
        min_con_met = 0

    if unique_con_met and min_con_met:

        # Do decoding
        clf_pipeline = Pipeline([
            ('zscore', sklpreprocessing.StandardScaler()),
            ('svm', sklsvm.LinearSVC(penalty='l2', max_iter=10000))  # originally 2000
        ])

        if 'lick_time_bins' in locaNMF_data.keys():
            time_bins = locaNMF_data['lick_time_bins']
        else:
            time_bins = locaNMF_data['time_rel_stim']

        return_weights = True
        num_cv = 2
        num_shuffle = 50  # number of trial shuffles for null distribution

        decoding_time_bins, decoding_bin_width = process_decoding_time_bins(decoding_time_bins, time_bins=time_bins,
                                                                            decoding_bin_width=decoding_bin_width)

        if fit_each_feature:
            num_features = np.shape(X_1)[0]
            accuracy_per_window_1 = []
            accuracy_per_window_2 = []

            for feature_idx in np.arange(num_features):

                f_i_accuracy_per_window_1, accuracy_per_window_null_1, weights_per_window_1, _, weight_names, model_per_window_1, _ = \
                    decode_task_var_windowed(
                        X_t=X_1[[feature_idx], :, :], y=y_1,
                        time_bins=time_bins, decoding_time_bins=decoding_time_bins,
                        decoding_bin_width=decoding_bin_width,
                        num_shuffle=num_shuffle,
                        return_weights=return_weights, weight_type=weight_type,
                        num_cv=num_cv, clf=clf_pipeline,
                        run_in_parallel=run_in_parallel,
                        num_cv_repeats=num_cv_repeats)
                f_i_accuracy_per_window_2, accuracy_per_window_null_2, weights_per_window_2, _, weight_names, model_per_window_2, _ = decode_task_var_windowed(
                    X_t=X_2[[feature_idx], :, :], y=y_2,
                    time_bins=time_bins, decoding_time_bins=decoding_time_bins, decoding_bin_width=decoding_bin_width,
                    num_shuffle=num_shuffle,
                    return_weights=return_weights, weight_type=weight_type,
                    num_cv=num_cv, clf=clf_pipeline,
                    run_in_parallel=run_in_parallel,
                    num_cv_repeats=num_cv_repeats)

                accuracy_per_window_1.append(f_i_accuracy_per_window_1)
                accuracy_per_window_2.append(f_i_accuracy_per_window_2)

            # Also do a fit with all features to compare more easily...
            f_i_accuracy_per_window_1, accuracy_per_window_null_1, weights_per_window_1, _, weight_names, model_per_window_1, _ = \
                decode_task_var_windowed(
                    X_t=X_1, y=y_1,
                    time_bins=time_bins, decoding_time_bins=decoding_time_bins,
                    decoding_bin_width=decoding_bin_width,
                    num_shuffle=num_shuffle,
                    return_weights=return_weights, weight_type=weight_type,
                    num_cv=num_cv, clf=clf_pipeline,
                    run_in_parallel=run_in_parallel,
                    num_cv_repeats=num_cv_repeats)
            f_i_accuracy_per_window_2, accuracy_per_window_null_2, weights_per_window_2, _, weight_names, model_per_window_2, _ = decode_task_var_windowed(
                X_t=X_2, y=y_2,
                time_bins=time_bins, decoding_time_bins=decoding_time_bins, decoding_bin_width=decoding_bin_width,
                num_shuffle=num_shuffle,
                return_weights=return_weights, weight_type=weight_type,
                num_cv=num_cv, clf=clf_pipeline,
                run_in_parallel=run_in_parallel,
                num_cv_repeats=num_cv_repeats)

            accuracy_per_window_1.append(f_i_accuracy_per_window_1)
            accuracy_per_window_2.append(f_i_accuracy_per_window_2)
            accuracy_per_window_3 = np.nan
            accuracy_per_window_null_3 = np.nan
            weights_per_window_3 = np.nan

            # Train-test indices not needed for single-region decoding analysis
            train_test_indices_1 = np.nan
            train_test_indices_2 = np.nan
            train_test_indices_3 = np.nan

        else:
            accuracy_per_window_1, accuracy_per_window_null_1, weights_per_window_1, _, weight_names, model_per_window_1, train_test_indices_1 = \
                decode_task_var_windowed(
                    X_t=X_1, y=y_1,
                    time_bins=time_bins, decoding_time_bins=decoding_time_bins, decoding_bin_width=decoding_bin_width,
                    num_shuffle=num_shuffle,
                    return_weights=return_weights, weight_type=weight_type,
                    num_cv=num_cv, clf=clf_pipeline,
                    run_in_parallel=run_in_parallel,
                    num_cv_repeats=num_cv_repeats)

            accuracy_per_window_2, accuracy_per_window_null_2, weights_per_window_2, _, weight_names, model_per_window_2, train_test_indices_2 = decode_task_var_windowed(
                X_t=X_2, y=y_2,
                time_bins=time_bins, decoding_time_bins=decoding_time_bins, decoding_bin_width=decoding_bin_width,
                num_shuffle=num_shuffle,
                return_weights=return_weights, weight_type=weight_type,
                num_cv=num_cv, clf=clf_pipeline,
                run_in_parallel=run_in_parallel,
                num_cv_repeats=num_cv_repeats)
            
            # Combine state 1 and 2
            X_3 = np.concatenate([X_1, X_2], axis=1)
            y_3 = np.concatenate([y_1, y_2])
            accuracy_per_window_3, accuracy_per_window_null_3, weights_per_window_3, _, weight_names, model_per_window_3, train_test_indices_3 = decode_task_var_windowed(
                X_t=X_3, y=y_3,
                time_bins=time_bins, decoding_time_bins=decoding_time_bins, decoding_bin_width=decoding_bin_width,
                num_shuffle=num_shuffle,
                return_weights=return_weights, weight_type=weight_type,
                num_cv=num_cv, clf=clf_pipeline,
                run_in_parallel=run_in_parallel,
                num_cv_repeats=num_cv_repeats)
            
            if do_across_window_decoding: 
                # Across window decoding 
                # Train model on time window i, then test the window on activity in time window j
                across_window_matrix_1 = decode_task_var_across_window(X_t=X_1, y=y_1, 
                                                                       time_bins=time_bins, decoding_time_bins=decoding_time_bins,
                             decoding_bin_width=decoding_bin_width,
                             clf=clf_pipeline, num_cv=num_cv, num_cv_repeats=num_cv_repeats)
                
                across_window_matrix_2 = decode_task_var_across_window(X_t=X_2, y=y_2, 
                                                                       time_bins=time_bins, decoding_time_bins=decoding_time_bins,
                             decoding_bin_width=decoding_bin_width,
                             clf=clf_pipeline, num_cv=num_cv, num_cv_repeats=num_cv_repeats)

        accuracy_per_window_train_1_test_2 = np.nan
        accuracy_per_window_null_train_1_test_2 = np.nan

    else:
        accuracy_per_window_1 = np.nan
        accuracy_per_window_null_1 = np.nan
        accuracy_per_window_2 = np.nan
        accuracy_per_window_3 = np.nan 
        accuracy_per_window_null_2 = np.nan
        accuracy_per_window_null_3 = np.nan
        weights_per_window_1 = np.nan
        weights_per_window_2 = np.nan
        weights_per_window_3 = np.nan
        decoding_time_bins = np.nan
        accuracy_per_window_train_1_test_2 = np.nan
        accuracy_per_window_null_train_1_test_2 = np.nan
        weight_names = np.nan
        train_test_indices_1 = np.nan
        train_test_indices_2 = np.nan
        train_test_indices_3 = np.nan

    # Put together decoder results into a dict
    decoder_results = {}
    # decoder_results['choice_reward_matrix_1'] = choice_reward_matrix_1
    # decoder_results['choice_reward_matrix_2'] = choice_reward_matrix_2
    decoder_results['contigency_table_1'] = contigency_table_1
    decoder_results['contigency_table_2'] = contigency_table_2
    decoder_results['accuracy_per_window_1'] = accuracy_per_window_1
    decoder_results['accuracy_per_window_2'] = accuracy_per_window_2
    decoder_results['accuracy_per_window_3'] = accuracy_per_window_3
    decoder_results['accuracy_per_window_null_1'] = accuracy_per_window_null_1
    decoder_results['accuracy_per_window_null_2'] = accuracy_per_window_null_2
    decoder_results['accuracy_per_window_null_3'] = accuracy_per_window_null_3
    decoder_results['weights_per_window_1'] = weights_per_window_1
    decoder_results['weights_per_window_2'] = weights_per_window_2
    decoder_results['weights_per_window_3'] = weights_per_window_3
    decoder_results['weight_names'] = weight_names
    decoder_results['decoding_time_bins'] = decoding_time_bins
    decoder_results['accuracy_per_window_train_1_test_2'] = accuracy_per_window_train_1_test_2
    decoder_results['accuracy_per_window_null_train_1_test_2'] = accuracy_per_window_null_train_1_test_2
    decoder_results['balancing_method'] = balancing_method
    decoder_results['activity_to_use'] = activity_to_use
    decoder_results['decoding_bin_width'] = decoding_bin_width
    decoder_results['state_p_threshold'] = state_p_threshold
    decoder_results['train_test_indices_1'] = train_test_indices_1
    decoder_results['train_test_indices_2'] = train_test_indices_2
    decoder_results['train_test_indices_3'] = train_test_indices_3
    decoder_results['df_1'] = df_1
    decoder_results['df_2'] = df_2
    decoder_results['df_1_columns'] = df_1.columns
    decoder_results['df_2_columns'] = df_2.columns


    if do_across_window_decoding:
        decoder_results['across_window_matrix_1'] = across_window_matrix_1
        decoder_results['across_window_matrix_2'] = across_window_matrix_2

    return decoder_results

def get_weights_in_brain_space(spatial_weights, model_weights, summary_method='mean',
                               scale_s_weight_method='sum-to-one'):

    width, height, num_components = np.shape(spatial_weights)
    weight_matrix = np.zeros((width, height, num_components)) + np.nan

    for component_idx in np.arange(num_components):
        locanmf_spatial_component = spatial_weights[:, :, component_idx].astype(float)
        locanmf_spatial_component[locanmf_spatial_component == 0] = np.nan
        if scale_s_weight_method == 'sum-to-one':
            locanmf_spatial_component = locanmf_spatial_component / np.nansum(locanmf_spatial_component)

        weight_matrix[:, :, component_idx] = locanmf_spatial_component * model_weights[component_idx]

    if summary_method == 'mean':
        weight_matrix = np.nanmean(weight_matrix, axis=2)
    elif summary_method == 'sum':
        weight_matrix = np.nansum(weight_matrix, axis=2)
    elif summary_method == 'max':
        weight_matrix = np.nanmax(weight_matrix, axis=2)

    # weight_matrix[weight_matrix == 0] = np.nan

    return weight_matrix


def make_contigency_table(behaviour_df, row_name, col_name, row_vals=[0, 1],
                          col_vals=[0, 1]):

    num_row = len(row_vals)
    num_col = len(col_vals)

    contigency_table = np.zeros((num_row, num_col)) + np.nan

    for row_idx, row_val in enumerate(row_vals):
        for col_idx, col_val in enumerate(col_vals):

            subset_df = behaviour_df.loc[
                (behaviour_df[row_name] == row_val) &
                (behaviour_df[col_name] == col_val)
            ]

            trial_count = len(subset_df)

            contigency_table[row_idx, col_idx] = trial_count



    return contigency_table


def feature_permutation_test(X, y, clf, cv_splitter, feature_groups=None,
                             num_permutation=100, include_full_model=False,
                             num_cv=5):

    num_trials, num_features = np.shape(X)
    if feature_groups is None:
        feature_groups = np.arange(num_features)

    # Get the trial permutation indices (can just use the same for each feature...)
    trial_permutation_indices = [np.random.permutation(np.arange(num_trials)) for x in np.arange(num_permutation)]

    # permutation accuracy per group
    num_groups = len(np.unique(feature_groups))
    accuracy_per_group_permuted = np.zeros((num_groups, ))

    # Make the different feature sets
    for group_idx in np.arange(num_groups):

        feature_indices = np.where(feature_groups == group_idx)[0]

        # run classifications in parallel
        X_list = []
        for permutation_idx in np.arange(num_permutation):
            X_perm = X.copy()
            if len(feature_indices) == 1:
                X_perm[:, feature_indices] = X[trial_permutation_indices[permutation_idx], feature_indices].reshape(-1, 1)
            else:
                X_perm[:, feature_indices] = X[trial_permutation_indices[permutation_idx], feature_indices]
            X_list.append(X_perm)

        parallel_results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(sklselection.cross_val_score)(
                clone(clf), X_temp, y, cv=cv_splitter,
            ) for X_temp in X_list
        )

        accuracy_per_group_permuted[group_idx] = np.mean(parallel_results)

    if include_full_model:
        full_model_accuracy = sklselection.cross_val_score(clone(clf), X, y, cv=cv_splitter)
        return accuracy_per_group_permuted, full_model_accuracy
    else:
        return accuracy_per_group_permuted



def get_cross_validated_projections(decoding_results, neural_data, neural_data_time_bins, 
                                    state_number=1):
    """_summary_

    Parameters
    ----------
    decoding_results : _type_
        _description_
    neural_data : _type_
        _description_
    neural_data_time_bins : _type_
        _description_
    state_number : int, optional
        which state to project from, by default 1
        1 : stochastic state 
        2 : biased state

    Returns
    -------
    _type_
        _description_
    """


    num_cv = np.shape(decoding_results['train_test_indices_1'])[3]
    num_trials = np.shape(neural_data)[1]

    weight_name = 'weights_per_window_%.f' % state_number
    train_test_indices_name = 'train_test_indices_%.f' % state_number

    weights_per_window = decoding_results[weight_name]
    num_windows = np.shape(weights_per_window)[1]

    train_test_indices = decoding_results[train_test_indices_name]

    df = decoding_results['df_%.f' % state_number]
    df_actual_trials = df[:, 3]

    decoding_time_bins = decoding_results['decoding_time_bins']

    projections = np.zeros((num_trials, num_windows, num_cv)) + np.nan

    for cv_idx in np.arange(num_cv):

        for window_idx in np.arange(num_windows):
                                    
            window_start = decoding_time_bins[window_idx]
            window_end = decoding_time_bins[window_idx+1]
            subset_time_index = np.where((neural_data_time_bins >= window_start) & 
                                         (neural_data_time_bins <= window_end))[0]
            subset_activity = np.mean(neural_data[:, :, subset_time_index], axis=2)  # features X trials

            # do some z-scoring 
            subset_activity = (subset_activity - np.mean(subset_activity, axis=1).reshape(-1, 1)) / np.std(subset_activity, axis=1).reshape(-1, 1)
        

            projections[:, window_idx, cv_idx] = np.matmul(weights_per_window[0, window_idx, :, cv_idx], subset_activity)

            # remove the trials that were used for training
            trials_used_in_training = np.where(train_test_indices[0, :, window_idx, cv_idx] == 1)[0]
            trials_to_remove = df_actual_trials[trials_used_in_training].astype(int)

            projections[trials_to_remove, window_idx, cv_idx] = np.nan

    return projections


if __name__ == "__main__":
    app()


