import pdb
from pathlib import Path

from loguru import logger
from matplotlib.style.core import available
from pexpect import searcher_string
from tqdm import tqdm
from rich.progress import track
import typer

from config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, DATASET_YAML, RC_WF_PROP, PROJ_ROOT, NPY_MATLAB_DIR
from config import RISKYCHOICE_SUBJECTS
import os
import scipy.io as spio
import pandas as pd
import glob
import json
import numpy as np

# for tif to mat conversion for locaNMF processing
from ScanImageTiffReader import ScanImageTiffReader
import hdf5storage

# date conversion
import datetime

# copying files
import shutil

# Alignment
import cv2
import scipy.signal as ssignal

# load matlab data
import h5py
import hdf5storage as h5s


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    params = mp_util.read_params(DATASET_YAML, process_name=None)
    mp_util.print_params(params)

    for process in params['processes_to_run']:

        if process == 'tif_to_mat':
            logger.info("Converting tif files to mat files")
            convert_tif(params['tif_paths'], output_format='.mat')
            logger.info('Conversion complete')
        elif process == 'tif_to_npy':
            logger.info("Converting tif files to npy files")
            convert_tif(params['tif_paths'], output_format='.npy',
                        delete_tiffs=params['delete_tiffs'],
                        overwrite_existing=params['overwrite_existing'])
            logger.info('Conversion complete')

        elif process == 'copy_file':
            logger.info('Copying files from server to computer')

            copy_file(source_folder=SERVER_DATA_DIR, target_folder=INTERIM_DATA_DIR,
                      subject=params['subject'], protocol=params['protocol'],
                      filetype=params['filetype'], files_to_get=params['files_to_get'])

        elif process == 'delete_file':

            logger.info('Deleting files from computer')
            delete_file(target_folder=INTERIM_DATA_DIR,
                        subject=params['subject'], protocol=params['protocol'],
                        filetype=params['filetype']
                        )




    logger.success("Processing dataset complete.")
    # -----------------------------------------


@app.command()
def get_riskychoice_data():
    """
    Converts Oli's all_data_nodecon.mat files into 3 files used for downstream analysis 
    - behaviour.csv 
    - pixel_ave.mat
    _ motSVD.npy
    Full pipeline:
    1. Run copy_file from mp.dataset to with filetype: 'riskyChoiceWideField'
    2. Run this 
    3. Do locaNMF / pixel average decoding
    """

    # check if the processed data is already there
    pixel_mat_search_results = glob.glob(os.path.join(INTERIM_DATA_DIR, '*pixel_ave.mat'))
    behaviour_csv_search_results = glob.glob(os.path.join(INTERIM_DATA_DIR, '*behaviour.csv'))

    pdb.set_trace()

    if (len(pixel_mat_search_results) > 0) & (len(behaviour_csv_search_results) > 0):
        print('Processed pixel averaged and behaviour data found, skipping data extraction step')
        return 0
    else:
        print('Performing data extraction step')

    import matlab.engine  # import matlab engine early to avoid segmentation fault
    allen_map_fpath = os.path.join(INTERIM_DATA_DIR, 'allenDorsalMapSM.mat')
    eng = matlab.engine.start_matlab()
    eng.cd(str(PROJ_ROOT), nargout=0)

    # add npy-matlab to path
    eng.addpath(NPY_MATLAB_DIR, nargout=0)

    file_pattern = 'all_data_nodecon_updated_hemocorr_filter.mat'

    source_folder = INTERIM_DATA_DIR

    for subjid in RISKYCHOICE_SUBJECTS:

        # session_data_fpaths = glob.glob(os.path.join(source_folder, f'{subjid}*all_data_nodecon_withoutFiltering.mat'))
        session_data_fpaths = glob.glob(os.path.join(source_folder, f'{subjid}*{file_pattern}'))
        if len(session_data_fpaths) == 0:
            logger.info("WARNING: NO FILES FOR FOR SUBJECT")
        
        for s_d_fpath in track(session_data_fpaths):

            date = os.path.basename(s_d_fpath).split('_')[1]

            # roi_mask_fpath = os.path.join(source_folder, f'{subjid}_{date}_allen_roimask.mat')

            # just call the same .mat mask, they are the same across all recordings since they are
            # already registered
            roi_mask_fpath = os.path.join(source_folder, 'oli_generic_allen_roimask.mat')
            if not os.path.exists(roi_mask_fpath):
                # another location to find the allen roimask
                roi_mask_fpath = os.path.join(source_folder, 'no_decon', f'{subjid}_{date}_allen_roimask.mat')

            pixel_ave_savepath = os.path.join(source_folder, f'{subjid}_{date}_pixel_ave.mat')
            behaviour_data_savepath = os.path.join(source_folder, f'{subjid}_{date}_behaviour.csv')
            face_data_savepath = os.path.join(source_folder, f'{subjid}_{date}_motSVD.npy')
            if (not os.path.exists(behaviour_data_savepath)) or (not os.path.exists(pixel_ave_savepath)):
                eng.getRiskyChoiceData(s_d_fpath, roi_mask_fpath, allen_map_fpath, pixel_ave_savepath,
                                    behaviour_data_savepath, nargout=0)

            try:
                if not os.path.exists(face_data_savepath):
                    eng.getRiskyChoiceFaceData(s_d_fpath, face_data_savepath, nargout=0)
            except:
                logger.info(f'No face data for {subjid} {date}')

    eng.quit()



def load_data(data_type='monkeyMP', camera_fov='frontcam', output_option='df', subject=None, subject2=None, datasetID=None,
              include_miss_trials=False, protocol=None, sessid=None, date=None, exclude_ignore_trials=True,
              include_pixel_data=False, loc_thresh=70, Vc_Uc_datatype='mat', custom_folder_fstring=None,
              custom_dir=None, alignment_window=[-3, 5], resample_time_bins=None, resample_fs=None, decoding_params=None, 
              aligned_to='sound', within_date_session_idx=0, min_trials=10, max_consecutive_misses=15,
              sessids_to_ignore=None, model_name=None, y_target='reward'):
    """
    Main function for loading data

    Parameters
    ----------
    data_type : str
        which type of data to load, supported ones include
        'monkeyMP' : monkey matching pennies data
    camera_fov : str
        which camera field of view to get
        only applies when loading facemap data
    output_option: str
        the output type to get
    subject : str
        subject name (mouse/monkey)
    subject2 : str
        subject name of the oppponent playing against subject (only applies to multiplayer games)
    datasetID: int
        which dataset to load for monkey data
        only applies for data_type = 'monkeyMP'
    include_miss_trials: bool
        whether to include miss trials in behaviour and neural data
    protocol : str
        name of the protocol
    sessid : str
        session id to get for particular protocol
        only applies to Joanna's data
    date : str
        date of the experiment to get
        set to None to load all dates
    exclude_ignore_trials : bool
        whether to exclude ignore trials
        only applies to Joanna's data
    include_pixel_data : bool
        whether or not to include widefield pixel data
        only applicable when loading locaNMF data
    loc_thresh : int
        localisation threshold to load for locaNMF data
    Vc_Uc_datatype : str
        data format of where you stored your Vc and Uc
        either npy or mat
    min_trials : int 
        minimum number of trials to include an experiment's data
        this is to exclude sessions where the experiment was terminated early because something went wrong
    max_consecutive_misses : int
        number of consecutive miss trials before excluding the rest of the sessions's trials 
        this is to exclude trials where the experiment should have been ended earlier.
    Returns
    --------
    data : dict, or other
        your data object
    """


    if data_type == 'riskychoice-decoding':

        data = dict()
        # find pixel ave files
        pixel_ave_fpaths = glob.glob(os.path.join(INTERIM_DATA_DIR, '{subject}*pixel_ave.mat'.format(subject=subject)))
        print('Number of pixel ave files found: %.f' % len(pixel_ave_fpaths))

        pixel_data_store = []
        behaviour_data_store = []

        for px_a_fpath in pixel_ave_fpaths:

            pixel_data = spio.loadmat(px_a_fpath)['pixel_ave']  # area x trial x timepoints

            date = os.path.basename(px_a_fpath).split('_')[1]
            behaviour_data_fpath = os.path.join(INTERIM_DATA_DIR,
                                                f'{subject}_{date}_behaviour.csv')

            behaviour_df = pd.read_csv(behaviour_data_fpath)

            # exclude because it has 210 frames for some reason... (Oli suggested)
            if os.path.basename(px_a_fpath) == 'OLI-M-0021_20230906_pixel_ave.mat':
                continue

            behaviour_data_store.append(behaviour_df)


            pixel_data_store.append(pixel_data)
            
        data['time_rel_stim'] = np.arange(-RC_WF_PROP['pre_stim'], RC_WF_PROP['post_stim'], 1/RC_WF_PROP['fs'])  
        data['pixel_ave'] = np.concatenate(pixel_data_store, axis=1)

        behaviour_df_combined = pd.concat(behaviour_data_store)
        behaviour_df_combined['date'] = behaviour_df_combined['SessionDate']
        data['behaviour_df'] = behaviour_df_combined

    elif data_type == 'riskychoice-decoding-from-face':

        data = dict()
        # find pixel ave files
        mot_svd_fpaths = glob.glob(os.path.join(INTERIM_DATA_DIR, '{subject}*motSVD.npy'.format(subject=subject)))

        mot_svd_data_store = []
        behaviour_data_store = []

        for msvd_fpath in mot_svd_fpaths:
            mot_svd_data = np.load(msvd_fpath)  # SVD x timepoints x trial

            date = os.path.basename(msvd_fpath).split('_')[1]
            behaviour_data_fpath = os.path.join(INTERIM_DATA_DIR,
                                                f'{subject}_{date}_behaviour.csv')

            behaviour_df = pd.read_csv(behaviour_data_fpath)

            behaviour_data_store.append(behaviour_df)
            mot_svd_data_store.append(mot_svd_data)

        data['time_rel_stim'] = np.arange(-2, 5, 1 / 90)  # Oli told me this
        num_svd_components_to_keep = 100 # TEMP: noticed that some data has 299 components for some reason
        mot_svd_data_store = [x[0:num_svd_components_to_keep, :, :] for x in mot_svd_data_store]
        mot_svd_combined = np.concatenate(mot_svd_data_store, axis=2)
        data['mot_svd'] = np.swapaxes(mot_svd_combined, 1, 2)  # change axes so it is SVD x trial x timepoints
        behaviour_df_combined = pd.concat(behaviour_data_store)
        behaviour_df_combined['date'] = behaviour_df_combined['SessionDate']
        data['behaviour_df'] = behaviour_df_combined
    
    elif data_type == 'riskychoice-decoding-from-sleap':

        data = dict()

        subject_folder = os.path.join(RISKYCHOICE_WIDEFIELD_DATA_DIR, subject)

        subject_files = np.sort(
            glob.glob(os.path.join(INTERIM_DATA_DIR, f'{subject}*_sleap.mat'))
        )

        all_date_data = [] 
        subset_dates = []
        for fpath in subject_files: 

            date_data = spio.loadmat(fpath)['roi_trial_time']
            all_date_data.append(date_data)
            subset_dates.append(int(os.path.basename(fpath).split('_')[1]))
        
        ext_folder = '/media/timsit/X9Pro/'  # NOTE: need this, for some reason INTERIM_DATA_DIR on my computer doesn't work... may be because I have some old files..
        df_files = np.sort(glob.glob(os.path.join(ext_folder, f'{subject}*_behaviour.csv')))
        all_date_df = []
        for fpath in df_files: 

            date_df = pd.read_csv(fpath)
            all_date_df.append(date_df)
        

        all_date_df = pd.concat(all_date_df)
        all_date_df_subset = all_date_df.loc[
            all_date_df['SessionDate'].isin(subset_dates)
        ]

        data['time_rel_stim'] = np.arange(-1, 5, 1 / 30)  # TODO: double check with Oli
        data['sleap'] = np.concatenate(all_date_data, axis=1)
        all_date_df_subset['date'] = all_date_df_subset['SessionDate']
        data['behaviour_df'] = all_date_df_subset
        

    elif data_type == 'riskychoice-decoding-behaviour_df':

        data = dict()
        # find pixel ave files
        pixel_ave_fpaths = glob.glob(os.path.join(INTERIM_DATA_DIR, '{subject}*pixel_ave.mat'.format(subject=subject)))

        behaviour_data_store = []

        for px_a_fpath in pixel_ave_fpaths:

            date = os.path.basename(px_a_fpath).split('_')[1]
            behaviour_data_fpath = os.path.join(INTERIM_DATA_DIR,
                                                f'{subject}_{date}_behaviour.csv')

            behaviour_df = pd.read_csv(behaviour_data_fpath)
            behaviour_data_store.append(behaviour_df)


        data = pd.concat(behaviour_data_store)

    elif data_type == 'riskychoice-decoding-results':

        data = {}
        y_target = decoding_params['y_target']
        feature_used = decoding_params['feature_used']


        if subject is None:
            from config import RISKYCHOICE_SUBJECTS
            subject = RISKYCHOICE_SUBJECTS

        for subject_idx, subject_id in enumerate(subject):
            decoding_result_fpath = os.path.join(PROCESSED_DATA_DIR, 'decoding',
                                                 '{subject}_all_None_locaNMF_decoding_{y_target}_using_{feature_used}_resample_states_{resample_states}_balance_states_trialcounts_{balance_states_trialcounts}.npz'.format(
                                                     subject=subject_id, y_target=y_target, feature_used=feature_used,
                                                     resample_states=decoding_params['resample_states'],
                                                     balance_states_trialcounts=decoding_params['balance_states_trialcounts']))

            decoding_result = np.load(decoding_result_fpath)

            data[subject_id] = decoding_result
    
    elif data_type in ['decoding_results', 'decoding-results', 'decoding-result', 'decoding_result']:

        if type(subject) is list:
            data = {}
            for subject_idx, subject_id in enumerate(subject):
                decoding_result_fpath = os.path.join(PROCESSED_DATA_DIR, 'decoding',
                                                 '{subject}_all_None_locaNMF_decoding_{y_target}_using_{feature_used}_resample_states_{resample_states}_balance_states_trialcounts_{balance_states_trialcounts}.npz'.format(
                                                     subject=subject_id, y_target=y_target, feature_used=feature_used,
                                                     resample_states=decoding_params['resample_states'],
                                                     balance_states_trialcounts=decoding_params['balance_states_trialcounts']))

            decoding_result = np.load(decoding_result_fpath)

            data[subject_id] = decoding_result
        else:
            model_fpath = glob.glob(os.path.join(PROCESSED_DATA_DIR, 'decoding',
                                                '%s*%s*%s.npz' % (subject, y_target, model_name)))[0]
            data = np.load(model_fpath, allow_pickle=True)
            

    else:
        Warning('Invalid data_type specified!')

    return data



def copy_file(source_folder, target_folder, subject, protocol, filetype='widefieldTIFs', files_to_get=None):


    if type(subject) is not list:
        subject = [subject]

    for subjid in subject:

        if filetype == 'widefieldTIFs' or filetype == 'widefieldTIFsAllen':

            if filetype == 'widefieldTIFs':
                source_fname = 'hemocorrected_signal_downsampled2.tiff'
                target_fname = 'hemocorrected_signal_downsampled2.tiff'
            elif filetype == 'widefieldTIFsAllen':
                source_fname = 'zdeltaff_downsampled2_allen_aligned_to_{subject}.tiff'.format(subject=subjid)
                target_fname = 'zdeltaff_downsampled2_allen_aligned_to_{subject}.tiff'.format(subject=subjid)


            source_widefield_folder = os.path.join(source_folder, 'widefield', 'joanna', 'preprocess', subjid)
            available_date_folder_names = [f.path for f in os.scandir(source_widefield_folder) if f.is_dir() ]

            logger.info('Copying files...')

            for date_folder_path in tqdm(available_date_folder_names):
                date_folder_name = os.path.basename(date_folder_path)
                # convert the date format into the format with only numbers
                date = datetime.datetime.strptime(date_folder_name, '%d-%b-%Y').strftime('%Y%m%d')
                if filetype == 'widefieldTIFs':
                    source_fpath = os.path.join(date_folder_path, 'MatchingPennies', source_fname)
                elif filetype == 'widefieldTIFsAllen':
                    source_fpath = os.path.join(date_folder_path, 'MatchingPennies', 'allen', source_fname)

                if not os.path.isfile(source_fpath):
                    continue

                target_exp_folder = os.path.join(target_folder, 'widefield_{subjid}_{date}_{protocol}'.format(
                    subjid=subjid, date=date, protocol=protocol,
                ))

                if not os.path.isdir(target_exp_folder):
                    os.makedirs(target_exp_folder)

                target_path = os.path.join(target_exp_folder, target_fname)

                if not os.path.exists(target_path):
                    # copy file
                    shutil.copyfile(source_fpath, target_path)

        elif filetype == 'riskyChoiceWideField':
            source_widefield_folder = os.path.join(source_folder, 'widefield', 'oli', 'pre_processed_data', subjid)
            available_date_folder_names = [f.path for f in os.scandir(source_widefield_folder) if f.is_dir()]
            available_date_folder_names = [x for x in available_date_folder_names if os.path.basename(x).isdigit()]
            # TODO: only subset folders with all numbers (dates)

            if files_to_get is None:
                files_to_get = ['all_data_nodecon.mat', 'allen_roimask.mat']

            for source_fname in files_to_get:

                from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
                with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task("Copying %s files..." % source_fname,
                                             total=len(available_date_folder_names))

                    for date_folder in available_date_folder_names:
                        date = os.path.basename(date_folder)
                        target_fname = '{subject}_{date}_{source_fname}'.format(subject=subjid,
                                                                                date=date,
                                                                                source_fname=source_fname)

                        source_fpath = os.path.join(date_folder, source_fname)

                        # Try the no_decon folder as well
                        if not os.path.exists(source_fpath):
                            source_fpath = os.path.join(date_folder, 'no_decon', source_fname)

                        if not os.path.isfile(source_fpath):
                            progress.update(task, advance=1)
                            continue

                        if not os.path.isdir(target_folder):
                            os.makedirs(target_folder)

                        target_path = os.path.join(target_folder, target_fname)

                        if not os.path.exists(target_path):
                            # copy file
                            shutil.copyfile(source_fpath, target_path)

                        progress.update(task, advance=1)


    return None

def delete_file(target_folder=INTERIM_DATA_DIR,
                subject='JOA-M-0008',
                protocol='MatchingPennies_WF',
                filetype='tiffs'):


    if type(subject) is not list:
        subject = [subject]

    for subjid in subject:

        search_str = '*{subjid}_*{protocol}'.format(subjid=subjid, protocol=protocol)
        target_subfolders = glob.glob(os.path.join(target_folder, search_str))

        if len(target_subfolders) == 0:
            logger.info('No subfolders found, check your parameters')

        for t_subfolder in target_subfolders:

            if filetype == 'tiffs':

                files_to_delete = glob.glob(os.path.join(t_subfolder, '*.tiff'))

            for fpath in files_to_delete:
                logger.info('Deleting %s' % os.path.basename(fpath))
                os.remove(fpath)

    return None

def get_wf_shared_regions(wf_session_data_list=None, subject=None, dates=None,
                          protocol='MatchingPennies_WF'):

    if wf_session_data_list is not None:
        area_names_per_session = [x['pixel_area_names'] for x in wf_session_data_list]
        shared_areas = set.intersection(*[set(x) for x in area_names_per_session])
        shared_areas = np.array(list(shared_areas))
    else:
        wf_session_data_list = load_data(subject=subject, date=date, protocol=protocol,
                                        data_type='locaNMF', include_miss_trials=False,
                                        include_pixel_data=True)

    return shared_areas


def combine_wf_data_across_sessions(wf_session_data_list,
                                    fields_to_combine=['behaviour_df'],
                                    shared_areas=None):


    combined_data = {}

    for field_name in fields_to_combine:

        if field_name == 'behaviour_df':
            combined_data['behaviour_df'] = pd.concat([x['behaviour_df'] for x in wf_session_data_list])

        elif field_name == 'pixel_ave_lick_aligned':

            pixel_ave_lick_aligned_list = []
            area_names_per_session = []
            for session_data in wf_session_data_list:

                session_areas = session_data['pixel_area_names']
                subset_index = np.isin(session_areas, shared_areas)
                area_names_per_session.append(session_areas[subset_index])

                subset_pixel_ave_lick_aligned = session_data['pixel_ave_lick_aligned'][subset_index, :, :]
                pixel_ave_lick_aligned_list.append(subset_pixel_ave_lick_aligned)

            pixel_ave_lick_aligned = np.concatenate(pixel_ave_lick_aligned_list, axis=1)

            subset_index = np.sum(np.isnan(pixel_ave_lick_aligned), axis=(1, 2)) == 0
            combined_data['pixel_ave_lick_aligned'] = pixel_ave_lick_aligned[subset_index, :, :]
            combined_data['areaNames'] = shared_areas[subset_index]

        elif field_name == 'lick_time_bins':

            combined_data['lick_time_bins'] = wf_session_data_list[0]['lick_time_bins']

    return combined_data


def subset_behaviour_data(behaviour_data, cond_dict):
    """Subset behaviour based on specified conditions

    Parameters
    ----------
    behaviour_data : _type_
        _description_
    cond_dict : _type_
        _description_

    Returns
    -------
    subset_df : pandas dataframe
        a subset of behaviour_data
    trial_indices : numpy ndarray 
        subset of the trial indices of the original behaviour data
        this can be used to subset other data (eg. neural activity)
    """
    
    behaviour_data['indices'] = np.arange(len(behaviour_data))
    subset_df = behaviour_data.copy()

    for cond_col, cond_val in cond_dict.items():

        subset_df = subset_df.loc[subset_df[cond_col].isin(cond_val)]
        
    trial_indices = subset_df['indices']

    return subset_df, trial_indices


def align_face_data(movement_data, ttl_in_camera_frames=None, subject=None,
                    date=None, protocol=None, camera_fov='froncam', exp_df=None, 
                    window_width=[-3, 5], camera_fs=90,
                    resample_time_bins=None):
    """
    Align face data to ttl trigger (sound onset time) 
    
    Parameters
    ----------
    :param movement_data :
        numpy ndarray of shape (numTimePoints, numFeatures)
    :param ttl_in_camera_frames:
    :param window_width : list
    
    camera_fs : int
        camera sampling rate
    :return:
    """


    if np.ndim(movement_data) == 1:
        movement_data = movement_data.reshape(-1, 1)

    # if no ttl_in_camera_frames provide, then will load based on subejct date etc.
    if ttl_in_camera_frames is None:

        ttl_data = load_data(data_type='camera_TTL', subject=subject, date=date, 
                     camera_fov=camera_fov, custom_dir=None, protocol=protocol, within_date_session_idx=0)
        
        ttl_data_in_counts = ttl_data['ttl_in'].groupby(['fname']).count()['timestamp_frame'].values

        # See if wrong folder selected for the same date, if so, try the second folder 
        if np.all(np.abs(ttl_data_in_counts - len(exp_df)) > 10):
            ttl_data = load_data(data_type='camera_TTL', subject=subject, date=date, camera_fov=camera_fov,
                                        custom_dir=None, protocol=protocol, within_date_session_idx=1)
            ttl_data_in_counts = ttl_data['ttl_in'].groupby(['fname']).count()['timestamp_frame'].values

        if np.all(np.abs(ttl_data_in_counts - len(exp_df)) > 10):
            ttl_data = load_data(data_type='camera_TTL', subject=subject, date=date, camera_fov=camera_fov,
                                        custom_dir=None, protocol=protocol, within_date_session_idx=2)
               
        
        if (len(ttl_data['ttl_in']) != len(exp_df)) & (len(np.unique(ttl_data['ttl_in']['fname'])) > 1):
            for fname in np.unique(ttl_data['ttl_in']['fname']):
                subset_df = ttl_data['ttl_in'].loc[
                    ttl_data['ttl_in']['fname'] == fname
                ]

                if (len(subset_df) == len(exp_df)) or ((len(subset_df)+1) == len(exp_df)):
                    ttl_data['ttl_in'] = subset_df
                    ttl_data['ttl_out'] = ttl_data['ttl_out'].loc[
                        ttl_data['ttl_out']['fname'] == fname[0:-6] + 'out.csv'
                    ]


        ttl_in_camera_frames, camera_fs = get_camera_ttl_in_times(ttl_data['ttl_in'], 
                                                                     ttl_data['ttl_out'], 
                                                                     exp_df)
        

    num_features = np.shape(movement_data)[1]

    aligned_time_bins = np.arange(window_width[0], window_width[-1], 1 / camera_fs)
    num_aligned_bins = len(aligned_time_bins)
    num_trials = len(ttl_in_camera_frames)

    if resample_time_bins is not None:
        movement_aligned = np.zeros((num_features, num_trials, len(resample_time_bins)))
    else:
        movement_aligned = np.zeros((num_features, num_trials, num_aligned_bins))

    for trial_idx in np.arange(num_trials):
        ttl_frame = ttl_in_camera_frames[trial_idx]
        num_frames_prior = int(-window_width[0] * camera_fs)
        start_frame = ttl_frame - num_frames_prior
        num_frames_after = int(window_width[1] * camera_fs)
        end_frame = ttl_frame + num_frames_after

        if resample_time_bins is not None:
            original_time = np.linspace(window_width[0], window_width[1], end_frame - start_frame)
            for feature_idx in np.arange(num_features):
                if len(movement_data[start_frame:end_frame, feature_idx]) != len(original_time):
                    feature_intrp = np.nan
                else:
                    feature_intrp = np.interp(x=resample_time_bins, xp=original_time,
                                          fp=movement_data[start_frame:end_frame, feature_idx])
                movement_aligned[feature_idx, trial_idx, :] = feature_intrp
        else:
            movement_aligned[:, trial_idx, :] = movement_data[start_frame:end_frame, :].T


    return movement_aligned




def realign_from_sound_to_lick(components_per_trial, behaviour_df, lick_time_window=[-1.5, 1.5], 
                               time_from_sound_onset=None, fs=None,
                               return_lick_time_bins=False, exp_type='WF'):
    """Realigns sonud-aligned neural (or movement) data to lick-aligned neural (or movement) data

    Parameters
    ----------
    components_per_trial : numpy ndarray
        locaNMF or some other multidimensional activity, should have shape (numDim, numTrials, numTimePoint)
    behaviour_df : pandas dataframe
        pandas dataframe with the behaviour data, in order to extract lick times
    lick_time_window : list, optional
        _description_, by default [-1.5, 1.5]

    Returns
    -------
    components_per_trial_lick_aligned : numpy ndarray 
        locaNMF or some other multidimensional activity, with shape (numDim, numTrials, numLickTimePoints)
        numLickTimePoints is detedmined by lick_time_window and fs
    subset_trials : numpy ndarray 
        trial numbers that were subset, this is becaused missed trials have no defined lick times so they 
        are excluded after this alignment process
    """

    if fs is None:
        if exp_type == 'WF': 
            fs = MP_WF_PROP['fs']
        elif exp_type == 'photometry':
            fs = PHOTOMETRY_PROP['fs']


    num_regions = np.shape(components_per_trial)[0]

    # Also re-align to lick times
    subset_trials = np.where(behaviour_df['choice'] != 'M')[0]
    behaviour_df = behaviour_df.iloc[subset_trials]
    
    components_per_trial = components_per_trial[:, subset_trials, :]

    num_trials = len(subset_trials)

    lick_times_per_trial = behaviour_df['LickTimestamp'].values - behaviour_df['SoundTimestamp'].values
    lick_time_bins = np.arange(lick_time_window[0], lick_time_window[1], 1 / fs)

    components_per_trial_lick_aligned = np.zeros((num_regions, num_trials, len(lick_time_bins)))

    # if include_pixel_data:
    #     numPixelRegions = len(pixel_data['area_names'])
    #     pixel_ave_lick_aligned = np.zeros((numPixelRegions, num_trials, len(lick_time_bins)))
    if time_from_sound_onset is None:
        if exp_type == 'WF':
            time_from_sound_onset = np.arange(-MP_WF_PROP['pre_stim'], MP_WF_PROP['post_stim'],
                                      1 / fs)
        elif exp_type == 'photometry':
            time_from_sound_onset = np.arange(-PHOTOMETRY_PROP['pre_stim'], PHOTOMETRY_PROP['post_stim'],
                                      1 / fs)
    

    # Get lick aligned data
    for trial_idx in np.arange(num_trials):
        lick_time = lick_times_per_trial[trial_idx]
        window_idx_to_get = np.where(
            (time_from_sound_onset >= lick_time + lick_time_window[0]) &
            (time_from_sound_onset <= lick_time + lick_time_window[1])
        )[0]
        # TEMP FIX FOR RARE CASES WHERE THERE ARE 91 frames
        window_idx_to_get = window_idx_to_get[0:len(lick_time_bins)]

        components_per_trial_lick_aligned[:, trial_idx, :] = components_per_trial[:, trial_idx,
                                                                window_idx_to_get]
        
    if return_lick_time_bins:
        return components_per_trial_lick_aligned, subset_trials, lick_time_bins
    else:
        return components_per_trial_lick_aligned, subset_trials




if __name__ == "__main__":
    app()
