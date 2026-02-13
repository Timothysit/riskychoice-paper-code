%% Script to save sleap data from Oli 
mice_to_get = {'OLI-M-0021', 'IAA-1125879', 'IAA-1125880', ...
               'OLI-M-0022', 'OLI-M-0041', 'OLI-M-0042', 'OLI-M-0057'};
server_data_dir = '/mnt/ogma/widefield/oli/pre_processed_data';
data_save_dir = '/home/timsit/matchingp/data/interim/';
file_name = 'all_data_just_video_concat.mat';
sleap_data_field = 'video_trace_concat_zscored_session';

for mouse_dix = 1:length(mice_to_get)

    mouse_name = mice_to_get{mouse_dix};
    subject_folder = fullfile(server_data_dir, mouse_name);
    
    % Go through subfolders 
    subfolders = {dir(fullfile(subject_folder, '*')).name};
    subfolders = subfolders( ...
        [dir(fullfile(subject_folder, '*')).isdir] & ...
        cellfun(@(s) ~isempty(regexp(s, '^\d+$', 'once')) & ~ismember(s, {'.', '..'}), subfolders) ...
    );
    
    for subIdx = 1:length(subfolders)
        subFolder = subfolders{subIdx};
        sleap_data_fpath = fullfile(subject_folder, subFolder, file_name);
        if isfile(sleap_data_fpath)
            sleap_data = load(sleap_data_fpath);
            sleap_data_table = sleap_data.behaviour_data_table;
            data_field = sleap_data_table.(sleap_data_field);
            roi_time_trial = cat(3, data_field{:});        
            roi_trial_time = permute(roi_time_trial, [1 3 2]);
            
            save_name = strcat(mouse_name, '_', subFolder, '_sleap.mat');
            save(fullfile(data_save_dir, save_name), 'roi_trial_time');

        end


    end
    
    

end