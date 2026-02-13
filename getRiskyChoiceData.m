function getRiskyChoiceData(session_fpath, allen_mask_fpath, allen_map_fpath, pixel_ave_savepath, behaviour_savepath)
%GETRISKYCHOICEPIXELAVE Summary of this function goes here
%   Detailed explanation goes here
% get the locanmf code 
% addpath(genpath('/home/timsit/locaNMF-preprocess'))

%% Load data
session_data = load(session_fpath);
behaviour_data_table = session_data.behaviour_data_table;
% load roimask 
allen_data = load(allen_mask_fpath);

%% Concatenate all the trials together

% Oli sent me these numbers 
% But how do I get them automatically???
width = 70; 
height = 73; 

num_trials = size(behaviour_data_table, 1);
num_time_points_per_trial = size(behaviour_data_table.imaging{1}, 2);
num_pixels = size(behaviour_data_table.imaging{1}, 1);

all_data = zeros(num_pixels, num_time_points_per_trial, num_trials);

for trial_idx = 1:num_trials 
    
    all_data(:, :, trial_idx) = behaviour_data_table.imaging{trial_idx};

end

all_data_reshaped = reshape(all_data, num_pixels, num_trials*num_time_points_per_trial);
Y = reshape(all_data_reshaped, width, height, num_trials*num_time_points_per_trial);


%% Get braimask 
roi_mask = allen_data.roimask;
brain_mask_full = roi_mask(1:140, :, end);
brain_mask = transpose(downsample(transpose(downsample(brain_mask_full,2)),2));
brain_mask = double(brain_mask);
brain_mask(brain_mask == 0) = NaN;

%% Save atlas.mat from the roi mask 
%
sm_allen_data = load(allen_map_fpath);

allen_values = double(roi_mask(1:140, :, :));
% allen_values_downsampled = imresize(allen_values, 0.5);
allen_values_downsampled = downsample(allen_values, 2);
allen_values_downsampled = downsample(permute(allen_values_downsampled, [2 1 3]), 2);
allen_values_downsampled = permute(allen_values_downsampled, [2, 1, 3]); 

% set up the atlas matrix to fill in each brain region
% with a different integer number 
atlas = zeros(width, height);

% Last one is just the whole brain, so going to ignore that one
numRegions = size(allen_values_downsampled, 3)-1;
areanames = struct();
for region_number = 1:numRegions
    % mask = allen_values_downsampled(:, :, region_number) > 0.2;
    mask = allen_values_downsampled(:, :, region_number) == 1;
    atlas(mask) = region_number;

    % assign number is area names for now since I don't know 
    % what they actually correspond to
    areaSide = sm_allen_data.dorsalMaps.sides{region_number};
    areaName = sm_allen_data.dorsalMaps.labels{region_number};
    areaName = strrep(areaName, '-', '_');
    areaName = [areaName '_' areaSide];
    areanames.(areaName) = region_number;
end

% save(fullfile(data_folder_path, 'atlas.mat'), 'atlas', 'areanames');
%}

%% Get pixel average per region 
pixel_ave = zeros(numRegions, num_time_points_per_trial, num_trials);

for region_number = 1:numRegions
    % mask = allen_values_downsampled(:, :, region_number) > 0.2;
    mask = allen_values_downsampled(:, :, region_number) == 1;
    mask_flattened = mask(:);
    pixel_ave(region_number, :, :) = mean(all_data(mask_flattened, :, :), 1);

end
% reshape to region x trial x time points    
pixel_ave = permute(pixel_ave, [1, 3, 2]);
save(pixel_ave_savepath, 'pixel_ave');




%% Save behaviour data as csv

columns_to_get = {'MouseID', 'SessionDate', 'TrialNumber', ...
                   'LocRandom', 'FreqRewardMap', 'AirPuffCue', 'LeftChoice', ...
                   'LotteryChoice', 'LotteryProb', 'MissTrial', 'deltaEV', ...
                   'LotteryProb', 'SoundkHz', 'LotteryLocLeft', 'AirPuffLeft', 'TrialRewarded', ...
                   'RewardDelivered_ul', 'p_engaged_state', 'p_disengaged1_state', 'p_disengaged2_state'};

subset_table = behaviour_data_table(:, columns_to_get);

writetable(subset_table, behaviour_savepath);


end

