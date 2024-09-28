% Code 8. extract_pca_features.m --> TIME DOMAIN
% ------------------------------------------------------------------------
% This code extracts the pca components of the time series from all the 
% subjects. First we extract each of the windows of the participants,
% then we create a massive matrix of all the channels and participants
% one below the other, and finally we calculare the Principal Components
% Analysis (PCA) of this matrix to obtain the most amount of information
% in a smaller dimension.
% ------------------------------------------------------------------------

% Get the user's home directory
home_dir = getenv('HOME');

% Define the output directory relative to the user's home directory
output_dir = './output';

% Define log file path relative to the output directory
log_file = fullfile(output_dir, 'pca_time_log.txt');

% Create the output directory if it does not exist
if ~isfolder(output_dir)
    mkdir(output_dir);
end

% Delete the log file if it already exists to replace it for a new run
if isfile(log_file)
    delete(log_file);
end

% Start logging
diary(log_file);

% Display start message
disp('Starting PCA feature extraction in the Time domain...');

% Load directory information
load(fullfile(output_dir, 'dirinfo.mat'));

% Filter out '.' and '..' from dirinfo
filtered_dirinfo = dirinfo(~ismember({dirinfo.name}, {'.', '..'}));
subj_num = length(filtered_dirinfo);

if ispc  % Check if the OS is Windows
    path = 'F:\Sleep Phase Detection\Challenge_dataset\training'; % LOCAL path for Windows
elseif isunix  % Check if the OS is Linux/Unix
    path = '/storage/projects/ce903/2024_Jan_SU_Group_3/raw_data/challenge_dataset/training'
    % path = '/storage/projects/ce903/2024_Jan_SU_Group_3/code/23-24_CE903-SU_team03/Code/sample training data';
    % path = '/Users/ashwin/Projects/University of Essex/CE903/23-24_CE903-SU_team03/Code/sample training data'; % CLUSTER path for Linux/Unix
else
    error('Unsupported operating system');
end

% Assign values
win_sec = 30;                       % Window size in seconds
fsamp = 200;                        % Sampling frequency
win_samp = win_sec * fsamp;         % Size of window in frequency
channels = 1:13;                    % Channels selected
num_channels = length(channels);    % Number of channels selected

% Initialize storage
data_onesubj_win = cell(subj_num, 1);
data_allsubj_win = [];

% Loop to append all the windows per subject in a cell
for subj = 1 : subj_num
    disp(['Processing subject ', num2str(subj)]);
    
    % Load data based on OS
    storage = cell2mat(struct2cell(load([path, '/', filtered_dirinfo(subj).name, '/', filtered_dirinfo(subj).name, '.mat'], 'val')));
    
    subjectData = storage;

    % Check if all channels of the subjectData have std greater than eps
    if all(arrayfun(@(ch) std(subjectData(ch, :)) > eps, channels))
        data_onesubj_win{subj} = reshape_onesubj(subjectData, win_samp, channels);
        disp(['Processed subject ', num2str(subj)]);
    else
        disp(['Subject ', num2str(subj), ' excluded due to low variability in one or more channels.']);
    end
end

disp('-------------------------------------------')

% Loop to concatenate each participant's data into a massive matrix
for subj = 1 : subj_num
    if ~isempty(data_onesubj_win{subj})
        data_allsubj_win = [data_allsubj_win; data_onesubj_win{subj}];
        disp(['Including subject ', num2str(subj), ' in PCA.']);
    end
end

% Perform PCA
[coeff, score] = pca(data_allsubj_win, 'NumComponents', 200);

% Save PCA results to MAT file
save(fullfile(output_dir, 'matlab_pca_time.mat'), "score", "coeff", "-v7.3");

% Display a message indicating that the process has completed
disp('Process completed successfully.');

disp('-------------------------------------------')

% Stop logging
diary off;
