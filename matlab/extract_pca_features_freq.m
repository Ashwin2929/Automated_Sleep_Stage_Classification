% Extract PCA features in the Frequency domain (DONE IN BATCHES FOR MEMORY PURPOSES)

% Get the user's home directory
home_dir = getenv('HOME');

% Define the output directory relative to the user's home directory
output_dir = './output';

% Define log file path relative to the output directory
log_file = fullfile(output_dir, 'pca_freq_log.txt');

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
disp('Starting PCA feature extraction in the Frequency domain...');

% Load directory information
load(fullfile(output_dir, 'dirinfo.mat'));

% Check if the OS is Windows or Linux/Unix and set the path accordingly
if ispc  
    path = 'F:\Sleep Phase Detection\Challenge_dataset\training'; % LOCAL path for Windows
elseif isunix
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

% Initialize both spaces of storage required
num_subjects = size(dirinfo, 1) - 2;
data_onesubj_freq = cell(num_subjects, 1);
data_allsubj_freq = [];

% Loop to append all the windows per subject in a cell
for subj = 1:num_subjects
    subject_index = subj + 2;
    disp(['Processing subject ', num2str(subj)]);

    % Construct the full path for the subject data
    subject_path = fullfile(path, dirinfo(subject_index).name, [dirinfo(subject_index).name, '.mat']);
    subjectData = cell2mat(struct2cell(load(subject_path, 'val')));

    % Check if all channels of the subjectData have std greater than eps
    if all(arrayfun(@(ch) std(subjectData(ch,:)) > eps, channels))
        data_onesubj_freq{subj} = reshape_onesubj_freq(subjectData, win_samp, fsamp, channels);
        disp(['Processed subject ', num2str(subj)]);
    else
        disp(['Subject ', num2str(subj), ' excluded due to low variability in one or more channels.']);
    end
end

disp('-------------------------------------------')

% Loop to concatenate each participant's data into a massive matrix
for subj = 1:num_subjects
    if ~isempty(data_onesubj_freq{subj})
        data_allsubj_freq = [data_allsubj_freq; data_onesubj_freq{subj}];
        disp(['Including subject ', num2str(subj), ' in PCA.']);
    end
end

[coeff, score] = pca(data_allsubj_freq, 'NumComponents', 200);

% Save PCA results to MAT file
save(fullfile(output_dir, 'matlab_pca_freq.mat'), "score", "coeff", "-v7.3");

% Display a message indicating that the process has completed
disp('Process completed successfully.');

disp('-------------------------------------------')

% Stop logging
diary off;
