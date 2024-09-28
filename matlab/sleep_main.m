% Main script to call three separate MATLAB scripts sequentially with logging

% Clear workspace and command window
clear;
clc;

% Define log file path
log_file = './Code/matlab/output/log.txt';

% Create the output directory if it does not exist
if ~isfolder('./Code/matlab/output')
    mkdir('./Code/matlab/output');
end

% Delete the log file if it already exists to replace it for a new run
if isfile(log_file)
    delete(log_file);
end

% Start logging
diary(log_file);


% Display start message
disp('Starting the process...');

try
    % Run create_dirinfo script
    disp('Running create_dirinfo script...');
    run('./Code/matlab/create_dirinfo.m');
    disp('create_dirinfo script completed.');
catch ME
    disp(['Error in create_dirinfo script: ', ME.message]);
    % Log the error details
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
end

disp('-------------------------------------------')

try
    % Run extract_pca_features_freq script
    disp('Running extract_pca_features_freq script...');
    run('./Code/matlab/extract_pca_features_freq.m');
    disp('extract_pca_features_freq script completed.');
catch ME
    disp(['Error in extract_pca_features_freq script: ', ME.message]);
    % Log the error details
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
end

disp('-------------------------------------------')

try
    % Run extract_pca_features_time script
    disp('Running extract_pca_features_time script...');
    run('./Code/matlab/extract_pca_features_time.m');
    disp('extract_pca_features_time script completed.');
catch ME
    disp(['Error in extract_pca_features_time script: ', ME.message]);
    % Log the error details
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
end

disp('-------------------------------------------')

% Display completion message
disp('All scripts have been executed successfully.');

% Stop logging
diary off;
