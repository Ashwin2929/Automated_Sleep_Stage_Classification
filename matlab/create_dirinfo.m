% Main script to create directory info with logging

% Clear workspace and command window
clear;
clc;

% Define the output directory relative to the current script directory
output_dir = './output';

% Define log file path relative to the output directory
log_file = fullfile(output_dir, 'create_dirinfo_log.txt');

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
disp('Starting the directory info creation process...');

% Specify the directory path based on the operating system
if ispc  % Check if the OS is Windows
    path = 'Z:\2024_Jan_SU_Group_3\raw_data\challenge_dataset\training'; % LOCAL path for Windows
elseif isunix  % Check if the OS is Linux/Unix
    path = '/storage/projects/ce903/2024_Jan_SU_Group_3/raw_data/challenge_dataset/training'
    % path = '/storage/projects/ce903/2024_Jan_SU_Group_3/code/23-24_CE903-SU_team03/Code/sample training data';
    % path = '/Users/ashwin/Projects/University of Essex/CE903/23-24_CE903-SU_team03/Code/sample training data'; % CLUSTER path for Linux/Unix
else
    error('Unsupported operating system');
end

% Check if the directory exists
if ~isfolder(path)
    error('The specified directory does not exist: %s', path);
end

% Get directory information
dirinfo = dir(path);

% Filter out only directories, skipping files
dirinfo = dirinfo([dirinfo.isdir]);

% Save directory information to a .mat file
save(fullfile(output_dir, 'dirinfo.mat'), 'dirinfo');

% Display a message indicating that the file has been created
disp('dirinfo.mat has been created successfully.');

% Convert the directory information to a table
dirinfo_table = struct2table(dirinfo);

% Display the table
disp(dirinfo_table);

% Stop logging
diary off;

% Display completion message
disp('Directory info creation process completed successfully.');
