#!/usr/bin/python36

"""
Created on Tue Jun  4 14:45:29 2024

@author: ap23710
"""

import os
import numpy as np
import gc
import pandas as pd
from load_data import load_data
from signal_processing import check_standard_deviations
from signal_processing import frequency_domain_transform
from load_data import save_to_mat
from pca import perform_pca_frequency
from log import log_print
from load_label_data import get_labels
from load_label_data import extract_window_labels


def process_participant_data_pca_frequency(main_data_path, window_sampling, sampling_frequency, window_size_seconds):
    """
    Process participant data to perform PCA in the frequency domain.

    Parameters:
        - main_data_path: string. Path to the main data directory.
        - window_sampling: int. Number of samples per window.
        - sampling_frequency: int. Sampling frequency of the data.
        - window_size_seconds: int. Size of the window in seconds.

    Returns:
        - pca_coeff_freq_file_path: string. Path to the output file containing PCA coefficients for frequency data.
    """
    
    participant_directories = [directory.name for directory in os.scandir(main_data_path) if directory.is_dir()]
    
    batch_size = 10
    
    
    # Check if the all_data_for_pca file exists
    all_data_for_pca_freq_file_path = './output/all_data_for_pca_freq.mat'
    pca_coeff_freq_file_path='./output/pca_coefficients_frequency.mat'
    
    if os.path.exists(all_data_for_pca_freq_file_path):
        os.remove(all_data_for_pca_freq_file_path)
        log_print(f"{all_data_for_pca_freq_file_path} has been deleted for the latest process.")
    
    if os.path.exists(pca_coeff_freq_file_path):
        os.remove(pca_coeff_freq_file_path)
        log_print(f"{pca_coeff_freq_file_path} has been deleted for the latest process.")
    
    # Ensure output directory exists
    output_directory = './output'
    os.makedirs(output_directory, exist_ok=True)
    
    # Process data in batches
    for i in range(0, len(participant_directories), batch_size):
        batch_directories = participant_directories[i:i + batch_size]
        log_print("----------------------------------------------------------------------------------------")
        log_print(f"Processing batch {i // batch_size + 1}: Participants {i + 1} to {i + len(batch_directories)}")
        log_print("----------------------------------------------------------------------------------------")
    
        batch_data_for_pca_list = []
    
        # Process data for each participant
        for participant in batch_directories:
            log_print("----------------------------------------------------------------------------------------")
            log_print(f"Participant: {participant}")
    
            # Construct full path to participant's directory
            participant_data_path = os.path.join(main_data_path, participant)
            # log_print("Participant Path:", participant_data_path)
    
            # Load raw data for the participant
            participant_data, num_channels, channel_indices = load_data(participant, participant_data_path)
            participant_data = np.array(participant_data, dtype=np.float64)
            log_print(f"Raw Data Shape:  {participant_data.shape}")
            
            # Check standard deviations of signals
            standard_deviations = check_standard_deviations(participant_data)
    
            # Check if all channels have std greater than eps
            eps = np.finfo(float).eps
            if np.all(standard_deviations > eps):
                log_print("All channels have std greater than eps")
    
                if participant_data.size == 0:
                    log_print(f"Skipping participant {participant} due to low standard deviation in signals.")
                    continue
    
                participant_data = frequency_domain_transform(participant_data, window_sampling, num_channels, sampling_frequency)
                log_print(f"Frequency domain transformed data: {participant_data.shape}")
    
                # Load and process labels for the participant
                labels = get_labels(participant, participant_data_path)
                labels = extract_window_labels(labels, sampling_frequency, window_size_seconds)
                log_print(f"Labels shape after extraction: {labels.shape}")
    
                if len(labels) != participant_data.shape[0]:
                    log_print(f"Skipping participant {participant} due to mismatch in data and label lengths. Data length: {participant_data.shape[0]}, Labels length: {len(labels)}")
                    continue
    
                # Create a DataFrame and append labels
                participant_df = pd.DataFrame(participant_data)
                participant_df['label'] = labels
                
                # Combine data and labels for PCA
                data_with_labels = participant_df.values  # Convert DataFrame to NumPy array
                batch_data_for_pca_list.append(data_with_labels)
                
            else:
                log_print(f"Skipping participant {participant} due to low standard deviation in one or more channels.") 
            
        if batch_data_for_pca_list:
            batch_data_for_pca = np.concatenate(batch_data_for_pca_list, axis=0)
            save_to_mat(all_data_for_pca_freq_file_path, {'batch_data': batch_data_for_pca})
            log_print(f"Processed Batch {i // batch_size + 1} data saved to {all_data_for_pca_freq_file_path}")
    
        gc.collect()
            
    
    log_print("----------------------------------------------------------------------------------------")
    
    perform_pca_frequency(
    input_mat_file = all_data_for_pca_freq_file_path,
    output_mat_file = pca_coeff_freq_file_path,
    n_components=200)
    
    
    return pca_coeff_freq_file_path
