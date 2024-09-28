#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:01:09 2024

@author: ashwin
"""

import h5py
import numpy as np
from log import log_print


def check_label_order(frequency_pca_file, time_pca_file):
    try:
        # Load the PCA-transformed data from the time domain PCA file
        with h5py.File(time_pca_file, 'r') as f:
            time_data = np.array(f['score'])
            time_labels = time_data[:, -1]  # Assuming the last column is the label
        
        # Load the PCA-transformed data from the frequency domain PCA file
        with h5py.File(frequency_pca_file, 'r') as f:
            freq_data = np.array(f['score'])
            freq_labels = freq_data[:, -1]  # Assuming the last column is the label
        
        # Check if the label arrays are identical
        if np.array_equal(time_labels, freq_labels):
            log_print("The label order in the time PCA score data matches the label order in the frequency PCA score data.")
            return True
        else:
            log_print("The label order in the time PCA score data does not match the label order in the frequency PCA score data.")
            return False
    
    except Exception as e:
        log_print(f"An error occurred while checking label order: {e}")
        return False


def load_pca_scores(file_path):
    """
    Loads PCA scores from an HDF5-based .mat file.
    
    Parameters:
    file_path (str): Path to the .mat file.
    
    Returns:
    np.ndarray: The PCA scores.
    """
    with h5py.File(file_path, 'r') as f:
        # Assuming the PCA scores are stored under the key 'score'
        pca_scores = np.array(f['score'])
    return pca_scores

def append_pca_scores(frequency_file_path, time_file_path):
    """
    Loads PCA scores from two HDF5-based .mat files and appends them along the feature dimension.
    
    Parameters:
    frequency_file_path (str): Path to the frequency domain .mat file.
    time_file_path (str): Path to the time domain .mat file.
    
    Returns:
    np.ndarray: The concatenated PCA scores.
    """
    # Load PCA scores from both files
    pca_scores_frequency = load_pca_scores(frequency_file_path)
    pca_scores_time = load_pca_scores(time_file_path)
    
    appended_pca_scores_with_labels_file_path = './output/appended_pca_scores_with_labels.mat'
    
    
    if check_label_order(frequency_file_path, time_file_path):
        # Separate data and labels
        data_frequency = pca_scores_frequency[:, :-1]  # All columns except the last one are data
        data_time = pca_scores_time[:, :-1]            # All columns except the last one are data
        labels = pca_scores_frequency[:, -1]           # Assuming the last column is the label
    
        # Concatenate the PCA scores along the feature dimension
        appended_pca_scores = np.concatenate((data_frequency, data_time), axis=1)
    
        # Append the label column
        appended_pca_scores_with_labels = np.column_stack((appended_pca_scores, labels))
        
        # Write the result to a .mat file
        with h5py.File(appended_pca_scores_with_labels_file_path, 'w') as f:
            f.create_dataset('score', data=appended_pca_scores_with_labels)
        
        log_print(f"All PCA score data shape: {appended_pca_scores_with_labels.shape}")
        
        return appended_pca_scores_with_labels_file_path
    else:
        log_print("Label order mismatch. Cannot append PCA scores.")
        return None

