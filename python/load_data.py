# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:07:24 2024

@author: ap23710
"""

import os
import scipy.io
import numpy as np
import h5py

def load_data(filename, directory):
    """
    Load data from a .mat file and extract specified channels.

    Parameters:
        - filename: string. The name of the .mat file (without extension).
        - directory: string. The path to the directory containing the .mat file.

    Returns:
        - transposed_signals: array. The extracted and transposed signals.
        - num_channels: int. The number of channels in the data.
        - channel_indices: list of int. The indices of the channels used.
    """
    
    # Constructing the path to the .mat file
    file_path = os.path.join(directory, filename + ".mat")

    # Loading the .mat file
    mat_contents = scipy.io.loadmat(file_path)

    # Extracting the data array from the loaded .mat file
    data_array = mat_contents.get('val')
    
    num_channels= data_array.shape[0]
    print("number of channels for ", filename, ":", num_channels)
    
    # Define channel configuration based on the number of channels
    channel_indices = []
    if num_channels == 13:
        # EEG, EMG, respiration - abdomen, respiration - chest, oximetry, ECG
        channel_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    elif num_channels == 5:
        # EMG, respiration - abdomen, respiration - chest, oximetry, ECG
        channel_indices = [8, 9, 10, 12, 13]

    # Extracting signals for specified channels
    extracted_signals = []
    for channel in channel_indices:
        channel_index = channel - 1
        extracted_signals.append(data_array[channel_index, :])

    # Transposing the signals array
    transposed_signals = np.transpose(extracted_signals)

    return transposed_signals, num_channels, channel_indices


# Helper function to save data to a .mat file using MATLAB 7.3 format
def save_to_mat(file_path, data_dict):
    """
    Save data to a .mat file using MATLAB 7.3 format.

    Parameters:
        - file_path: string. The path to the .mat file to save data.
        - data_dict: dict. A dictionary containing the data to save.

    Returns:
        - None
    """
    
    try:
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r+') as f:
                for key in data_dict:
                    if key in f:
                        # Concatenate existing and new data
                        existing_data = f[key][:]
                        new_data = np.concatenate((existing_data, data_dict[key]), axis=0)
                        del f[key]  # Delete the old dataset
                        f.create_dataset(key, data=new_data)  # Create a new dataset with concatenated data
                    else:
                        f.create_dataset(key, data=data_dict[key])
        else:
            with h5py.File(file_path, 'w') as f:
                for key, value in data_dict.items():
                    f.create_dataset(key, data=value)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}: {e}")


# Helper function to load data from a .mat file using h5py
def load_from_mat(file_path):
    """
    Load data from a .mat file using h5py.

    Parameters:
        - file_path: string. The path to the .mat file to load data from.

    Returns:
        - data_dict: dict. A dictionary containing the loaded data.
    """
    
    try:
        data_dict = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data_dict[key] = np.array(f[key])
        print(f"Data successfully loaded from {file_path}")
        return data_dict
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None
