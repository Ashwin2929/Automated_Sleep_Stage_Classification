#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:32:03 2024

@author: ashwin
"""

import os
import h5py
import numpy as np
from scipy.stats import mode

def get_labels(filename, path):
    """Import labels from a specified -arousal.mat file.
    
        Parameters:
            - filename: string. File name of the labels file (without "-arousal.mat").
            - fs: int. Sampling frequency.
            - window_size_seconds: int. Size of the window in seconds.
            - rows: int. Number of rows in the data file.
            - path: string. Path to the participant folder.

        Returns:
            - labels: array of ints. Labels of sleep stages (1: Wake, 2: N1, 3: N2, 4: N3, 5: REM, 6: Undefined).
    """
    dir_arousal = os.path.join(path, f"{filename}-arousal.mat")
    with h5py.File(dir_arousal, 'r') as file:
        arousal_data = file['data/sleep_stages']
        labels = np.zeros(arousal_data['wake'].shape[1])
        labels[arousal_data['wake'][0,:] == 1] = 1
        labels[arousal_data['nonrem1'][0,:] == 1] = 2
        labels[arousal_data['nonrem2'][0,:] == 1] = 3
        labels[arousal_data['nonrem3'][0,:] == 1] = 4
        labels[arousal_data['rem'][0,:] == 1] = 5
        labels[(arousal_data['wake'][0,:] == 0) & 
               (arousal_data['nonrem1'][0,:] == 0) & 
               (arousal_data['nonrem2'][0,:] == 0) & 
               (arousal_data['nonrem3'][0,:] == 0) & 
               (arousal_data['rem'][0,:] == 0)] = 6
    return labels


def extract_window_labels(labels, fs, window_size_seconds):
    """Extract label for each window.

        Parameters:
            - labels: array of ints. Imported labels for each data point.
            - fs: int. Sampling frequency.
            - window_size_seconds: int. Size of the window in seconds.

        Returns:
            - labels_windows: array of ints. Labels for each window.
    """
    window_size = window_size_seconds * fs
    num_windows = len(labels) // window_size
    labels = labels[:num_windows * window_size]
    labels = labels.reshape((num_windows, window_size))
    labels_windows = mode(labels, axis=1).mode.flatten()
    return labels_windows
