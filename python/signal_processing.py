# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:15:56 2024

@author: ap23710
"""

import numpy as np
from scipy.signal import welch

def normalize_signals(signals):
    """
    Normalize signals to have zero mean and unit standard deviation.

    Parameters:
        - signals: array. The input signals to normalize. Should be 2D (samples x channels).

    Returns:
        - normalized_signals: array. The normalized signals.
        - standard_deviations: list. Standard deviations of the original signals.
    """
    
    # Ensure signals is at least 2D
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    
    # Check if signals has the expected dimensions
    if signals.ndim != 2:
        raise ValueError("signals should be a 2D array")
    
    normalized_signals_list = []  # List to store normalized signals
    standard_deviations = []  # List to store standard deviations

    num_signals = signals.shape[1]  # Number of signals (columns)

    for i in range(num_signals):
        # Extracting one signal at a time
        current_signal = signals[:, i]

        # Calculating mean and standard deviation of the signal
        signal_mean = np.mean(current_signal)
        signal_standard_deviation = np.std(current_signal)

        # Check for low standard deviation
        if signal_standard_deviation > np.finfo(float).eps:
            # Normalizing the signal
            normalized_signal = (current_signal - signal_mean) / signal_standard_deviation
        else:
            # If the standard deviation is too low, set the normalized signal to zero
            normalized_signal = np.zeros_like(current_signal)

        # Appending normalized signal to the list
        normalized_signals_list.append(normalized_signal)

        # Appending standard deviation to the list
        standard_deviations.append(signal_standard_deviation)

    # Stack the normalized signals into a 2D array and return along with standard deviations
    normalized_signals = np.column_stack(normalized_signals_list)
    return normalized_signals, standard_deviations

# Function to check standard deviations
def check_standard_deviations(signals):
    """
    Check the standard deviations of the signals.

    Parameters:
        - signals: array. The input signals. Should be 2D (samples x channels).

    Returns:
        - standard_deviations: array. The standard deviations of the signals.
    """
    
    # Ensure signals is at least 2D
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    
    # Check if signals has the expected dimensions
    if signals.ndim != 2:
        raise ValueError("signals should be a 2D array")
    
    standard_deviations = np.std(signals, axis=0)
    
    return standard_deviations


def segment_signals(signals, fs, window_size_seconds):
    """
    Segment signals into windows of specified size.

    Parameters:
        - signals: array. The input signals to segment. Should be 2D (samples x channels).
        - fs: int. The sampling frequency of the signals.
        - window_size_seconds: int. The size of each window in seconds.

    Returns:
        - windowed_signals: array. The segmented signals.
        - num_windows: int. The number of windows created.
    """
    
    # Ensure signals is at least 2D
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    
    # Check if signals has the expected dimensions
    if signals.ndim != 2:
        raise ValueError("signals should be a 2D array")
        
    num_channels = signals.shape[1]
    window_size = fs * window_size_seconds
    num_windows = signals.shape[0] // window_size
    signals = signals[:num_windows*window_size, :]
    windowed_signals = signals.reshape(num_windows, num_channels, window_size, order='F')
    return windowed_signals, num_windows


# Normalizing the data function
def normalize_data(data_channel):
    """
    Normalize a data channel to have zero mean and unit standard deviation.

    Parameters:
        - data_channel: array. The input data channel to normalize.

    Returns:
        - normalized: array. The normalized data channel.
    """
    
    mean = np.mean(data_channel)
    std = np.std(data_channel)
    normalized = (data_channel - mean) / std if std > 0 else data_channel
    return normalized


# Normalizing the data function
def normalize_data_time(data_channel):
    """
    Normalize a data channel with time-domain specific adjustments.

    Parameters:
        - data_channel: array. The input data channel to normalize.

    Returns:
        - normalized: array. The normalized data channel.
    """
    
    data_channel[data_channel < -1985] = 0
    mean_value = np.mean(data_channel)
    max_abs = np.max(np.abs(data_channel))
    normalized = (data_channel - mean_value) / max_abs if max_abs > 0 else data_channel
    return normalized


# Reshape the data into windows function
def reshape_channel_data(data_channel, window_sampling):
    """
    Reshape a data channel into windows of specified size.

    Parameters:
        - data_channel: array. The input data channel to reshape.
        - window_sampling: int. The number of samples per window.

    Returns:
        - data_channel_wind: array. The reshaped data channel.
    """
    
    num_win30_dec = len(data_channel) / window_sampling
    num_win30 = int(num_win30_dec)
    valid_samples = num_win30 * window_sampling

    if num_win30_dec > num_win30:
        data_channel = data_channel[:valid_samples]

    data_channel_wind = data_channel.reshape(num_win30, window_sampling)
    return data_channel_wind


def time_domain_transform(data_all, window_sampling, num_channels):
    """
    Transform data to the time domain by normalizing and reshaping it into windows.

    Parameters:
        - data_all: array. The input data to transform. Should be 2D (samples x channels).
        - window_sampling: int. The number of samples per window.
        - num_channels: int. The number of channels in the data.

    Returns:
        - reshaped_data: array. The transformed data in the time domain.
    """
    
    reshaped_data = []

    for channel in range(num_channels):
        data_channel = data_all[:, channel]

        data_channel[data_channel > 10 * np.std(data_channel)] = 0

        data_channel_normalized = normalize_data_time(data_channel)

        data_channel_wind = reshape_channel_data(data_channel_normalized, window_sampling)

        reshaped_data.append(data_channel_wind)

    # Stack along the first axis to form a 2D array
    reshaped_data = np.hstack(reshaped_data)


    return reshaped_data

# Function to transform data to the frequency domain
def frequency_domain_transform(data_all, window_sampling, num_channels, sampling_frequency):
    """
    Transform data to the frequency domain using the Welch method.

    Parameters:
        - data_all: array. The input data to transform. Should be 2D (samples x channels).
        - window_sampling: int. The number of samples per window.
        - num_channels: int. The number of channels in the data.
        - sampling_frequency: int. The sampling frequency of the data.

    Returns:
        - data_all_win_freq: array. The transformed data in the frequency domain.
    """
    
    pwelch_window = sampling_frequency  # 200 samples
    pw_overlap = max(1, round(0.2 * pwelch_window))  # Ensure overlap is a positive integer
    NFFT = max(256, 2 ** int(np.ceil(np.log2(pwelch_window))))  # 256
    sample_40Hz = int(np.ceil((45 / (sampling_frequency / 2)) * (NFFT / 2)))  # 58

    data_all_win_freq = []

    for channel in range(num_channels):
        data_channel = data_all[:, channel]
        
        data_channel_wind = reshape_channel_data(data_channel, window_sampling)

        # Applying the welch method with parameters aligned to MATLAB's pwelch
        freq, pxx = welch(data_channel_wind.T, fs=sampling_frequency, window='hamming', nperseg=pwelch_window, noverlap=pw_overlap, nfft=NFFT, axis=0, detrend=False)

        # Handle zero values in pxx before applying log10
        min_nonzero = np.nanmin(pxx[pxx > 0]) if np.any(pxx > 0) else 1e-10
        pxx[pxx == 0] = min_nonzero
        pxx_useful = np.log10(pxx)
        pxx_useful = pxx_useful[:sample_40Hz, :].T

        # Eliminate outliers
        pxx_std = np.std(pxx_useful, axis=1, keepdims=True)
        outlier_mask = np.abs(pxx_useful) > 10 * pxx_std
        pxx_useful[outlier_mask] = 0

        # Replace -inf, inf, and NaN values
        pxx_useful = np.nan_to_num(pxx_useful)

        # Normalize the PSD
        max_abs_pxx = np.max(np.abs(pxx_useful), axis=1, keepdims=True)
        max_abs_pxx[max_abs_pxx == 0] = 1  # Prevent division by zero
        pxx_useful_norm = pxx_useful / max_abs_pxx

        # Subtract the mean from each row
        pxx_useful_norm -= np.mean(pxx_useful_norm, axis=1, keepdims=True)

        data_all_win_freq.append(pxx_useful_norm)

    # Concatenate all channel data along the columns
    data_all_win_freq = np.hstack(data_all_win_freq)

    return data_all_win_freq

