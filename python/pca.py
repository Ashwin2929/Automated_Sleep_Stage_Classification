# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:33:51 2024

@author: ap23710
"""


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from log import log_print
import h5py
import numpy as np



def generate_shuffle_indices(num_samples):
    """
    Generate and save shuffle indices for a given number of samples.

    Parameters:
        - num_samples: int. The number of samples to generate shuffle indices for.

    Returns:
        - indices: array of ints. The generated shuffle indices.
    """
    
    np.random.seed(42)  # Set a fixed random seed for reproducibility
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffle_file='shuffle_indices.npy'
    np.save(shuffle_file, indices)
    print(f'Shuffle indices saved to {shuffle_file}')
    return indices


def perform_pca_frequency(input_mat_file, output_mat_file, n_components=200):
    """
    Perform PCA on frequency domain data and save the results.

    Parameters:
        - input_mat_file: string. Path to the input .mat file containing data.
        - output_mat_file: string. Path to the output .mat file to save PCA results.
        - n_components: int. Number of PCA components to retain (default is 200).

    Returns:
        - None
    """
    
    try:
        # Load the data from the .mat file using h5py
        with h5py.File(input_mat_file, 'r') as f:
            all_data = np.array(f['batch_data'])
        
        indices = generate_shuffle_indices(all_data.shape[0])
        
        # Load shuffle indices
        # indices = np.load(shuffle_file)
        
        # Shuffle the data using the loaded indices
        all_data = all_data[indices]
        
        # Separate labels from data
        labels = all_data[:, -1]  # Assuming the last column is the label
        data = all_data[:, :-1]   # All columns except the last one are data
        
        # Convert to a pandas DataFrame for easier handling (optional)
        df_data = pd.DataFrame(data)
        log_print(f'All data for PCA: {df_data.shape}')
        
        # Center the data (important for PCA)
        data_centered = df_data - df_data.mean(axis=0)
        
        # Perform PCA
        pca = PCA(n_components=n_components, svd_solver='full')
        score = pca.fit_transform(data_centered)
        coeff = pca.components_  # Transpose to match MATLAB's output format
        
        # Re-append labels to the PCA scores
        labeled_score = np.column_stack((score, labels))
        
        # Save PCA coefficients and labeled scores to a new .mat file using h5py
        with h5py.File(output_mat_file, 'w') as f:
            f.create_dataset('coeff', data=coeff)
            f.create_dataset('score', data=labeled_score)
        
        log_print(f'PCA coefficients shape: {coeff.shape}')
        log_print(f'PCA scores shape: {score.shape}')
        log_print(f'Labeled PCA scores shape: {labeled_score.shape}')
        log_print(f'PCA coefficients and labeled scores saved to {output_mat_file}')
    
    except Exception as e:
        log_print(f"An error occurred during PCA: {e}")



def perform_pca_time(input_mat_file, output_mat_file, n_components=200, chunk_size=20000, shuffle_file='shuffle_indices.npy'):
    """
    Perform Incremental PCA on time domain data and save the results.

    Parameters:
        - input_mat_file: string. Path to the input .mat file containing data.
        - output_mat_file: string. Path to the output .mat file to save PCA results.
        - n_components: int. Number of PCA components to retain (default is 200).
        - chunk_size: int. Size of chunks to process incrementally (default is 20000).
        - shuffle_file: string. Path to the .npy file containing shuffle indices (default is 'shuffle_indices.npy').

    Returns:
        - None
    """
    
    try:
        # Load the entire dataset to shuffle it (if it's possible to fit in memory)
        with h5py.File(input_mat_file, 'r') as f:
            dataset = np.array(f['batch_data'])
            num_samples = dataset.shape[0]
            log_print(f'Total number of samples: {num_samples}')
        
        # Load shuffle indices
        indices = np.load(shuffle_file)
        
        # Shuffle the data using the loaded indices
        dataset = dataset[indices]
        
        # Separate labels from data
        labels = dataset[:, -1]  # Assuming the last column is the label
        data = dataset[:, :-1]   # All columns except the last one are data
        
        # Initialize IncrementalPCA
        ipca = IncrementalPCA(n_components=n_components)
        
        # Fit the IncrementalPCA model on shuffled chunks
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_data = data[start_idx:end_idx]
            
            # Center the data (important for PCA)
            chunk_data_centered = chunk_data - np.mean(chunk_data, axis=0)
            
            ipca.partial_fit(chunk_data_centered)
            log_print(f'Processed chunk: {start_idx} to {end_idx}')
        
        # Transform the data in chunks and store the results
        transformed_data = np.zeros((num_samples, n_components))
        
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_data = data[start_idx:end_idx]
            
            # Center the data
            chunk_data_centered = chunk_data - np.mean(chunk_data, axis=0)
            
            transformed_chunk = ipca.transform(chunk_data_centered)
            transformed_data[start_idx:end_idx] = transformed_chunk
            log_print(f'Transformed chunk: {start_idx} to {end_idx}')
        
        # Re-append labels to the transformed data
        labeled_transformed_data = np.column_stack((transformed_data, labels))
        
        # Save the PCA coefficients and labeled scores to a new .mat file
        with h5py.File(output_mat_file, 'w') as f:
            f.create_dataset('coeff', data=ipca.components_)
            f.create_dataset('score', data=labeled_transformed_data)
        
        # Log the shapes and save confirmation
        log_print(f'PCA coefficients shape: {ipca.components_.shape}')
        log_print(f'PCA scores shape: {transformed_data.shape}')
        log_print(f'Labeled PCA scores shape: {labeled_transformed_data.shape}')
        log_print(f'PCA coefficients and labeled scores saved to {output_mat_file}')
    
    except Exception as e:
        log_print(f"An error occurred during PCA: {e}")

