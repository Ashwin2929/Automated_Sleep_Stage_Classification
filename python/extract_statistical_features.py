import numpy as np
import h5py
from scipy.stats import iqr, skew, kurtosis
from load_pca_scores import load_pca_scores
from log import log_print


def extract_stat_features(pca_scores_with_labels_file_path):
    """
    Extract statistical features from PCA scores, excluding the label column.

    Parameters:
        - pca_scores_with_labels: 2D array of shape (number of samples, number of features + 1). The last column is the label.

    Returns:
        - features_with_labels: 2D array of shape (number of samples, number of extracted features + 1). 
          The last column is the label.
    """
    
    stat_features_pca_scores_file_path = './output/stat_features_pca_scores.mat'
    
    pca_scores_with_labels = load_pca_scores(pca_scores_with_labels_file_path)  
    
    # Separate PCA scores and labels
    pca_scores = pca_scores_with_labels[:, :-1]
    labels = pca_scores_with_labels[:, -1]

    # Extract statistical features
    intqr = iqr(pca_scores, axis=1)
    skewness_values = skew(pca_scores, axis=1, bias=False)
    kurtosis_values = kurtosis(pca_scores, axis=1, bias=False)
    stdev = np.std(pca_scores, axis=1, ddof=0)
    
    # Combine all features into a single array
    features = np.column_stack((intqr, skewness_values, kurtosis_values, stdev))
    
    # Append the label column
    features_with_labels = np.column_stack((features, labels))
    
    # Write the result to a .mat file
    with h5py.File(stat_features_pca_scores_file_path, 'w') as f:
        f.create_dataset('score', data=features_with_labels)
        
    log_print(f"Stat features PCA score data shape: {features_with_labels.shape}")
    
    return stat_features_pca_scores_file_path
