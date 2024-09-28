#!/usr/bin/python36

"""
Created on Mon Jul  1 19:18:25 2024

@author: ashwin
"""

from process_participant_data_pca_time import process_participant_data_pca_time
from process_participant_data_pca_frequency import process_participant_data_pca_frequency
from log import log_print
from load_pca_scores import append_pca_scores
from extract_statistical_features import extract_stat_features
from random_forest import train_evaluate_random_forest
from xGBoost import train_evaluate_xgboost
from CNN import train_evaluate_cnn

def main():
    """
    Main function to execute the processing and classification of the dataset.

    Parameters:
        None

    Returns:
        None
    """
    
    # main_data_path = "/storage/projects/ce903/2024_Jan_SU_Group_3/raw_data/challenge_dataset/training"
    main_data_path = "/Users/ashwin/Projects/University of Essex/CE903/23-24_CE903-SU_team03/Code/sample training data"
    # main_data_path = "Z:/2024_Jan_SU_Group_3/23-24_CE903-SU_team03/sample training data"
    sampling_frequency = 200
    window_size_seconds = 30
    window_sampling = window_size_seconds * sampling_frequency
    
    
    log_print("----------------------------------------------------------------------------------------")
    log_print("Processing frequency PCA")
    log_print("----------------------------------------------------------------------------------------")
    
    pca_coeff_freq_file_path = process_participant_data_pca_frequency(main_data_path, window_sampling, sampling_frequency, window_size_seconds)
    
    log_print("----------------------------------------------------------------------------------------")
    log_print("Completed frequency PCA")
    log_print("----------------------------------------------------------------------------------------")
    
    log_print("----------------------------------------------------------------------------------------")
    log_print("Processing time PCA")
    log_print("----------------------------------------------------------------------------------------")
    
    pca_coeff_time_file_path = process_participant_data_pca_time(main_data_path, window_sampling, sampling_frequency, window_size_seconds)
    
    log_print("----------------------------------------------------------------------------------------")
    log_print("Completed time PCA")
    log_print("----------------------------------------------------------------------------------------")
    
    
    appended_pca_scores_with_labels_file_path = append_pca_scores(pca_coeff_freq_file_path, pca_coeff_time_file_path)
    
    log_print("Completed appending Time and Frequency PCA scores")
    log_print("----------------------------------------------------------------------------------------")
    
    stat_features_pca_scores_file_path = extract_stat_features(appended_pca_scores_with_labels_file_path)
    
    log_print("Completed extracting statistical features for PCA scores")
    log_print("----------------------------------------------------------------------------------------")
    

    rf_classification_report = train_evaluate_random_forest(stat_features_pca_scores_file_path)
    rf_classification_report.to_csv('./output/rf_classification_report.csv', index=True)
    
    log_print("----------------------------------------------------------------------------------------")
    log_print(f"classification report: {rf_classification_report}" )
    log_print("----------------------------------------------------------------------------------------")
    
    xgb_classification_report = train_evaluate_xgboost(stat_features_pca_scores_file_path)
    xgb_classification_report.to_csv('./output/xgb_classification_report.csv', index=True)
    
    log_print("----------------------------------------------------------------------------------------")
    log_print(f"classification report: {xgb_classification_report}" )
    log_print("----------------------------------------------------------------------------------------")
    
    cnn_classification_report = train_evaluate_cnn(appended_pca_scores_with_labels_file_path)
    cnn_classification_report.to_csv('./output/cnn_classification_report.csv', index=True)
    
    log_print("----------------------------------------------------------------------------------------")
    log_print(f"classification report: {cnn_classification_report}" )
    log_print("----------------------------------------------------------------------------------------")
    

if __name__ == "__main__":
    main()
