#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import gc
import pandas as pd
import logging
import h5py
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from scipy.signal import welch
from scipy.stats import iqr, skew, kurtosis, mode


# Ensure output directory exists
output_directory = './output'
os.makedirs(output_directory, exist_ok=True)

# Configure logging to overwrite the log file
logging.basicConfig(filename=os.path.join(output_directory, 'log.txt'), 
                    level=logging.INFO, 
                    format='%(message)s', 
                    filemode='w')

def log_print(message):
    """
    Log a message to both the console and the log file.

    Parameters:
        - message: string. The message to log.

    Returns:
        - None
    """
    
    print(message)  # Print to console
    logging.info(message)  # Log to file

log_print("----------------------------------------------------------------------------------------")
log_print("This is the log for sleep.")
log_print("----------------------------------------------------------------------------------------")

# Ensure logging handlers are properly flushed
logging.shutdown()

def train_evaluate_random_forest(data_path, test_size=0.2, random_state=42):
    """
    Train and evaluate a Random Forest Classifier on PCA-transformed data with hyperparameter tuning, feature scaling, and cross-validation.
    
    Parameters:
    data_path (str): Path to the PCA-transformed data file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed used by the random number generator.
    
    Returns:
    None
    """
    
    data = load_pca_scores(data_path)
    
    log_print("Training the Random Forest model...")
    # Load the PCA-transformed data
    X = data[:, :-1]  # Features (all columns except the last one)
    y = data[:, -1]   # Labels (last column)
    
    log_print(f'data shape: {X.shape}')
    log_print(f'label shape: {y.shape}')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create a pipeline with feature scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Stratified K-Folds cross-validator
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and best score
    log_print(f"Best parameters: {grid_search.best_params_}")
    log_print(f"Best cross-validation score: {grid_search.best_score_}")
    
    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    return report_df

def train_evaluate_xgboost(data_path, test_size=0.2, random_state=42):
    """
    Train and evaluate an XGBoost Classifier on PCA-transformed data with hyperparameter tuning, feature scaling, and cross-validation.
    
    Parameters:
    data_path (str): Path to the PCA-transformed data file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed used by the random number generator.
    
    Returns:
    None
    """
    
    data = load_pca_scores(data_path)
    
    log_print("Training the XGBoost model...")
    # Load the PCA-transformed dat
    X = data[:, :-1]  # Features (all columns except the last one)
    y = data[:, -1]   # Labels (last column)
    
    # Encode labels to be zero-indexed internally
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    
    # Create a pipeline with feature scaling and XGBoost
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Stratified K-Folds cross-validator
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and best score
    log_print(f"Best parameters: {grid_search.best_params_}")
    log_print(f"Best cross-validation score: {grid_search.best_score_}")
    
    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Decode predictions back to original labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    
    # Evaluate the model
    report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    return report_df

def cnn_sleep(input_shape, num_classes):
    """
    Create the CNN model.
    
    Parameters:
    input_shape (tuple): Shape of the input data.
    num_classes (int): Number of classes in the output layer.
    
    Returns:
    model (Sequential): The compiled CNN model.
    """
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_evaluate_cnn(data_path, n_splits=5, epochs=10, batch_size=64, output_csv='classification_reports.csv'):
    """
    Train and evaluate a CNN using k-fold cross-validation.

    Parameters:
    data_path (str): The path to the data file.
    n_splits (int): Number of folds for k-fold cross-validation.
    epochs (int): Number of epochs to train the model.
    batch_size (int): Batch size for training.
    output_csv (str): Path to save the classification reports.

    Returns:
    None
    """
    log_print("Training the CNN model...")
    
    data = load_pca_scores(data_path)
    
    # Split data into features and labels
    X = data[:, :-1]  # Features (all columns except the last one)
    y = data[:, -1]   # Labels (last column)
    
    num_classes = len(np.unique(y))
    
    log_print(f'Data shape: {X.shape}')
    log_print(f'Label shape: {y.shape}')
    log_print(f'Number of classes: {num_classes}')
    
    y = y - y.min()

    # One-hot encode the labels
    y = to_categorical(y, num_classes=num_classes)

    # Reshape X to add a channel dimension (necessary for Conv1D)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Define k-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    reports = []

    # Training and evaluating the model using k-fold cross-validation
    fold_no = 1
    for train_index, test_index in kf.split(X):
        try:
            log_print(f'Training fold {fold_no}...')
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = cnn_sleep((X.shape[1], 1), num_classes)
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
            
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            report = classification_report(y_true, y_pred_classes, output_dict=True)
            log_print(f'Classification report for fold {fold_no}:\n{classification_report(y_true, y_pred_classes)}')
            
            report_df = pd.DataFrame(report).transpose()
            report_df['fold'] = fold_no
            reports.append(report_df)
            
            fold_no += 1

        except Exception as e:
            log_print(f'Error during training fold {fold_no}: {e}')

    # Concatenate all fold reports into a single DataFrame
    all_reports_df = pd.concat(reports, ignore_index=True)
    
    # Calculate the average classification report
    average_report = all_reports_df.groupby(level=0).mean()
    
    return average_report

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


def process_participant_data_pca_time(main_data_path, window_sampling, sampling_frequency, window_size_seconds):
    """
    Process participant data to perform PCA in the time domain.

    Parameters:
        - main_data_path: string. Path to the main data directory.
        - window_sampling: int. Number of samples per window.
        - sampling_frequency: int. Sampling frequency of the data.
        - window_size_seconds: int. Size of the window in seconds.

    Returns:
        - pca_coeff_time_file_path: string. Path to the output file containing PCA coefficients for time data.
    """
    
    participant_directories = [directory.name for directory in os.scandir(main_data_path) if directory.is_dir()]
    
    batch_size = 10
    
    
    # Check if the all_data_for_pca file exists
    all_data_for_pca_time_file_path = './output/all_data_for_pca_time.mat'
    pca_coeff_time_file_path='./output/pca_coefficients_time.mat'
    
    if os.path.exists(all_data_for_pca_time_file_path):
        os.remove(all_data_for_pca_time_file_path)
        log_print(f"{all_data_for_pca_time_file_path} has been deleted for the latest process.")
    
    if os.path.exists(pca_coeff_time_file_path):
        os.remove(pca_coeff_time_file_path)
        log_print(f"{pca_coeff_time_file_path} has been deleted for the latest process.")
    
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
    
                participant_data = time_domain_transform(participant_data, window_sampling, num_channels)
                log_print(f"Time domain transformed data: {participant_data.shape}")
    
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
            save_to_mat(all_data_for_pca_time_file_path, {'batch_data': batch_data_for_pca})
            log_print(f"Processed Batch {i // batch_size + 1} data saved to {all_data_for_pca_time_file_path}")
    
        gc.collect()
            
    
    log_print("----------------------------------------------------------------------------------------")
    
    perform_pca_time(
    input_mat_file = all_data_for_pca_time_file_path,
    output_mat_file = pca_coeff_time_file_path,
    n_components=200)
    
    return pca_coeff_time_file_path

def main():
    """
    Main function to execute the processing and classification of the dataset.

    Parameters:
        None

    Returns:
        None
    """
    
    main_data_path = "/storage/projects/ce903/2024_Jan_SU_Group_3/raw_data/challenge_dataset/training"
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