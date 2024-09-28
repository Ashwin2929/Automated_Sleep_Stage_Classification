#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:18:25 2024

@author: ashwin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from load_pca_scores import load_pca_scores
from log import log_print


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


