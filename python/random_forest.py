#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 01:22:09 2024

@author: ashwin
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from log import log_print
from load_pca_scores import load_pca_scores


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

