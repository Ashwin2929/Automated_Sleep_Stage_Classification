#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 01:23:32 2024

@author: ashwin
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from log import log_print
from load_pca_scores import load_pca_scores

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
