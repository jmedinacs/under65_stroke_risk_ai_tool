"""
Data preprocessing pipeline for stroke prediction (under-65 cohort).

This module handles preprocessing steps including:
- Dropping statistically insignificant features
- One-hot encoding of categorical variables
- Median imputation or (optional) Random Forest imputation for missing values
- (Optional) SMOTE-Tomek resampling for class imbalance
- Saving preprocessing artifacts (e.g., column order, imputer) for deployment

Author: jarpy  
Created on: July 9, 2025
"""


from utils.data_io import save_clean_data, load_clean_data
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import os 
import pandas as pd 
import json
import joblib

def drop_insignificant_features(X_train, X_test):
    """
    Chi-squared test identified that 'gender' and 'residence_type' are statistically insignificant
    features, so they are dropped from both training and test datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned X_train and X_test with insignificant features removed.
    """
    columns_to_drop = ["gender", "residence_type"]
    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)
    print("Dropped insignificant features.")
    return X_train, X_test

def encode_categorical_features(X_train, X_test):
    """
    One-hot encode categorical features based on training data only.
    Ensures both X_train and X_test have the same columns.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded X_train and aligned X_test.
    """
    # Encode training set
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)

    # Extract the column names in order
    column_order = list(X_train_encoded.columns)

    # Save column order to JSON for ensemble and deployment
    with open("../../models/columns.json","w") as f:
        json.dump(column_order,f)
    print("Column order saved to models/columns.json")

    # Encode test set
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)

    # Align test set to have same columns as train (fill missing with 0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    return X_train_encoded, X_test_encoded

def apply_smote(X_train, y_train):
    """ 
    Applies SMOTE-TOMEK to the training data to address class imbalance.

    Args:
        X_train (pd.DataFrame): Encoded training features.
        y_train (pd.Series): Target labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled training features and labels.
    """
    smote = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote._fit_resample(X_train, y_train)
    print("Smote-Tomek Implemented to training data")
    return X_train_resampled, y_train_resampled

def normalize_continuous(X_train, X_test):
    """ 
    Placeholder for normalization logic. Consider using MinMaxScaler or StandardScaler here.
    """
    pass

def random_forest_impute_bmi(X_train, X_test):
    """ """
    # Separate rows with and without BMI in training data
    train_known = X_train[X_train['bmi'].notna()]
    train_missing = X_train[X_train['bmi'].isna()]
    
    # Separate rows with and without BMI in test data
    test_known = X_test[X_test['bmi'].notna()]
    test_missing = X_test[X_test['bmi'].isna()]
    
    # Prepare the training data for the RF model
    X_train_model = train_known.drop(columns=['bmi'])
    y_train_model = train_known['bmi']
    
    # Train the regressor model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42) # create the model
    rf_model.fit(X_train_model, y_train_model) # Train the model with bmi as target
    
    # Predict and fill BMI for missing rows in X_train
    if not train_missing.empty:
        X_missing_model = train_missing.drop(columns=['bmi']) # drop the bmi column from training data
        bmi_pred = rf_model.predict(X_missing_model) # predict the missing bmi
        X_train.loc[train_missing.index,'bmi'] = bmi_pred # Use the index of missing bmi and assign prediction

    # Predict and fill BMI for missing rows in X_test
    if not test_missing.empty:
        X_missing_model = test_missing.drop(columns=['bmi'])
        bmi_pred = rf_model.predict(X_missing_model)
        X_test.loc[test_missing.index,'bmi'] = bmi_pred
        
    # save the trained model
    joblib.dump(rf_model,'../../models/rf_bmi_imputer.joblib')
    
    return X_train, X_test
    
        
def median_impute(X_train, X_test):
    """
    Imputes missing values using the median of each feature from the training set.

    Returns:
        X_train_imputed (pd.DataFrame): Median-imputed training data
        X_test_imputed (pd.DataFrame): Median-imputed test data
    """
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    # Save the imputer for deployment use
    os.makedirs("../../models", exist_ok=True)
    joblib.dump(imputer,"../../models/median_imputer.joblib")
    print("Median imputer saved to models/median_imputer.joblib")

    X_train_median = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_median = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X_train_median, X_test_median

def preprocess_data(smote = False, rf_impute = False):
    """
    Preprocesses the data for model training and evaluation.

    Steps:
    - Loads split train/test CSVs
    - Drops insignificant features
    - One-hot encodes categorical features
    - Imputes missing values using median (or placeholder for RF)
    - Optionally applies SMOTE-Tomek to training data

    Args:
        smote (bool): Whether to apply SMOTE-Tomek oversampling.
        rf_impute (bool): Whether to use Random Forest imputation for BMI (vs. median).

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrames/Series)
    """
    # Load training and test sets
    X_test = load_clean_data("X_test")
    X_train = load_clean_data("X_train")
    y_test = load_clean_data("y_test")
    y_train = load_clean_data("y_train")

    # Drop insignificant features before encoding
    X_train, X_test = drop_insignificant_features(X_train, X_test)

    # One-hot encode categorical features for training data
    X_train, X_test = encode_categorical_features(X_train, X_test)

    X_train_impute = X_train.copy()
    X_test_impute = X_test.copy()
    if not rf_impute:
        X_train_impute, X_test_impute = median_impute(X_train, X_test)

    X_train_resampled = X_train_impute.copy()
    y_train_resampled = y_train.copy()
    if smote:
        # Apply Smote-Tomek on training data
        X_train_resampled, y_train_resampled = apply_smote(X_train_impute, y_train)

if __name__ == '__main__':
    preprocess_data(smote=True)
