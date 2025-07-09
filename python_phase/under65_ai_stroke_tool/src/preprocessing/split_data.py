"""
split_data.py

Splits the cleaned stroke risk dataset into training and testing sets.

This script:
- Loads the cleaned dataset using load_clean_data()
- Separates features (X) and target (y)
- Performs a stratified 80/20 train-test split to preserve class balance
- Saves each split (X_train, X_test, y_train, y_test) to /data/processed

Usage:
Can be run independently or imported as part of a larger preprocessing pipeline.
"""

from utils.data_io import load_clean_data, save_clean_data
import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_target(df):
    """
    Splits the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): The full cleaned dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y

def train_test_stratified_split(X, y):
    """
    Performs a stratified train/test split to preserve class distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )    
    return X_train, X_test, y_train, y_test

def split_dataset():
    """
    Full dataset splitting pipeline.

    Steps:
    - Loads cleaned dataset
    - Splits into X (features) and y (target)
    - Performs stratified 80/20 split
    - Saves all 4 outputs as CSVs for consistent downstream use
    """
    df = load_clean_data("under65_clean_data")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)

    save_clean_data(X_train, "X_train")
    save_clean_data(X_test, "X_test")
    save_clean_data(y_train, "y_train")
    save_clean_data(y_test, "y_test")

    print("Dataset split complete and saved.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.value_counts().to_dict()}, y_test: {y_test.value_counts().to_dict()}")

if __name__ == "__main__":
    split_dataset()
