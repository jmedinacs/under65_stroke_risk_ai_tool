"""
data_io.py

This module handles all data input and output operations for the under-65 stroke risk AI tool.
It provides functions to load the raw SQL-exported dataset, load the fully preprocessed version,
and save the final cleaned data for modeling.

Author: John Medina
Date: 7/8/2025
"""

import pandas as pd
import os

def load_raw_data(filepath="../../data/processed/under65_sql_processed_data.csv"):
    """
    Load the raw under-65 dataset exported from SQL, prior to imputation or transformation.

    Parameters:
    filepath (str): Relative path to the raw CSV file

    Returns:
    pd.DataFrame: Loaded raw dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def load_clean_data(filename="default"):
    """
    Load the final cleaned dataset (after imputation and log transformation).

    Parameters:
    filepath (str): Relative path to the cleaned CSV file

    Returns:
    pd.DataFrame: Loaded preprocessed dataset
    """
    
    filepath=f"../../data/processed/{filename}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def save_clean_data(df, filename="default"):
    """
    Save the cleaned and preprocessed dataset to the processed data folder.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): File name (without extension) to save under /data/processed/.
    """
    
    filepath=f"../../data/processed/{filename}.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to: {filepath}")
