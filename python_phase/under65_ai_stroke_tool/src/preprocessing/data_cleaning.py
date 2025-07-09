"""
clean_data.py

This module handles the initial cleaning of the under-65 stroke risk dataset.
It performs basic data inspection, standardizes text fields, removes duplicates,
and saves the cleaned dataset for downstream use.

Functions:
- inspect_data(df): Displays summary info, stats, and missing values.
- standardize_text_fields(df): Cleans and formats all object-type columns.
- remove_duplicates(df): Removes duplicate rows while ignoring the 'id' column.
- clean_data(): Full cleaning pipeline that ties the steps together.

Output:
Saves the cleaned dataset as 'under65_clean_data.csv' in the processed data folder.

Intended Use:
This script can be run independently or imported as a utility within a larger pipeline.

Author: John Medina
Date: 7/8/2025
"""


import utils.data_io as util
import pandas as pd

def inspect_data(df):
    """
    Display basic information about the DataFrame for initial inspection.

    Prints:
    - Data types and non-null counts
    - Summary statistics for numeric columns
    - Count of missing values per column
    """
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

def standardize_text_fields(df):
    """
    Standardize all object (text) columns in a DataFrame by:
    - Stripping leading/trailing whitespace
    - Converting text to lowercase
    - Replacing spaces with underscores

    Parameters:
        df (pd.DataFrame): The DataFrame to process

    Returns:
        pd.DataFrame: The standardized DataFrame
    """
    # Identify the categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(" ", "_")
    print(f"Standardized {len(cat_cols)} text columns: lowercased, trimmed, and replaced spaces with underscore")
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows while ignoring the 'id' column during comparison.

    Parameters:
        df (pd.DataFrame): The DataFrame to deduplicate

    Returns:
        pd.DataFrame: The deduplicated DataFrame
    """
    before_rows = df.shape[0]

    # Drop duplicates ignoring the 'id' column
    df = df[df.drop(columns=['id']).duplicated(keep='first') == False].copy()

    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows.")
    return df

def clean_data():
    """
    End-to-end data cleaning function.

    Steps:
    - Loads the raw dataset using util.load_raw_data()
    - Performs basic inspection (info, describe, missing values)
    - Standardizes text fields (categorical columns)
    - Removes duplicate rows (ignoring 'id')
    - Saves the cleaned dataset to /data/processed/under65_clean_data.csv
    """
    # Load the raw data
    df = util.load_raw_data()

    # Initial inspection (shape, nulls, summary stats)
    inspect_data(df)

    # Standardize all text-based categorical features
    df = standardize_text_fields(df)

    # Remove duplicate records (ignoring 'id')
    df = remove_duplicates(df)

    # Save cleaned dataset for downstream use
    util.save_clean_data(df, "under65_clean_data")

if __name__ == '__main__':
    clean_data()
