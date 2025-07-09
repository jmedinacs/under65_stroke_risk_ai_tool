'''
Created on Jul 8, 2025

@author: jarpy
'''
import utils.data_io as util

def inspect_data(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

def standardize_text_fields(df):
    """
    Standardize all object (text) columns in a DataFrame by:
    - Stripping whitespace
    - Converting to lowercase
    - Replacing spaces with underscores

    Parameters:
    df (pd.DataFrame): The DataFrame to process

    Returns:
    pd.DataFrame: The standardized DataFrame
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(" ","_")
    print(f"Standardized {len(cat_cols)} text columns: lowercased, trimmed, and replaced spaces with underscore")
    return df

def remove_duplicates(df):
    before_rows = df.shape[0]
    
    # Drop duplicates ignoring the 'id' column
    df = df[df.drop(columns=['id']).duplicated(keep='first')==False].copy()
    
    after_rows = df.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows.")
    return df

def clean_data():
    """ """
    # Load the raw data
    df = util.load_raw_data()
    inspect_data(df)
    df = standardize_text_fields(df)
    df = remove_duplicates(df)
    util.save_clean_data(df)
    

if __name__ == '__main__':
    clean_data()