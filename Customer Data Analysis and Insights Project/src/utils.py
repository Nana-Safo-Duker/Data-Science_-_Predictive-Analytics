"""
Utility functions for data analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(data_path):
    """
    Load dataset from CSV file
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    return pd.read_csv(data_path)

def save_results(data, output_path, filename):
    """
    Save analysis results to file
    
    Parameters:
    -----------
    data : pd.DataFrame or dict
        Data to save
    output_path : str or Path
        Output directory path
    filename : str
        Output filename
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path / filename, index=False)
    else:
        import json
        with open(output_path / filename, 'w') as f:
            json.dump(data, f, indent=2)

def get_data_summary(df):
    """
    Get summary statistics for the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Summary statistics
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum()
    }

