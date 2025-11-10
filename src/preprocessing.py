import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def get_feature_types(X: pd.DataFrame):
    """Identify numeric and categorical columns in the DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing two lists - numeric columns and categorical columns.
    """
    # Numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    return num_cols, cat_cols

def create_preprocessor(num_cols: list, cat_cols: list):
    """Create a preprocessing pipeline for numeric and categorical features.

    Args:
        num_cols (list): List of numeric column names.
        cat_cols (list): List of categorical column names. 
    """
    numeric_transformer = SimpleImputer(strategy="median")
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    
    return preprocessor