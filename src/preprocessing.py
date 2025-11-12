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
        ("imputer", SimpleImputer(strategy="constant", fill_value='Missing') ),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    
    return preprocessor

def basic_feature_filtering(X_train: pd.DataFrame, X_test: pd.DataFrame, sparse_threshold: float = 0.10, high_cardinality_threshold: int = 900):
    """
    Basic feature cleaning:
      - drops completely empty;
      - the drops are very "leaky" (< sparse_threshold filled);
      - drops categorical ones with n_unique > high_cardinality_threshold.
    Returns:
      X_train_clean, X_test_clean, cols_to_drop.
    """
    n_raws = len(X_train)
    non_null_ratio = X_train.notnull().sum() / n_raws
    cols_very_sparse = non_null_ratio[non_null_ratio < sparse_threshold].index
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    nunique_cat = X_train[cat_cols].nunique()
    cols_high_cardinality = nunique_cat[nunique_cat > high_cardinality_threshold].index
    cols_to_drop = sorted(set(cols_very_sparse) | set(cols_high_cardinality))

    X_train_clean = X_train.drop(columns=cols_to_drop)
    X_test_clean = X_test.drop(columns=cols_to_drop)

    return X_train_clean, X_test_clean, cols_to_drop

