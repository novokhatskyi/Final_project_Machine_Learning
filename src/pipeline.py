from sklearn.pipeline import Pipeline
from .preprocessing import get_feature_types, create_preprocessor
from .modeling import get_model

def create_full_pipeline(X, model_name, **model_kwargs):
    """Create a full pipeline with preprocessing and modeling."""
    num_cols, cat_cols = get_feature_types(X)
    preprocessor = create_preprocessor(num_cols, cat_cols)
    model = get_model(model_name, **model_kwargs)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return full_pipeline