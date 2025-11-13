import os
import pandas as pd

from src.pipeline import create_full_pipeline
from src.evaluation import evaluate_model
from src.preprocessing import basic_feature_filtering

DATA_DIR = "data"
SUBMISSIONS_DIR = "submissions"

def load_data():
    """Loads train and test from the data folder/."""
    train_path = os.path.join(DATA_DIR, "final_proj_data.csv")
    test_path = os.path.join(DATA_DIR, "final_proj_test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y = train["y"]
    X = train.drop(columns=["y"])

    X_clean, test_clean, _ = basic_feature_filtering(X, 
                                                       test, 
                                                       sparse_threshold=0.10, 
                                                       high_cardinality_threshold=900,
    )

    return X_clean, y, test_clean

def run_cv(X, y):
    """Creates a pipeline and calculates balanced_accuracy through CV."""
    # тут можна поміняти модель: "rf" або "gb"
    pipeline = create_full_pipeline(X, model_name="rf")

    scores = evaluate_model(
        pipeline,
        X,
        y,
        n_splits=5,
        scoring="balanced_accuracy",
        random_state=42,
    )

    print("CV scores:", scores)
    print(f"Mean balanced_accuracy: {scores.mean() * 100:.2f}%")
    print(f"Std (spread across folds): {scores.std():.4f}")

def train_and_predict(X, y, X_test):
    """Trains the pipeline on all train data and returns predictions for the test."""
    pipeline = create_full_pipeline(X, model_name="rf")

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)

    return y_pred

def save_submission(y_pred, filename="submission.csv"):
    """Saves the file for Kaggle to the folder submissions/."""
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    index = range(len(y_pred))

    submission = pd.DataFrame({
        "index": index,
        "y": y_pred.astype(int)
    })

    out_path = os.path.join(SUBMISSIONS_DIR, filename)
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")

def main():
    # 1. Uploading data
    X, y, X_test = load_data()

    # 2. We estimate the model using CV
    print("Running cross-validation...")
    run_cv(X, y)

    # 3. We train on all the data and make a prediction for the test
    print("Training on full train and predicting test...")
    y_pred = train_and_predict(X, y, X_test)

    # 4. We are saving the summit
    save_submission(y_pred, filename="submission_best.csv")


if __name__ == "__main__":
    main()