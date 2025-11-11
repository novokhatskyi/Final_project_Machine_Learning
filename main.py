import os
import pandas as pd

from src.pipeline import create_full_pipeline
from src.evaluation import evaluate_model
from sklearn.model_selection import cross_val_score, StratifiedKFold

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

    return X, y, test

def run_cv(X, y):
    """Creates a pipeline and calculates balanced_accuracy through CV."""
    # тут можна поміняти модель: "rf" або "gb"
    pipeline = create_full_pipeline(X, model_name="rf", n_estimators=300)

    scores = evaluate_model(
        pipeline,
        X,
        y,
        n_splits=5,
        scoring="balanced_accuracy",
        random_state=42,
    )

    print("CV scores:", scores)
    print("Mean balanced_accuracy:", scores.mean())
    print("Std:", scores.std())

def train_and_predict(X, y, X_test):
    """Trains the pipeline on all train data and returns predictions for the test."""
    pipeline = create_full_pipeline(X, model_name="rf", n_estimators=300)

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
    pipeline = create_full_pipeline(X, model_name="gb", n_estimators=300)

    # 2. We estimate the model using CV
    print("Running cross-validation...")
    run_cv(X, y)

    # 3. We train on all the data and make a prediction for the test
    print("Training on full train and predicting test...")
    y_pred = train_and_predict(X, y, X_test)

    # 4. We are saving the summit
    save_submission(y_pred, filename="submission_rf_baseline.csv")


if __name__ == "__main__":
    main()