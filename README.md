Final_project_Machine_Learning (Kaggle)

Anonymized telecom marketing dataset. Goal — maximize Balanced Accuracy and produce a valid Kaggle submission.

🔎 Overview

Task: binary classification (y ∈ {0,1})

Data size: ~10,000 rows, ~230 features (~190 numeric, ~40 categorical)

Metric: balanced_accuracy

Platform: Kaggle

📂 Data
data/
├─ final_proj_data.csv    # train (contains y)
└─ final_proj_test.csv    # test (no y)


Submission format:

index,y
0,0
1,1
...

🧠 Approach

Feature filtering (basic_feature_filtering)

drop columns with < 10% non-missing values

drop categorical columns with > 900 unique values (likely IDs)

Preprocessing (create_preprocessor)

numeric → SimpleImputer(strategy="median")

categorical → SimpleImputer(strategy="constant", fill_value="Missing")

OneHotEncoder(handle_unknown="ignore", sparse_output=False)

Model (get_model("gb"))

GradientBoostingClassifier with tuned defaults:

n_estimators=600, learning_rate=0.1, max_depth=5, min_samples_leaf=30, random_state=42

Validation

StratifiedKFold(n_splits=5), scoring="balanced_accuracy"

Final training

train on full train set and generate submission

📈 Results (5-fold CV)
Fold	Score
1	0.7939
2	0.7847
3	0.7966
4	0.8037
5	0.7833

Mean balanced_accuracy: 0.7924
Std: 0.0076

🗂️ Project Structure
.
├─ data/
│  ├─ final_proj_data.csv
│  └─ final_proj_test.csv
├─ submissions/
│  └─ submission_gb_best.csv
├─ src/
│  ├─ preprocessing.py      # basic_feature_filtering, create_preprocessor, get_feature_types
│  ├─ modeling.py           # get_model("rf" | "gb") with GB defaults
│  ├─ pipeline.py           # create_full_pipeline(...)
│  └─ evaluation.py         # evaluate_model(...), StratifiedKFold + cross_val_score
├─ notebooks/               # optional EDA/experiments
├─ main.py                  # load → clean → CV → train → submission
├─ requirements.txt
└─ README.md

⚙️ Setup

Create a virtual environment and install dependencies.

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt


If pipreqs throws a UnicodeDecodeError, just freeze the environment:

python -m pip freeze > requirements.txt

🚀 Run

Place CSVs in data/

Execute:

python main.py


The script will:

filter features,

run 5-fold CV (balanced accuracy),

train on full train,

save a submission to submissions/submission_gb_best.csv.

🔧 Configuration

src/preprocessing.py → basic_feature_filtering(...)

sparse_threshold=0.10

high_cardinality_threshold=900

src/modeling.py → get_model("gb")

n_estimators=600, learning_rate=0.1, max_depth=5, min_samples_leaf=30

📝 Notes

handle_unknown="ignore" in OneHotEncoder ensures robustness to unseen categories in test.

Tree-based boosting works well with one-hot features; scaling/PCA is not required here.

All steps are consistent between train/test via Pipeline for honest evaluation.