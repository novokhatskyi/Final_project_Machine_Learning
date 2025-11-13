# Final Project — Machine Learning (Kaggle, Binary Classification)

**Short description.** Reproducible ML pipeline for an anonymized telecom dataset: feature filtering, preprocessing, Gradient Boosting training, 5-fold CV (balanced accuracy) and Kaggle submission.

---

## Goals
- Solve a binary classification task with a clear, interpretable pipeline.
- Provide training, evaluation (5-fold CV) and inference (submission).
- Ensure reproducibility (fixed params, pinned requirements).

---

## Data
```
data/
├─ final_proj_data.csv      # train (contains y)
└─ final_proj_test.csv      # test (no y)
```

**Submission format**
```csv
index,y
0,0
1,1
...
```

---

## Quick Start

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd Final_project_Machine_Learning
   ```

2. **Create environment**
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # (if you need to generate)
   # python -m pip freeze > requirements.txt
   ```

4. **Prepare data**
   - Place CSV files into `data/` as shown above.

5. **Run cross-validation and training**
   ```bash
   python main.py
   ```
   The script prints CV scores and saves a submission to:
   ```
   submissions/submission_gb_best.csv
   ```

---

## Project Structure
```
.
├─ data/
│  ├─ final_proj_data.csv
│  └─ final_proj_test.csv
├─ submissions/
│  └─ submission_gb_best.csv
├─ src/
│  ├─ preprocessing.py      # basic_feature_filtering, create_preprocessor, get_feature_types
│  ├─ modeling.py           # get_model("gb" | "rf") with GB defaults
│  ├─ pipeline.py           # create_full_pipeline(...)
│  └─ evaluation.py         # evaluate_model(...), StratifiedKFold + cross_val_score
├─ notebooks/               # optional EDA/experiments
├─ main.py                  # load → clean → CV → train → submission
├─ requirements.txt
└─ README.md
```

---

## Preprocessing & Model

- **Feature filtering** (`basic_feature_filtering`)
  - drop columns with **< 10%** non-missing values
  - drop categorical columns with **> 900** unique values (likely IDs)
- **Preprocessing** (`create_preprocessor`)
  - numeric → `SimpleImputer(strategy="median")`
  - categorical → `SimpleImputer(strategy="constant", fill_value="Missing")`
    + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- **Model** (`get_model("gb")`)
  - `GradientBoostingClassifier` with defaults:
    - `n_estimators=600`, `learning_rate=0.1`, `max_depth=5`, `min_samples_leaf=30`, `random_state=42`

---

## Results (5-fold CV)

| Fold | Balanced Accuracy |
|-----:|------------------:|
| 1    | 0.7939            |
| 2    | 0.7847            |
| 3    | 0.7966            |
| 4    | 0.8037            |
| 5    | 0.7833            |
| **Mean** | **0.7924**    |
| **Std**  | **0.0076**    |

---

## Configuration
- `src/preprocessing.py` → `basic_feature_filtering(...)`
  - `sparse_threshold=0.10`
  - `high_cardinality_threshold=900`
- `src/modeling.py` → `get_model("gb")`
  - `n_estimators=600`, `learning_rate=0.1`, `max_depth=5`, `min_samples_leaf=30`

---

## Notes
- `OneHotEncoder(handle_unknown="ignore")` guards against unseen categories in test.
- Tree-based boosting works well with one-hot features; scaling/PCA not required.
- Keep repo UTF-8; preview README in VS Code with `Cmd/Ctrl+Shift+V`.

---

## Author
Oleksandr Novokhatskyi
