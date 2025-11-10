from sklearn.model_selection import cross_val_score, StratifiedKFold

def evaluate_model(model, X, y, n_splits=5, scoring="balanced_accuracy", random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores
