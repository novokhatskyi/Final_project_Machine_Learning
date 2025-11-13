from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(name, **kwargs):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            n_jobs=kwargs.get("n_jobs", -1),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 12),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            criterion=kwargs.get("criterion", "log_loss"),
            max_features=kwargs.get("max_features", 0.7),
            max_samples=kwargs.get("max_samples", 0.7),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 600),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 5),
            min_samples_leaf=kwargs.get("min_samples_leaf", 30),
            random_state=kwargs.get("random_state", 42),
        )
    raise ValueError(f"Unknown model name: {name}")
