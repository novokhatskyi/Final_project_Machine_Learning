from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_model(name: str = "gb", **kwargs):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            n_jobs=kwargs.get("n_jobs", -1),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=kwargs.get("random_state", 42),
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 3),
            random_state=kwargs.get("random_state", 42),
        )
    raise ValueError(f"Unknown model name: {name}")
