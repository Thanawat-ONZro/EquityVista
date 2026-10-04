import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def make_direction(y_close):
    dy = np.diff(y_close, prepend=y_close[0])
    return np.where(dy > 1e-6, 1, np.where(dy < -1e-6, 2, 0))

def train_rf(X, y_dir, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_dir, test_size=test_size, random_state=seed, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    return {
        "accuracy": accuracy,
        "model": model,
        "probs": probs,
        "classes": model.classes_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
