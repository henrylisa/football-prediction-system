from pathlib import Path
import joblib
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "home_attack","home_defence","home_points","home_xgf","home_xga",
    "away_attack","away_defence","away_points","away_xgf","away_xga",
    "goal_diff_form"
]

class Predictor:
    def __init__(self):
        self.clf = Pipeline([
            ("scale", StandardScaler()),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=42
            ))
        ])

    def fit(self, X, y):
        self.clf.fit(X[FEATURES], y)
        return self

    def save(self, path="models/model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, path="models/model.joblib"):
        self.clf = joblib.load(path)
        return self

def poisson_markets(lam_h, lam_a, max_goals=8):
    ph = poisson.pmf(np.arange(max_goals+1), lam_h)
    pa = poisson.pmf(np.arange(max_goals+1), lam_a)
    matrix = np.outer(ph, pa)
    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()
    over25 = sum(matrix[i,j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j >= 3)
    btts = sum(matrix[i,j] for i in range(1,max_goals+1) for j in range(1,max_goals+1))
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {
        "home":float(home), "draw":float(draw), "away":float(away),
        "over25":float(over25), "btts":float(btts),
        "score":f"{idx[0]}-{idx[1]}"
    }

def expected_goals(row):
    # Transparent baseline; replace with trained goal-rate models when sufficient xG history exists.
    lh = max(0.15, 0.58*row.home_xgf + 0.42*row.away_xga + 0.20)
    la = max(0.15, 0.58*row.away_xgf + 0.42*row.home_xga)
    return lh, la
