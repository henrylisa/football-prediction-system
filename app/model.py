import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import build_training_data


FEATURES = [
    "home_elo",
    "away_elo",
    "elo_difference",

    "home_points_avg",
    "away_points_avg",

    "home_goals_for_avg",
    "away_goals_for_avg",

    "home_goals_against_avg",
    "away_goals_against_avg",

    "home_goal_difference_avg",
    "away_goal_difference_avg",

    "home_win_rate",
    "away_win_rate",

    # Home-specific / away-specific form
    "home_home_points_avg",
    "away_away_points_avg",

    "home_home_goals_for_avg",
    "away_away_goals_for_avg",

    "home_home_goals_against_avg",
    "away_away_goals_against_avg",

    "home_home_win_rate",
    "away_away_win_rate",
]


MODEL_PATH = Path("models/1x2_model.joblib")


def train_model():
    df = build_training_data()

    train = df[df["season"] <= 2023].copy()
    test = df[df["season"] == 2024].copy()

    if len(train) < 100:
        raise RuntimeError(
            f"Not enough training data: {len(train)}"
        )

    if test.empty:
        raise RuntimeError(
            "No 2024 test data available."
        )

    X_train = train[FEATURES]
    y_train = train["result"]

    X_test = test[FEATURES]
    y_test = test["result"]

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(
        f"Training matches: {len(train)}"
    )

    print(
        f"Test matches: {len(test)}"
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1, 2]
    )

    print()
    print("1X2 MODEL RESULTS")
    print("=================")
    print(
        f"Accuracy: {accuracy:.4f}"
    )
    print(
        f"Log loss: {loss:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Home",
                "Draw",
                "Away",
            ],
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    return model


if __name__ == "__main__":
    train_model()