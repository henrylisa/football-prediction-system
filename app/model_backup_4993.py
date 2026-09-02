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
import math


def expected_goals(row):
    """
    Estimate expected goals using recent attacking
    and defensive form.
    """

    home_attack = max(float(row["home_goals_for_avg"]), 0.2)
    away_attack = max(float(row["away_goals_for_avg"]), 0.2)

    home_defence = max(float(row["home_goals_against_avg"]), 0.2)
    away_defence = max(float(row["away_goals_against_avg"]), 0.2)

    home_elo = float(row.get("home_elo", 1500))
    away_elo = float(row.get("away_elo", 1500))

    elo_diff = (home_elo + 60.0) - away_elo

    elo_factor = 1.0 + max(min(elo_diff / 2000.0, 0.20), -0.20)

    home_xg = (
        0.60 * home_attack
        + 0.40 * away_defence
    ) * elo_factor

    away_xg = (
        0.60 * away_attack
        + 0.40 * home_defence
    ) / elo_factor

    home_xg = max(0.15, min(home_xg, 4.0))
    away_xg = max(0.15, min(away_xg, 4.0))

    return home_xg, away_xg


def poisson_probability(goals, expected):
    return (
        math.exp(-expected)
        * expected ** goals
        / math.factorial(goals)
    )


def poisson_markets(home_xg, away_xg, max_goals=8):
    """
    Convert expected goals into 1X2, Over 2.5,
    BTTS and most-likely-score probabilities.
    """

    home_probs = [
        poisson_probability(i, home_xg)
        for i in range(max_goals + 1)
    ]

    away_probs = [
        poisson_probability(i, away_xg)
        for i in range(max_goals + 1)
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over25 = 0.0
    btts = 0.0

    best_score = "0-0"
    best_score_probability = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):

            probability = home_probs[h] * away_probs[a]

            if h > a:
                home_win += probability
            elif h == a:
                draw += probability
            else:
                away_win += probability

            if h + a >= 3:
                over25 += probability

            if h > 0 and a > 0:
                btts += probability

            if probability > best_score_probability:
                best_score_probability = probability
                best_score = f"{h}-{a}"

    total = home_win + draw + away_win

    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "over25": over25,
        "btts": btts,
        "score": best_score,
    }

if __name__ == "__main__":
    train_model()
import math


def expected_goals(row):
    """
    Estimate expected goals using recent attacking
    and defensive form.
    """

    home_attack = max(float(row["home_goals_for_avg"]), 0.2)
    away_attack = max(float(row["away_goals_for_avg"]), 0.2)

    home_defence = max(float(row["home_goals_against_avg"]), 0.2)
    away_defence = max(float(row["away_goals_against_avg"]), 0.2)

    home_elo = float(row.get("home_elo", 1500))
    away_elo = float(row.get("away_elo", 1500))

    elo_diff = (home_elo + 60.0) - away_elo

    elo_factor = 1.0 + max(min(elo_diff / 2000.0, 0.20), -0.20)

    home_xg = (
        0.60 * home_attack
        + 0.40 * away_defence
    ) * elo_factor

    away_xg = (
        0.60 * away_attack
        + 0.40 * home_defence
    ) / elo_factor

    home_xg = max(0.15, min(home_xg, 4.0))
    away_xg = max(0.15, min(away_xg, 4.0))

    return home_xg, away_xg


def poisson_probability(goals, expected):
    return (
        math.exp(-expected)
        * expected ** goals
        / math.factorial(goals)
    )


def poisson_markets(home_xg, away_xg, max_goals=8):
    """
    Convert expected goals into 1X2, Over 2.5,
    BTTS and most-likely-score probabilities.
    """

    home_probs = [
        poisson_probability(i, home_xg)
        for i in range(max_goals + 1)
    ]

    away_probs = [
        poisson_probability(i, away_xg)
        for i in range(max_goals + 1)
    ]

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over25 = 0.0
    btts = 0.0

    best_score = "0-0"
    best_score_probability = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):

            probability = home_probs[h] * away_probs[a]

            if h > a:
                home_win += probability
            elif h == a:
                draw += probability
            else:
                away_win += probability

            if h + a >= 3:
                over25 += probability

            if h > 0 and a > 0:
                btts += probability

            if probability > best_score_probability:
                best_score_probability = probability
                best_score = f"{h}-{a}"

    total = home_win + draw + away_win

    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "over25": over25,
        "btts": btts,
        "score": best_score,
    }