import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from .features import build_training_data
from .model import FEATURES, train_model


def run():
    print("Building historical features...")

    df = build_training_data()

    # Train on 2022-2023
    train = df[df["season"] <= 2023].copy()

    # Strict out-of-sample test on 2024
    test = df[df["season"] == 2024].copy()

    if train.empty:
        print("No training data available.")
        return

    if test.empty:
        print("No 2024 test data available.")
        return

    print(f"Training samples: {len(train)}")
    print(f"Backtest samples: {len(test)}")

    # Train model
    model = train_model()

    X_test = test[FEATURES]
    y_test = test["result"]

    # Generate probabilities
    probabilities = model.predict_proba(X_test)

    predictions = model.predict(X_test)

    # Ensure probability columns are in Home/Draw/Away order
    classes = list(model.named_steps["classifier"].classes_)

    probability_matrix = np.zeros(
        (len(test), 3)
    )

    for i, cls in enumerate(classes):
        probability_matrix[:, int(cls)] = probabilities[:, i]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    logloss = log_loss(
        y_test,
        probability_matrix,
        labels=[0, 1, 2]
    )

    # Multiclass Brier score
    brier = np.mean(
        np.sum(
            (
                probability_matrix
                - np.eye(3)[y_test.astype(int)]
            ) ** 2,
            axis=1
        )
    )

    # Individual class Brier scores
    home_brier = brier_score_loss(
        (y_test == 0).astype(int),
        probability_matrix[:, 0]
    )

    draw_brier = brier_score_loss(
        (y_test == 1).astype(int),
        probability_matrix[:, 1]
    )

    away_brier = brier_score_loss(
        (y_test == 2).astype(int),
        probability_matrix[:, 2]
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    print()
    print("================================")
    print("1X2 TIME-SERIES BACKTEST")
    print("================================")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Log loss: {logloss:.4f}"
    )

    print(
        f"Multiclass Brier: {brier:.4f}"
    )

    print()
    print("Brier scores:")
    print(
        f"Home: {home_brier:.4f}"
    )
    print(
        f"Draw: {draw_brier:.4f}"
    )
    print(
        f"Away: {away_brier:.4f}"
    )

    print()
    print("Confusion matrix")
    print("----------------")
    print(
        "             Pred Home  Pred Draw  Pred Away"
    )

    print(
        f"Actual Home     {matrix[0][0]:5d}"
        f"       {matrix[0][1]:5d}"
        f"       {matrix[0][2]:5d}"
    )

    print(
        f"Actual Draw     {matrix[1][0]:5d}"
        f"       {matrix[1][1]:5d}"
        f"       {matrix[1][2]:5d}"
    )

    print(
        f"Actual Away     {matrix[2][0]:5d}"
        f"       {matrix[2][1]:5d}"
        f"       {matrix[2][2]:5d}"
    )

    # Prediction distribution
    predicted_classes = np.argmax(
        probability_matrix,
        axis=1
    )

    print()
    print("Prediction distribution")
    print("-----------------------")

    print(
        f"Home: {(predicted_classes == 0).sum()}"
    )

    print(
        f"Draw: {(predicted_classes == 1).sum()}"
    )

    print(
        f"Away: {(predicted_classes == 2).sum()}"
    )

    results = test[
        [
            "fixture_id",
            "kickoff",
            "league_id",
            "season",
            "result",
        ]
    ].copy()

    results["home_probability"] = (
        probability_matrix[:, 0]
    )

    results["draw_probability"] = (
        probability_matrix[:, 1]
    )

    results["away_probability"] = (
        probability_matrix[:, 2]
    )

    results["prediction"] = predicted_classes

    results["correct"] = (
        results["prediction"]
        == results["result"]
    )

    print()
    print(
        f"Correct predictions: "
        f"{results['correct'].sum()}/{len(results)}"
    )

    print(
        f"Accuracy check: "
        f"{results['correct'].mean():.4f}"
    )

    return {
        "samples": len(results),
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier": brier,
        "home_brier": home_brier,
        "draw_brier": draw_brier,
        "away_brier": away_brier,
    }


if __name__ == "__main__":
    run()