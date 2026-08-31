import argparse
import yaml

from .db import init_db
from .collect import collect_date_range
from .predict import predict_upcoming
from .report import make_report
from .backtest import run as backtest_run
from .historical import collect_historical


def cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "init-db",
            "collect",
            "historical",
            "train",
            "predict",
            "report",
            "backtest",
            "daily",
        ],
    )

    args = parser.parse_args()

    config = cfg()
    leagues = config["leagues"]

    if args.command == "init-db":
        init_db()

    elif args.command == "collect":
        init_db()
        print(
            "Collected:",
            collect_date_range(leagues, 2)
        )

    elif args.command == "historical":
        init_db()

        seasons = [2022, 2023, 2024]

        print(
            "Collecting historical data for seasons:",
            seasons
        )

        total = collect_historical(
            leagues,
            seasons
        )

        print(
            "Historical collection complete:",
            total,
            "fixtures"
        )

    elif args.command == "train":
        print(
            "Training hook: historical CSV ingestion "
            "should populate fixtures first."
        )
        print(
            "The actual ML training pipeline will be "
            "implemented after historical data collection."
        )

    elif args.command == "predict":
        predictions = predict_upcoming()
        print(
            predictions.to_string(index=False)
        )

    elif args.command == "report":
        print(
            make_report(
                config["report"]["output_dir"],
                config["report"]["top_n"],
            )
        )

    elif args.command == "backtest":
        backtest_run()

    elif args.command == "daily":
        init_db()

        print(
            "Collected:",
            collect_date_range(leagues, 2)
        )

        predictions = predict_upcoming()

        print(
            "Predictions:",
            len(predictions)
        )

        print(
            "Report:",
            make_report(
                config["report"]["output_dir"],
                config["report"]["top_n"],
            )
        )


if __name__ == "__main__":
    main()