# Automated Football Prediction System

A production-oriented Python starter for daily football predictions.

## Stack
- Python 3.11+
- SQLite (easy local start; swap to PostgreSQL later)
- API-Football for current fixtures/statistics/odds
- Football-Data CSV for historical training data
- pandas / NumPy / scikit-learn
- Poisson scoreline model + Gradient Boosting ensemble
- Backtesting with chronological, leakage-safe splits
- Daily Markdown/CSV report

API-Football currently documents fixtures, standings, fixture statistics, injuries, odds and predictions, with coverage varying by competition. Keep the API key in `.env`, never in source code.

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Put your API-Football key in `.env`.

Edit `config.yaml` to choose leagues. Then:

```bash
python -m app.cli init-db
python -m app.cli train
python -m app.cli predict
python -m app.cli report
python -m app.cli backtest
```

Or run the complete daily pipeline:

```bash
python -m app.cli daily
```

## Data model

The database stores:
- leagues
- teams
- fixtures
- fixture statistics
- odds snapshots
- team ratings
- feature snapshots
- predictions
- backtest runs

## Daily automation

Linux cron example:

```cron
15 6 * * * cd /path/to/football_prediction_system && /path/to/.venv/bin/python -m app.cli daily >> logs/daily.log 2>&1
```

For GitHub Actions, see `.github/workflows/daily.yml`.

## Important modelling rule

All features are calculated only from matches that occurred before the target fixture. This prevents future-information leakage.

The model is probabilistic. It does not guarantee winning outcomes. Evaluate calibration, log loss, Brier score, and out-of-sample ROI before trusting any strategy.
