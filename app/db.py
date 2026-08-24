import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def db_path():
    url = os.getenv("DATABASE_URL", "sqlite:///data/predictions.db")
    return Path(url.replace("sqlite:///", ""))

def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS leagues(
    league_id INTEGER PRIMARY KEY, name TEXT, season INTEGER
);
CREATE TABLE IF NOT EXISTS teams(
    team_id INTEGER PRIMARY KEY, name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fixtures(
    fixture_id INTEGER PRIMARY KEY,
    league_id INTEGER, season INTEGER, kickoff TEXT,
    home_id INTEGER, away_id INTEGER,
    home_goals INTEGER, away_goals INTEGER,
    status TEXT,
    venue TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fixture_stats(
    fixture_id INTEGER PRIMARY KEY,
    home_shots INTEGER, away_shots INTEGER,
    home_shots_on_target INTEGER, away_shots_on_target INTEGER,
    home_possession REAL, away_possession REAL,
    home_corners INTEGER, away_corners INTEGER,
    home_fouls INTEGER, away_fouls INTEGER,
    home_xg REAL, away_xg REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS odds(
    fixture_id INTEGER PRIMARY KEY,
    home_odds REAL, draw_odds REAL, away_odds REAL,
    over25_odds REAL, under25_odds REAL,
    btts_yes_odds REAL, btts_no_odds REAL,
    captured_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions(
    fixture_id INTEGER PRIMARY KEY,
    model_version TEXT,
    home_prob REAL, draw_prob REAL, away_prob REAL,
    over25_prob REAL, btts_yes_prob REAL,
    expected_home_goals REAL, expected_away_goals REAL,
    most_likely_score TEXT,
    confidence TEXT,
    value_market TEXT, value_edge REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS backtest_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT DEFAULT CURRENT_TIMESTAMP,
    market TEXT, samples INTEGER, accuracy REAL,
    brier REAL, log_loss REAL, roi REAL
);
"""

def init_db():
    with connect() as c:
        c.executescript(SCHEMA)
        c.commit()
