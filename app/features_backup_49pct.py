from collections import defaultdict, deque

import pandas as pd

from .db import connect


DEFAULT_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE_ELO = 60.0


def load_fixtures():
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                fixture_id,
                league_id,
                season,
                kickoff,
                home_id,
                away_id,
                home_goals,
                away_goals,
                status
            FROM fixtures
            WHERE home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY kickoff ASC
            """
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def form_stats(history, team_id, n=8):
    recent = list(history[team_id])[-n:]

    if not recent:
        return {
            "points": 1.0,
            "gf": 1.0,
            "ga": 1.0,
            "gd": 0.0,
            "win_rate": 0.33,
        }

    points = []
    goals_for = []
    goals_against = []
    wins = 0

    for match in recent:
        if match["team_id"] == team_id:
            gf = match["gf"]
            ga = match["ga"]
        else:
            gf = match["ga"]
            ga = match["gf"]

        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            points.append(3)
            wins += 1
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)

    return {
        "points": sum(points) / len(points),
        "gf": sum(goals_for) / len(goals_for),
        "ga": sum(goals_against) / len(goals_against),
        "gd": (
            sum(goals_for) - sum(goals_against)
        ) / len(goals_for),
        "win_rate": wins / len(recent),
    }


def expected_elo(home_elo, away_elo):
    difference = (
        home_elo
        + HOME_ADVANTAGE_ELO
        - away_elo
    )

    return 1.0 / (
        1.0 + 10 ** (-difference / 400.0)
    )


def update_elo(
    home_elo,
    away_elo,
    home_goals,
    away_goals,
):
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals == away_goals:
        actual = 0.5
    else:
        actual = 0.0

    expected = expected_elo(
        home_elo,
        away_elo
    )

    margin = abs(
        home_goals - away_goals
    )

    # Mild margin-of-victory adjustment
    if margin > 1:
        margin_multiplier = (
            1.0
            + 0.15 * (margin - 1)
        )
    else:
        margin_multiplier = 1.0

    change = (
        K_FACTOR
        * margin_multiplier
        * (actual - expected)
    )

    return (
        home_elo + change,
        away_elo - change,
    )


def build_training_data(recent_matches=8):
    df = load_fixtures()

    if df.empty:
        raise RuntimeError(
            "No finished fixtures found."
        )

    # Team history
    history = defaultdict(
        lambda: deque(
            maxlen=recent_matches
        )
    )

    # Elo ratings
    elo = defaultdict(
        lambda: DEFAULT_ELO
    )

    rows = []

    for _, match in df.iterrows():

        home_id = int(match["home_id"])
        away_id = int(match["away_id"])

        home_elo = elo[home_id]
        away_elo = elo[away_id]

        home_form = form_stats(
            history,
            home_id,
            recent_matches
        )

        away_form = form_stats(
            history,
            away_id,
            recent_matches
        )

        home_goals = int(
            match["home_goals"]
        )

        away_goals = int(
            match["away_goals"]
        )

        if home_goals > away_goals:
            result = 0
        elif home_goals == away_goals:
            result = 1
        else:
            result = 2

        rows.append(
            {
                "fixture_id": int(
                    match["fixture_id"]
                ),

                "kickoff": match["kickoff"],

                "league_id": int(
                    match["league_id"]
                ),

                "season": int(
                    match["season"]
                ),

                "home_elo": home_elo,
                "away_elo": away_elo,

                "elo_difference": (
                    home_elo
                    + HOME_ADVANTAGE_ELO
                    - away_elo
                ),

                "home_points_avg":
                    home_form["points"],

                "away_points_avg":
                    away_form["points"],

                "home_goals_for_avg":
                    home_form["gf"],

                "away_goals_for_avg":
                    away_form["gf"],

                "home_goals_against_avg":
                    home_form["ga"],

                "away_goals_against_avg":
                    away_form["ga"],

                "home_goal_difference_avg":
                    home_form["gd"],

                "away_goal_difference_avg":
                    away_form["gd"],

                "home_win_rate":
                    home_form["win_rate"],

                "away_win_rate":
                    away_form["win_rate"],

                "result": result,
            }
        )

        # IMPORTANT:
        # Update Elo only AFTER the match.
        #
        # This prevents future information
        # from leaking into the prediction.

        new_home_elo, new_away_elo = update_elo(
            home_elo,
            away_elo,
            home_goals,
            away_goals,
        )

        elo[home_id] = new_home_elo
        elo[away_id] = new_away_elo

        # Update historical form AFTER match
        history[home_id].append(
            {
                "team_id": home_id,
                "gf": home_goals,
                "ga": away_goals,
            }
        )

        history[away_id].append(
            {
                "team_id": away_id,
                "gf": away_goals,
                "ga": home_goals,
            }
        )

    return pd.DataFrame(rows)