from pathlib import Path
import pandas as pd
from .db import connect

def make_report(output_dir="reports", top_n=20):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with connect() as c:
        df=pd.read_sql_query("""
        SELECT p.*, th.name home_team, ta.name away_team, f.kickoff
        FROM predictions p
        JOIN fixtures f ON f.fixture_id=p.fixture_id
        JOIN teams th ON th.team_id=f.home_id
        JOIN teams ta ON ta.team_id=f.away_id
        WHERE f.home_goals IS NULL
        ORDER BY p.home_prob DESC
        """, c)
    df=df.head(top_n)
    today=pd.Timestamp.now().strftime("%Y-%m-%d")
    path=Path(output_dir)/f"daily_{today}.md"
    lines=[f"# Daily Football Prediction Report — {today}",
           "", "Probabilities are model estimates, not guarantees.", "",
           "| Match | Home | Draw | Away | O2.5 | BTTS | Score | Confidence |",
           "|---|---:|---:|---:|---:|---:|---|---|"]
    for _,r in df.iterrows():
        lines.append(
            f"| {r.home_team} vs {r.away_team} | "
            f"{r.home_prob:.1%} | {r.draw_prob:.1%} | {r.away_prob:.1%} | "
            f"{r.over25_prob:.1%} | {r.btts_yes_prob:.1%} | "
            f"{r.most_likely_score} | {r.confidence} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(Path(output_dir)/f"daily_{today}.csv", index=False)
    return path
