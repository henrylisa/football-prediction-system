from datetime import datetime
import pandas as pd
from .db import connect
from .model import poisson_markets, expected_goals
from .features import add_rolling_features

def load_matches():
    with connect() as c:
        return pd.read_sql_query("""
        SELECT f.*, s.home_xg, s.away_xg
        FROM fixtures f LEFT JOIN fixture_stats s ON f.fixture_id=s.fixture_id
        ORDER BY kickoff
        """, c)

def predict_upcoming():
    df=load_matches()
    if df.empty: return pd.DataFrame()
    feat=add_rolling_features(df)
    merged=df.merge(feat,on="fixture_id")
    upcoming=merged[merged.home_goals.isna()].copy()
    out=[]
    for _,r in upcoming.iterrows():
        lh,la=expected_goals(r)
        p=poisson_markets(lh,la)
        vals=(p["home"],p["draw"],p["away"])
        label=["HOME","DRAW","AWAY"][int(max(range(3),key=lambda i:vals[i]))]
        conf="HIGH" if max(vals)>=.55 else "MEDIUM" if max(vals)>=.42 else "LOW"
        out.append({
            "fixture_id":r.fixture_id,
            "kickoff":r.kickoff,
            "home":r.home_id, "away":r.away_id,
            "home_prob":p["home"],"draw_prob":p["draw"],"away_prob":p["away"],
            "over25_prob":p["over25"],"btts_yes_prob":p["btts"],
            "expected_home_goals":lh,"expected_away_goals":la,
            "most_likely_score":p["score"],"prediction":label,"confidence":conf
        })
    result=pd.DataFrame(out)
    with connect() as c:
        for _,r in result.iterrows():
            c.execute("""INSERT OR REPLACE INTO predictions
            (fixture_id,model_version,home_prob,draw_prob,away_prob,over25_prob,btts_yes_prob,
             expected_home_goals,expected_away_goals,most_likely_score,confidence)
             VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (
                r.fixture_id,"v1.0",r.home_prob,r.draw_prob,r.away_prob,r.over25_prob,
                r.btts_yes_prob,r.expected_home_goals,r.expected_away_goals,
                r.most_likely_score,r.confidence))
        c.commit()
    return result
