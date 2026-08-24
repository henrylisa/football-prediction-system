import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from .db import connect
from .features import add_rolling_features
from .model import poisson_markets, expected_goals

def run():
    with connect() as c:
        df=pd.read_sql_query("""
        SELECT f.*, s.home_xg, s.away_xg
        FROM fixtures f LEFT JOIN fixture_stats s ON f.fixture_id=s.fixture_id
        WHERE f.home_goals IS NOT NULL AND f.away_goals IS NOT NULL
        ORDER BY kickoff
        """, c)
    if len(df)<50:
        print("Not enough completed matches for a meaningful backtest.")
        return
    feat=add_rolling_features(df)
    d=df.merge(feat,on="fixture_id")
    y=[]; probs=[]; btts_y=[]; btts_p=[]; roi=[]; stakes=0; profit=0
    for _,r in d.iterrows():
        lh,la=expected_goals(r); p=poisson_markets(lh,la)
        actual=0 if r.home_goals>r.away_goals else 1 if r.home_goals==r.away_goals else 2
        y.append(actual); probs.append([p["home"],p["draw"],p["away"]])
        by=int(r.home_goals>0 and r.away_goals>0); btts_y.append(by); btts_p.append(p["btts"])
    y=np.array(y); probs=np.array(probs)
    pred=probs.argmax(axis=1)
    acc=(pred==y).mean()
    ll=log_loss(y,probs,labels=[0,1,2])
    br=brier_score_loss((y==0).astype(int),probs[:,0])
    print(f"Samples: {len(y)}")
    print(f"1X2 accuracy: {acc:.3f}")
    print(f"Log loss: {ll:.4f}")
    print(f"Home-win Brier: {br:.4f}")
    return {"samples":len(y),"accuracy":acc,"log_loss":ll,"brier":br}
