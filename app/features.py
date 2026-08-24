import numpy as np
import pandas as pd

def add_rolling_features(matches, recent=8):
    rows = []
    matches = matches.sort_values("kickoff").copy()

    history = {}
    for _, m in matches.iterrows():
        h, a = int(m.home_id), int(m.away_id)

        def stats(t):
            x = history.get(t, [])
            if not x:
                return {"gf":1.2,"ga":1.2,"pts":1.0,"xgf":1.2,"xga":1.2}
            z = pd.DataFrame(x[-recent:])
            return z.mean().to_dict()

        hs, as_ = stats(h), stats(a)

        rows.append({
            "fixture_id": m.fixture_id,
            "home_attack": hs["gf"],
            "home_defence": hs["ga"],
            "home_points": hs["pts"],
            "home_xgf": hs["xgf"],
            "home_xga": hs["xga"],
            "away_attack": as_["gf"],
            "away_defence": as_["ga"],
            "away_points": as_["pts"],
            "away_xgf": as_["xgf"],
            "away_xga": as_["xga"],
            "goal_diff_form": (hs["gf"]-hs["ga"]) - (as_["gf"]-as_["ga"]),
        })

        hg, ag = m.home_goals, m.away_goals
        if pd.notna(hg) and pd.notna(ag):
            history.setdefault(h, []).append({
                "gf":hg, "ga":ag, "pts":3 if hg>ag else 1 if hg==ag else 0,
                "xgf": m.home_xg if pd.notna(m.home_xg) else hg,
                "xga": m.away_xg if pd.notna(m.away_xg) else ag
            })
            history.setdefault(a, []).append({
                "gf":ag, "ga":hg, "pts":3 if ag>hg else 1 if hg==ag else 0,
                "xgf": m.away_xg if pd.notna(m.away_xg) else ag,
                "xga": m.home_xg if pd.notna(m.home_xg) else hg
            })
    return pd.DataFrame(rows)
