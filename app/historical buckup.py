import pandas as pd
from pathlib import Path
from .db import connect

def import_football_data(csv_path, league_id, season):
    df=pd.read_csv(csv_path)
    required={"HomeTeam","AwayTeam","FTHG","FTAG"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {missing}")
    with connect() as c:
        team_cache={}
        def tid(name):
            if name not in team_cache:
                row=c.execute("SELECT team_id FROM teams WHERE name=?", (name,)).fetchone()
                if row: team_cache[name]=row[0]
                else:
                    c.execute("INSERT INTO teams(name) VALUES(?)",(name,))
                    team_cache[name]=c.execute("SELECT last_insert_rowid()").fetchone()[0]
            return team_cache[name]
        for i,r in df.iterrows():
            if pd.isna(r.FTHG) or pd.isna(r.FTAG): continue
            home,away=tid(r.HomeTeam),tid(r.AwayTeam)
            kickoff=pd.to_datetime(r.get("Date"),dayfirst=True,errors="coerce")
            kickoff=(kickoff.isoformat() if pd.notna(kickoff) else f"{season}-{i+1:02d}-01T00:00:00")
            fixture_id=int(f"{league_id}{season}{i:06d}")
            c.execute("""INSERT OR REPLACE INTO fixtures
            (fixture_id,league_id,season,kickoff,home_id,away_id,home_goals,away_goals,status)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (fixture_id,league_id,season,kickoff,home,away,int(r.FTHG),int(r.FTAG),"FT"))
        c.commit()
