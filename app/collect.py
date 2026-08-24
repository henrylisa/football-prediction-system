from datetime import date, timedelta, datetime, timezone
import pandas as pd
from .api import APIFootball
from .db import connect

def upsert_fixtures(items):
    with connect() as c:
        for x in items:
            teams = x["teams"]; goals=x.get("goals") or {}
            fixture=x["fixture"]
            c.execute("""INSERT OR REPLACE INTO fixtures
            (fixture_id,league_id,season,kickoff,home_id,away_id,home_goals,away_goals,status,venue)
            VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                fixture["id"], x["league"]["id"], x["league"]["season"],
                fixture["date"], teams["home"]["id"], teams["away"]["id"],
                goals.get("home"), goals.get("away"),
                x["fixture"]["status"]["short"],
                (fixture.get("venue") or {}).get("name")
            ))
            c.execute("INSERT OR REPLACE INTO teams(team_id,name) VALUES(?,?)",
                      (teams["home"]["id"],teams["home"]["name"]))
            c.execute("INSERT OR REPLACE INTO teams(team_id,name) VALUES(?,?)",
                      (teams["away"]["id"],teams["away"]["name"]))
        c.commit()

def collect_today(leagues):
    api=APIFootball()
    d=date.today().isoformat()
    all_items=[]
    for league in leagues:
        all_items += api.fixtures(d, league["id"], league["season"])
    upsert_fixtures(all_items)
    return len(all_items)

def collect_date_range(leagues, days=2):
    api=APIFootball()
    total=0
    for offset in range(days+1):
        d=(date.today()+timedelta(days=offset)).isoformat()
        for league in leagues:
            items=api.fixtures(d, league["id"], league["season"])
            upsert_fixtures(items); total += len(items)
    return total
