import os
import requests
from dotenv import load_dotenv

load_dotenv()

class APIFootball:
    def __init__(self):
        self.base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
        self.key = os.getenv("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY is missing. Put it in .env")
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": self.key})

    def get(self, endpoint, **params):
        r = self.session.get(f"{self.base}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        return payload.get("response", [])

    def fixtures(self, date, league_id=None, season=None):
        p = {"date": date}
        if league_id: p["league"] = league_id
        if season: p["season"] = season
        return self.get("fixtures", **p)

    def fixture_statistics(self, fixture_id):
        return self.get("fixtures/statistics", fixture=fixture_id)

    def odds(self, fixture_id):
        return self.get("odds", fixture=fixture_id)

    def standings(self, league_id, season):
        return self.get("standings", league=league_id, season=season)

    def injuries(self, league_id, season):
        return self.get("injuries", league=league_id, season=season)
