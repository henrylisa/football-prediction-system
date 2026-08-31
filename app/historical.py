from .api import APIFootball
from .collect import upsert_fixtures


def collect_historical(leagues, seasons):
    """
    Download complete historical seasons for the configured leagues.

    Example:
        collect_historical(leagues, [2022, 2023, 2024])
    """

    api = APIFootball()
    total = 0

    for league in leagues:
        league_id = league["id"]
        league_name = league["name"]

        for season in seasons:
            print(
                f"Collecting {league_name} "
                f"(league={league_id}, season={season})..."
            )

            items = api.fixtures(
                date=None,
                league_id=league_id,
                season=season
            )

            upsert_fixtures(items)

            count = len(items)
            total += count

            print(f"  Added/updated: {count} fixtures")

    print(f"\nTotal fixtures collected: {total}")

    return total