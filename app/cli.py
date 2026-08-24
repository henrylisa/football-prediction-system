import argparse, yaml
from .db import init_db
from .collect import collect_today, collect_date_range
from .predict import predict_upcoming
from .report import make_report
from .backtest import run as backtest_run

def cfg():
    with open("config.yaml",encoding="utf-8") as f: return yaml.safe_load(f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command", choices=["init-db","collect","train","predict","report","backtest","daily"])
    a=p.parse_args()
    c=cfg(); leagues=c["leagues"]
    if a.command=="init-db": init_db()
    elif a.command=="collect":
        init_db(); print("Collected:",collect_date_range(leagues,2))
    elif a.command=="train":
        print("Training hook: historical CSV ingestion should populate fixtures first.")
        print("Use Football-Data CSVs as the historical source, then extend app/model.py with a trained classifier.")
    elif a.command=="predict":
        print(predict_upcoming().to_string(index=False))
    elif a.command=="report":
        print(make_report(c["report"]["output_dir"],c["report"]["top_n"]))
    elif a.command=="backtest": backtest_run()
    elif a.command=="daily":
        init_db()
        print("Collected:",collect_date_range(leagues,2))
        print("Predictions:",len(predict_upcoming()))
        print("Report:",make_report(c["report"]["output_dir"],c["report"]["top_n"]))

if __name__=="__main__":
    main()
