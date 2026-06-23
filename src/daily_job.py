from __future__ import annotations

import sys
from pathlib import Path

import schedule
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.run_pipeline import run_pipeline


def job() -> None:
    run_pipeline(period="5y")


schedule.every().day.at("18:30").do(job)

if __name__ == "__main__":
    print("Daily PSX forecasting job started. Keep this process running.")
    while True:
        schedule.run_pending()
        time.sleep(60)
