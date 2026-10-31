import argparse
import importlib
import json
import logging
from pathlib import Path

from config.settings import ROOT, initialize_directories
from src.data.database import Database
from src.data.audit import audit_html
from src.data.scraper import sync_database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "verify", "sync", "audit"])
    parser.add_argument("--html", type=Path, help="Saved official HTML for independent offline audit")
    args = parser.parse_args()
    initialize_directories()
    logging.basicConfig(level=logging.INFO, filename=ROOT / "logs" / "application.log")
    if args.command == "verify":
        for name in ("numpy", "pandas", "scipy", "sklearn", "streamlit", "plotly",
                     "requests", "bs4", "lxml", "joblib", "pytest", "sqlite3"):
            importlib.import_module(name)
        print("All required dependencies available")
    elif args.command == "init":
        Database()
        print("SQLite schema initialized")
    elif args.command == "sync":
        print(json.dumps(sync_database(Database(), force=True)))
    else:
        db = Database()
        archive = args.html or Path(db.state()["raw_archive"])
        result = audit_html(archive.read_text(encoding="utf-8-sig"), db)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
