import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import DATABASE, SOURCE
from src.data.validator import validate_draw


def now():
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path=DATABASE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self):
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("Database is newer than this application")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS lottery_draws (
                    id INTEGER PRIMARY KEY, year INTEGER NOT NULL, week INTEGER NOT NULL,
                    draw_date TEXT UNIQUE, number_1 INTEGER NOT NULL CHECK(number_1 BETWEEN 1 AND 90),
                    number_2 INTEGER NOT NULL, number_3 INTEGER NOT NULL, number_4 INTEGER NOT NULL,
                    number_5 INTEGER NOT NULL CHECK(number_5 BETWEEN 1 AND 90),
                    imported_at TEXT NOT NULL, source TEXT NOT NULL, prizes TEXT NOT NULL,
                    UNIQUE(year,week),
                    CHECK(number_1<number_2 AND number_2<number_3 AND number_3<number_4 AND number_4<number_5));
                CREATE INDEX IF NOT EXISTS draws_chronology ON lottery_draws(year,week);
                CREATE TABLE IF NOT EXISTS model_experiments(
                    id INTEGER PRIMARY KEY, model_name TEXT, model_version TEXT,
                    trained_at TEXT, dataset_size INTEGER, artifact TEXT, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions(
                    id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, prediction_date TEXT,
                    target_draw_date TEXT, combination_number INTEGER, model TEXT,
                    seed INTEGER, metadata TEXT NOT NULL, UNIQUE(batch_id,combination_number));
                CREATE TABLE IF NOT EXISTS system_logs(
                    id INTEGER PRIMARY KEY, created_at TEXT, level TEXT, event TEXT, details TEXT);
                CREATE TABLE IF NOT EXISTS data_quality(
                    id INTEGER PRIMARY KEY, synchronized_at TEXT, row_number INTEGER,
                    severity TEXT, message TEXT, raw TEXT);
                CREATE TABLE IF NOT EXISTS sync_state(key TEXT PRIMARY KEY, value TEXT);
                PRAGMA user_version=1;
            """)

    def draws(self):
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM lottery_draws ORDER BY year,week", conn)

    def state(self):
        with self.connect() as conn:
            return dict(conn.execute("SELECT key,value FROM sync_state").fetchall())

    def data_fingerprint(self):
        with self.connect() as conn:
            rows = conn.execute("""SELECT year,week,draw_date,number_1,number_2,number_3,
                number_4,number_5,prizes FROM lottery_draws ORDER BY year,week""").fetchall()
        canonical = [[*tuple(row)[:8], json.loads(row["prizes"])] for row in rows]
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()

    def set_state(self, values):
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO sync_state VALUES (?,?)",
                             [(k, str(v)) for k, v in values.items()])

    def log(self, level, event, details=""):
        with self.connect() as conn:
            conn.execute("INSERT INTO system_logs(created_at,level,event,details) VALUES(?,?,?,?)",
                         (now(), level, event, str(details)))

    def import_records(self, records, issues, rebuild=False):
        if not records:
            raise ValueError("Cannot import an empty dataset")
        timestamp, imported = now(), 0
        with self.connect() as conn:
            # A rebuild validates against existing draws before changing anything.
            for record in records:
                record = validate_draw(record)
                existing = conn.execute(
                    "SELECT * FROM lottery_draws WHERE (year=? AND week=?) OR (draw_date IS NOT NULL AND draw_date=?)",
                    (record["year"], record["week"], record["draw_date"])).fetchone()
                if existing:
                    old_numbers = [existing[f"number_{i}"] for i in range(1, 6)]
                    if (existing["year"], existing["week"]) != (record["year"], record["week"]):
                        raise ValueError(f"Source date conflicts with saved year/week for {record['draw_date']}")
                    if old_numbers != record["numbers"] or (existing["draw_date"] and record["draw_date"]
                                                          and existing["draw_date"] != record["draw_date"]):
                        raise ValueError(f"Source conflicts with saved drawing {record['year']}/{record['week']}")
                    conn.execute("UPDATE lottery_draws SET draw_date=COALESCE(draw_date,?),prizes=? WHERE id=?",
                                 (record["draw_date"], json.dumps(record["prizes"]), existing["id"]))
                    continue
                conn.execute("""INSERT INTO lottery_draws
                    (year,week,draw_date,number_1,number_2,number_3,number_4,number_5,imported_at,source,prizes)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (record["year"], record["week"], record["draw_date"], *record["numbers"],
                     timestamp, SOURCE, json.dumps(record["prizes"])))
                imported += 1
            conn.execute("DELETE FROM data_quality")
            conn.executemany("""INSERT INTO data_quality
                (synchronized_at,row_number,severity,message,raw) VALUES(?,?,?,?,?)""",
                [(timestamp, i["row"], i["severity"], i["message"], json.dumps(i["raw"], default=str))
                 for i in issues])
        self.log("INFO", "reconcile" if rebuild else "import", f"{imported} new drawings")
        return imported

    def table(self, name, limit=None):
        if name not in {"data_quality", "system_logs", "predictions", "model_experiments"}:
            raise ValueError("Unsupported table")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Query limit must be a positive integer")
        with self.connect() as conn:
            query = f"SELECT * FROM {name} ORDER BY id DESC"
            return pd.read_sql_query(query + (" LIMIT ?" if limit else ""), conn,
                                     params=(limit,) if limit else ())

    def prediction_count(self):
        with self.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def latest_predictions(self):
        with self.connect() as conn:
            return pd.read_sql_query("""SELECT * FROM predictions
                WHERE batch_id=(SELECT batch_id FROM predictions ORDER BY id DESC LIMIT 1)
                ORDER BY combination_number""", conn)

    def save_experiment(self, experiment, size):
        with self.connect() as conn:
            for result in experiment["results"]:
                conn.execute("""INSERT INTO model_experiments
                    (model_name,model_version,trained_at,dataset_size,artifact,metadata) VALUES(?,?,?,?,?,?)""",
                    (result["model"], experiment.get("engine_version", "unknown"), now(), size, experiment.get("artifact"),
                     json.dumps(result, default=str)))

    def save_predictions(self, predictions, batch_id, target_date=None, backtest=None):
        if len(predictions) != 10 or len({tuple(sorted(p["numbers"])) for p in predictions}) != 10:
            raise ValueError("A saved portfolio must contain exactly ten unique combinations")
        with self.connect() as conn:
            for i, prediction in enumerate(predictions, 1):
                validated = validate_draw({"year": datetime.now().year, "week": 1, "draw_date": None,
                                           "numbers": prediction["numbers"]})
                if prediction["numbers"] != validated["numbers"]:
                    raise ValueError("Saved prediction numbers must be in ascending order")
                metadata = {**prediction, "backtesting": backtest}
                payload = json.dumps(metadata, default=str, sort_keys=True)
                existing = conn.execute("SELECT metadata FROM predictions WHERE batch_id=? AND combination_number=?",
                                        (batch_id, i)).fetchone()
                if existing:
                    if json.loads(existing["metadata"]) != json.loads(payload):
                        raise ValueError("Cannot overwrite an existing prediction batch with different contents")
                    continue
                conn.execute("""INSERT INTO predictions
                    (batch_id,prediction_date,target_draw_date,combination_number,model,seed,metadata)
                    VALUES(?,?,?,?,?,?,?)""",
                    (batch_id, now(), target_date, i, prediction["model"], prediction["seed"],
                     payload))
