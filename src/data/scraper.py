import hashlib
import logging
import sqlite3
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import RAW, SOURCE
from src.data.parser import parse_html

LOG = logging.getLogger(__name__)


def sync_database(db, force=False, interval_hours=24, rebuild=False):
    state = db.state()
    fingerprint = db.data_fingerprint()
    local_unchanged = state.get("database_fingerprint") == fingerprint
    last = state.get("last_attempt")
    if not force and last and (state.get("attempt_fingerprint") == fingerprint or local_unchanged):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
        if age < interval_hours * 3600:
            return {"status": "cached", "imported": 0}
    timestamp = datetime.now(timezone.utc).isoformat()
    db.set_state({"last_attempt": timestamp, "attempt_fingerprint": fingerprint})
    headers = {"User-Agent": "OtoslottoResearch/1.0 (local historical analytics)"}
    if not rebuild and local_unchanged:
        for key, header in (("etag", "If-None-Match"), ("last_modified", "If-Modified-Since")):
            if state.get(key):
                headers[header] = state[key]
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=Retry(
                total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
            response = session.get(SOURCE, headers=headers, timeout=(10, 60))
            response.raise_for_status()
        if response.status_code == 304:
            if not local_unchanged:
                raise ValueError("Unexpected HTTP 304 while local data need full reconciliation")
            db.set_state({"last_success": timestamp, "last_error": ""})
            return {"status": "unchanged", "imported": 0}
        RAW.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(response.content).hexdigest()
        archive = RAW / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{digest[:12]}.html"
        archive.write_bytes(response.content)
        records, issues = parse_html(response.content.decode("utf-8-sig"))
        # A suddenly truncated source must not masquerade as a complete synchronization.
        local = db.draws()
        saved_keys = set(zip(local.year, local.week))
        source_keys = {(r["year"], r["week"]) for r in records}
        absent = saved_keys - source_keys
        if absent:
            raise ValueError(f"Official response omits {len(absent)} saved drawings "
                             f"(e.g. {sorted(absent)[:5]}); import aborted")
        imported = db.import_records(records, issues, rebuild=rebuild)
        synchronized_fingerprint = db.data_fingerprint()
        db.set_state({"last_success": timestamp, "last_error": "", "sha256": digest,
                      "etag": response.headers.get("ETag", ""),
                      "last_modified": response.headers.get("Last-Modified", ""),
                      "database_fingerprint": synchronized_fingerprint,
                      "attempt_fingerprint": synchronized_fingerprint,
                      "raw_archive": str(archive)})
        return {"status": "synchronized", "imported": imported,
                "records": len(records), "issues": len(issues)}
    except (requests.RequestException, ValueError, UnicodeError, OSError, sqlite3.Error) as exc:
        LOG.exception("Official synchronization failed")
        db.log("ERROR", "synchronization", str(exc))
        db.set_state({"last_error": str(exc)})
        raise
