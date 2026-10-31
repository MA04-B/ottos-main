from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.database import Database
from src.data.parser import ParseError, hungarian_date, hungarian_integer, parse_html
from src.data.scraper import sync_database
from src.data.validator import validate_draw

HEADER = """<table><tr><th>Év</th><th>Hét</th><th>Húzásdátum</th>
<th>5 találat (db)</th><th>5 találat (Ft)</th><th colspan="5">Számok</th></tr>"""


def html(rows):
    return HEADER + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</table>"


def record(**kwargs):
    return {"year": 2024, "week": 10, "draw_date": "2024-03-09",
            "numbers": [5, 1, 90, 20, 7], "prizes": {}, **kwargs}


def test_hungarian_formats():
    assert hungarian_date("2024.03.09.") == "2024-03-09"
    assert hungarian_date("") is None
    assert hungarian_integer("5 197\u00a0132 880 Ft") == 5197132880
    assert hungarian_integer("1.234.567 Ft") == 1234567
    with pytest.raises(ValueError):
        hungarian_date("not a date")
    with pytest.raises(ValueError):
        hungarian_integer("broken")


def test_dynamic_parser_and_missing_date():
    rows = [[2024, 10, "2024.03.09.", 0, "0 Ft", 90, 7, 20, 1, 5],
            [1957, 10, "", 0, "0 Ft", 16, 61, 71, 77, 89]]
    records, issues = parse_html("<table><tr><td>noise</td></tr></table>" + html(rows))
    assert len(records) == 2
    assert records[0]["draw_date"] is None
    assert records[1]["numbers"] == [1, 5, 7, 20, 90]
    assert len(issues) == 1


def test_reject_malformed_and_duplicate():
    row = [2024, 10, "2024.03.09.", 0, "0 Ft", 1, 5, 7, 20, 90]
    records, issues = parse_html(html([row, row, [2024, 11, "2024.03.16.", 0, "0 Ft", 1, 1, 7, 20, 90]]))
    assert len(records) == 1
    assert {i["severity"] for i in issues} == {"warning", "error"}
    with pytest.raises(ParseError):
        parse_html("<table><tr><th>Wrong structure</th></tr></table>")
    other = [*row[:-1], 89]
    with pytest.raises(ParseError, match="Conflicting"):
        parse_html(html([row, other]))


def test_parser_recognizes_reordered_headers():
    source = """<table><tr><th>Hét</th><th>Év</th>
    <th>Szám 1</th><th>Szám 2</th><th>Szám 3</th><th>Szám 4</th><th>Szám 5</th>
    <th>Húzásdátum</th></tr>
    <tr><td>10</td><td>2024</td><td>1</td><td>5</td><td>7</td><td>20</td><td>90</td>
    <td>2024.03.09.</td></tr></table>"""
    records, issues = parse_html(source)
    assert records[0]["numbers"] == [1, 5, 7, 20, 90]
    assert not issues


def test_validator_and_year_rollover():
    assert validate_draw(record())["numbers"] == [1, 5, 7, 20, 90]
    assert validate_draw(record(year=2025, week=52, draw_date="2026-01-01"))
    for numbers in ([1, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2, 3, 4, 91], [1, 2, 3, 4]):
        with pytest.raises(ValueError):
            validate_draw(record(numbers=numbers))


def test_database_atomic_updates_and_duplicate(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.import_records([record()], []) == 1
    assert db.import_records([record()], []) == 0
    assert len(db.draws()) == 1
    with pytest.raises(ValueError, match="conflicts"):
        db.import_records([record(week=11, draw_date="2024-03-16"),
                           record(numbers=[2, 3, 4, 5, 6])], [])
    assert len(db.draws()) == 1
    db.import_records([record(year=1957, week=10, draw_date=None),
                       record(year=1957, week=11, draw_date=None)], [])
    assert len(db.draws()) == 3


def test_mock_download_archive_and_conditionals(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.scraper.RAW", tmp_path / "raw")
    db = Database(tmp_path / "test.db")
    response = MagicMock()
    response.status_code = 200
    response.content = html([[2024, 10, "2024.03.09.", 0, "0 Ft", 1, 5, 7, 20, 90]]).encode()
    response.headers = {"ETag": "abc"}
    with patch("requests.Session.get", return_value=response) as get:
        result = sync_database(db, force=True)
        assert result["imported"] == 1
        assert get.call_args.kwargs["timeout"] == (10, 60)
        assert len(list((tmp_path / "raw").glob("*.html"))) == 1
        assert sync_database(db)["status"] == "cached"
        response.status_code = 304
        assert sync_database(db, force=True)["status"] == "unchanged"
        assert get.call_args.kwargs["headers"]["If-None-Match"] == "abc"


def test_failed_sync_preserves_saved_data(tmp_path):
    db = Database(tmp_path / "test.db")
    db.import_records([record()], [])
    with patch("requests.Session.get", side_effect=requests.Timeout("Offline")):
        with pytest.raises(requests.Timeout):
            sync_database(db, force=True)
    assert len(db.draws()) == 1
    assert "Offline" in db.state()["last_error"]


def test_statistics(tmp_path):
    from src.analysis.statistics import analyze
    db = Database(tmp_path / "test.db")
    db.import_records([record()], [])
    result = analyze(db.draws())
    assert result["frequency"]["count"].sum() == 5
    assert len(result["pairs"]) == 4005
    assert result["pairs"]["count"].sum() == 10
    assert result["triples"]["count"].sum() == 10
    assert len(result["gaps"]) == 90


def test_prediction_batch_persistence_is_idempotent(tmp_path):
    db = Database(tmp_path / "test.db")
    predictions = [{"numbers": [1, 5, 7, 20, 80 + i], "model": "uniform", "seed": 42,
                    "strategy": "weighted", "scores": [5 / 90] * 5} for i in range(10)]
    db.save_predictions(predictions, "batch-a")
    db.save_predictions(predictions, "batch-a")
    saved = db.table("predictions")
    assert len(saved) == 10
    assert saved.iloc[0].target_draw_date is None
    assert "weighted" in saved.iloc[0].metadata


def test_quality_and_explicit_source_conflict(tmp_path):
    db = Database(tmp_path / "test.db")
    db.import_records([record()], [{"row": 2, "severity": "warning", "message": "Missing date", "raw": []}])
    assert len(db.table("data_quality")) == 1
    with pytest.raises(ValueError):
        db.import_records([record(week=11)], [])
    assert len(db.draws()) == 1
    assert len(db.table("data_quality")) == 1
