"""Independent source/SQLite reconciliation without the production parser."""

import hashlib
import json
import re
from collections import Counter
from datetime import date

from bs4 import BeautifulSoup


def audit_html(html: str, db) -> dict:
    tables = []
    for table in BeautifulSoup(html, "html.parser").find_all("table"):
        headers = [c.get_text(" ", strip=True).casefold() for c in table.find_all("th")]
        if "év" in headers and "hét" in headers and any("szám" in c for c in headers):
            tables.append(table)
    if len(tables) != 1:
        raise ValueError("Independent audit needs one recognizable official table")
    source, invalid, duplicate_keys, duplicate_dates, prize_failures = {}, [], [], [], []
    dates = {}
    rows = tables[0].find_all("tr")
    # Audit the actual official 16-column contract independently of pandas.
    for row_number, row in enumerate(rows, 1):
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        if not cells:
            continue
        try:
            if len(cells) != 16:
                raise ValueError(f"Expected 16 official cells, got {len(cells)}")
            year, week = int(cells[0]), int(cells[1])
            numbers = sorted(int(v) for v in cells[11:16])
            if len(set(numbers)) != 5 or not all(1 <= n <= 90 for n in numbers):
                raise ValueError("Invalid official winning numbers")
            drawn = date.fromisoformat(cells[2].strip(".").replace(".", "-")).isoformat() if cells[2] else None
            key = (year, week)
            if key in source:
                duplicate_keys.append(list(key))
            if drawn and drawn in dates:
                duplicate_dates.append(drawn)
            if drawn:
                dates[drawn] = key
            prizes = {}
            for index, matches in enumerate((5, 4, 3, 2)):
                for offset, unit in ((0, "db"), (1, "Ft")):
                    value = cells[3 + index * 2 + offset]
                    cleaned = re.sub(r"\s|Ft", "", value).replace(".", "")
                    if cleaned.isdecimal():
                        prizes[f"{matches} találat ({unit})"] = int(cleaned)
                    elif not cleaned or cleaned == "-":
                        prizes[f"{matches} találat ({unit})"] = None
                    else:
                        prize_failures.append({"key": list(key), "value": value})
            source[key] = {"draw_date": drawn, "numbers": numbers, "prizes": prizes}
        except (ValueError, IndexError) as exc:
            invalid.append({"row": row_number, "message": str(exc), "raw": cells})
    local_frame = db.draws()
    local = {(int(r.year), int(r.week)): r for r in local_frame.itertuples()}
    mismatch = []
    for key in source.keys() & local.keys():
        expected, actual = source[key], local[key]
        for field, observed in (
            ("numbers", [int(getattr(actual, f"number_{i}")) for i in range(1, 6)]),
            ("draw_date", actual.draw_date),
            ("prizes", json.loads(actual.prizes)),
        ):
            if expected[field] != observed:
                mismatch.append({"key": list(key), "field": field, "expected": expected[field], "actual": observed})
    gaps = []
    incomplete_years = []
    years = sorted({y for y, _ in source})
    for year in years:
        weeks = sorted(w for y, w in source if y == year)
        start = min(weeks) if year == years[0] else 1
        missing = sorted(set(range(start, max(weeks) + 1)) - set(weeks))
        if missing:
            gaps.append({"year": year, "weeks": missing})
        if year not in (years[0], years[-1]) and (min(weeks) != 1 or max(weeks) not in (52, 53)):
            incomplete_years.append({"year": year, "first_week": min(weeks), "last_week": max(weeks)})
    local_numbers = local_frame[[f"number_{i}" for i in range(1, 6)]].to_numpy()
    invalid_local = [i for i, ns in enumerate(local_numbers)
                     if len(set(ns)) != 5 or list(ns) != sorted(ns) or not all(1 <= n <= 90 for n in ns)]
    result = {
        "source_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "source_rows": sum(bool(row.find_all("td")) for row in rows),
        "source_unique_draws": len(source),
        "database_draws": len(local),
        "source_invalid_rows": invalid,
        "source_duplicate_year_weeks": duplicate_keys,
        "source_duplicate_dates": duplicate_dates,
        "source_prize_failures": prize_failures,
        "missing_from_database": [list(k) for k in sorted(source.keys() - local.keys())],
        "database_only_draws": [list(k) for k in sorted(local.keys() - source.keys())],
        "field_mismatches": mismatch,
        "internal_year_week_gaps": gaps,
        "incomplete_source_years": incomplete_years,
        "invalid_database_number_rows": invalid_local,
        "database_duplicate_year_weeks": int(local_frame.duplicated(["year", "week"]).sum()),
        "database_duplicate_dates": int(local_frame.draw_date.dropna().duplicated().sum()),
        "official_missing_dates": sum(r["draw_date"] is None for r in source.values()),
        "draws_per_year": dict(sorted(Counter(y for y, _ in source).items())),
        "first_draw": list(min(source)) if source else None,
        "last_draw": list(max(source)) if source else None,
    }
    failure_fields = ("source_invalid_rows", "source_duplicate_year_weeks", "source_duplicate_dates",
                      "source_prize_failures", "missing_from_database", "database_only_draws", "field_mismatches",
                      "internal_year_week_gaps", "invalid_database_number_rows",
                      "incomplete_source_years",
                      "database_duplicate_year_weeks", "database_duplicate_dates")
    result["passed"] = bool(source) and not any(result[key] for key in failure_fields)
    return result
