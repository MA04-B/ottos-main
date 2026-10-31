import logging
import re
import unicodedata
from datetime import datetime
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from src.data.validator import validate_draw

LOG = logging.getLogger(__name__)


class ParseError(ValueError):
    pass


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower())
                   if not unicodedata.combining(c)).replace(" ", "")


def hungarian_integer(value):
    cleaned = re.sub(r"\s|Ft", "", str(value), flags=re.IGNORECASE)
    if cleaned in ("", "-", "nan"):
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    if not re.fullmatch(r"\d+", cleaned):
        raise ValueError(f"Invalid Hungarian integer: {value}")
    return int(cleaned)


def hungarian_date(value):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for pattern in ("%Y.%m.%d.", "%Y.%m.%d", "%Y-%m-%d", "%Y. %m. %d."):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized drawing date: {text}")


def parse_html(html: str) -> tuple[list[dict], list[dict]]:
    candidates = []
    for table in BeautifulSoup(html, "html.parser").find_all("table"):
        headers = [normalize(cell.get_text()) for cell in table.find_all("th")]
        if "ev" in headers and "het" in headers and any("szam" in h for h in headers):
            candidates.append(table)
    if len(candidates) != 1:
        raise ParseError(f"Expected one recognizable lottery table, found {len(candidates)}")
    # pandas expands colspan headers, so the five number columns are located by name.
    frame = pd.read_html(StringIO(str(candidates[0])), header=0, keep_default_na=False)[0]
    columns = [normalize(c) for c in frame.columns]
    year_index, week_index = columns.index("ev"), columns.index("het")
    date_columns = [i for i, c in enumerate(columns) if "datum" in c]
    number_columns = [i for i, c in enumerate(columns) if "szam" in c]
    if len(number_columns) != 5 or len(date_columns) != 1:
        raise ParseError("Table schema changed: need a date column and five number columns")
    records, issues, seen_dates, seen_weeks = [], [], {}, {}
    for row_index, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        raw = list(values)
        try:
            prizes = {}
            for i, value in enumerate(raw):
                if i in [year_index, week_index, *date_columns, *number_columns]:
                    continue
                if "talalat" not in columns[i]:
                    continue
                try:
                    prizes[str(frame.columns[i])] = hungarian_integer(value)
                except ValueError as exc:
                    LOG.warning("Malformed prize metadata in row %s: %s", row_index, exc)
                    prizes[str(frame.columns[i])] = None
                    issues.append({"row": row_index, "severity": "warning",
                                   "message": f"Malformed prize metadata: {exc}", "raw": raw})
            record = validate_draw({
                "year": hungarian_integer(raw[year_index]),
                "week": hungarian_integer(raw[week_index]),
                "draw_date": hungarian_date(raw[date_columns[0]]),
                "numbers": [hungarian_integer(raw[i]) for i in number_columns],
                "prizes": prizes,
            })
            key = (record["year"], record["week"])
            existing = seen_weeks.get(key) or seen_dates.get(record["draw_date"])
            if existing:
                if (existing["numbers"] != record["numbers"]
                        or (existing["year"], existing["week"]) != key
                        or (existing["draw_date"] and record["draw_date"]
                            and existing["draw_date"] != record["draw_date"])):
                    raise ParseError(f"Conflicting duplicate drawing {key}; import aborted")
                if not existing["draw_date"] and record["draw_date"]:
                    existing["draw_date"] = record["draw_date"]
                    seen_dates[record["draw_date"]] = existing
                issues.append({"row": row_index, "severity": "warning", "message": "Duplicate drawing skipped", "raw": raw})
                continue
            seen_weeks[key] = record
            if record["draw_date"]:
                seen_dates[record["draw_date"]] = record
            else:
                issues.append({"row": row_index, "severity": "warning",
                               "message": "Official source omits date; retained by year/week, no date invented", "raw": raw})
            records.append(record)
        except ParseError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            LOG.warning("Rejected source row %s: %s", row_index, exc)
            issues.append({"row": row_index, "severity": "error", "message": str(exc), "raw": raw})
    if not records or sum(i["severity"] == "error" for i in issues) > max(5, len(frame) * .05):
        raise ParseError("No usable records or excessive malformed rows; database unchanged")
    return sorted(records, key=lambda r: (r["year"], r["week"])), issues
