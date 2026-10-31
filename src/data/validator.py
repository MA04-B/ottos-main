from datetime import date
from numbers import Integral


def validate_draw(record: dict) -> dict:
    numbers = record["numbers"]
    if len(numbers) != 5 or any(not isinstance(n, Integral) or isinstance(n, bool) for n in numbers):
        raise ValueError("Exactly five integer winning numbers are required")
    if len(set(numbers)) != 5 or any(n < 1 or n > 90 for n in numbers):
        raise ValueError("Winning numbers must be distinct and between 1 and 90")
    if not isinstance(record["year"], Integral) or isinstance(record["year"], bool):
        raise ValueError("Drawing year must be an integer")
    if not isinstance(record["week"], Integral) or isinstance(record["week"], bool):
        raise ValueError("Drawing week must be an integer")
    if not 1957 <= record["year"] <= date.today().year:
        raise ValueError("Invalid drawing year")
    if not 1 <= record["week"] <= 53:
        raise ValueError("Invalid drawing week")
    if record["draw_date"]:
        parsed = date.fromisoformat(record["draw_date"])
        rollover = (parsed.year == record["year"] + 1 and parsed.month == 1
                    and parsed.day <= 7 and record["week"] >= 52)
        if (parsed.year != record["year"] and not rollover) or parsed > date.today():
            raise ValueError("Drawing date does not match year or is in the future")
    return {**record, "numbers": sorted(int(n) for n in numbers)}
