from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.features.engineering import validate_frame

NUMBER_COLUMNS = [f"number_{i}" for i in range(1, 6)]


def analyze(frame):
    if frame.empty:
        raise ValueError("Download historical results first")
    frame = validate_frame(frame)
    numbers = frame[NUMBER_COLUMNS].to_numpy(dtype=int)
    n = len(numbers)
    hits = np.zeros((n, 90), dtype=np.int8)
    hits[np.arange(n)[:, None], numbers - 1] = 1
    count = hits.sum(axis=0)
    pvalues = np.array([binomtest(int(c), n, 5 / 90).pvalue for c in count])
    order = np.argsort(pvalues)
    adjusted = np.empty(90)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * (90 - np.arange(90))))
    freq = pd.DataFrame({"number": range(1, 91), "count": count,
                         "draw_percentage": count / n * 100, "expected": n * 5 / 90,
                         "difference": count - n * 5 / 90, "holm_p_value": adjusted})
    gaps = []
    for i in range(90):
        positions = np.flatnonzero(hits[:, i])
        intervals = np.diff(positions) - 1
        gaps.append({"number": i + 1, "since_last": n - 1 - positions[-1] if len(positions) else n,
                     "average_gap": float(intervals.mean()) if len(intervals) else None,
                     "median_gap": float(np.median(intervals)) if len(intervals) else None,
                     "maximum_gap": int(intervals.max()) if len(intervals) else None,
                     "completed_gap_distribution": intervals.tolist()})
    pairs = Counter(pair for row in numbers for pair in combinations(row.tolist(), 2))
    triples = Counter(triple for row in numbers for triple in combinations(row.tolist(), 3))
    pair_frame = pd.DataFrame([{"pair": f"{a} + {b}", "count": pairs[(a, b)],
                                "expected": n * 20 / (90 * 89)}
                               for a, b in combinations(range(1, 91), 2)])
    triple_frame = pd.DataFrame([{"triple": " + ".join(map(str, t)), "count": c}
                                for t, c in triples.most_common()])
    distribution = pd.DataFrame({"year": frame.year, "week": frame.week,
                                 "sum": numbers.sum(axis=1), "odd": (numbers % 2).sum(axis=1),
                                 "low_1_45": (numbers <= 45).sum(axis=1),
                                 "spread": numbers[:, -1] - numbers[:, 0],
                                 "consecutive_pairs": (np.diff(numbers, axis=1) == 1).sum(axis=1)})
    decades = pd.DataFrame({"band": [f"{i+1}-{i+10}" for i in range(0, 90, 10)],
                            "count": [int(((numbers > i) & (numbers <= i + 10)).sum())
                                      for i in range(0, 90, 10)]})
    trends = pd.DataFrame(hits, columns=range(1, 91)).rolling(100, min_periods=10).mean()
    trends["draw_index"] = range(n)
    return {"frequency": freq, "gaps": pd.DataFrame(gaps), "pairs": pair_frame,
            "triples": triple_frame, "distribution": distribution,
            "decades": decades, "trends": trends}
