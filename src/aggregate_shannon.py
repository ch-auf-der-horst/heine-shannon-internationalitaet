#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate predicted letter-level category weights to correspondent-level profiles
(bidirectional Heine <-> partner), then compute Shannon index + evenness.

Inputs:
  - data/brief_predictions_top100.csv

Outputs:
  - data/korrespondenten_profile_pred_top100.csv
"""

from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, List, Optional, Sequence, Tuple

TARGET_COLS: Tuple[str, ...] = (
    "INTELLECTUAL",
    "NETWORK",
    "POLITICAL",
    "PATRONAGE",
    "AFFECTIVE",
    "INSTITUTIONAL",
    "EDITORIAL",
    "FAMILIAL",
    "TRANSACTIONAL",
    "CONFLICT",
)

def shannon_index(probs: List[float], base: float = 2.0) -> float:
    # probs should sum to 1, but we compute robustly
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * (math.log(p) / math.log(base))
    return s

def evenness(H: float, n: int, base: float = 2.0) -> float:
    if n <= 1:
        return 0.0
    Hmax = math.log(n) / math.log(base)
    return H / Hmax if Hmax > 0 else 0.0

def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_csv",
        default=str(_DATA / "brief_predictions_top100.csv"),
    )
    ap.add_argument(
        "--out",
        default=str(_DATA / "korrespondenten_profile_pred_top100.csv"),
    )
    args = ap.parse_args(argv)

    sums: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(TARGET_COLS))
    counts: Dict[str, int] = defaultdict(int)
    counts_dir: Dict[Tuple[str, str], int] = defaultdict(int)  # (partner, richtung)

    with Path(args.pred_csv).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            partner = (row.get("partner_vollname") or "").strip()
            if not partner:
                continue
            counts[partner] += 1
            counts_dir[(partner, (row.get("richtung") or "").strip())] += 1
            for i, k in enumerate(TARGET_COLS):
                v = (row.get(k) or "").strip()
                sums[partner][i] += float(v) if v else 0.0

    out_rows: List[Dict[str, str]] = []
    for partner in sorted(counts.keys(), key=lambda x: (-counts[x], x)):
        n = counts[partner]
        mean = [s / n for s in sums[partner]]
        H = shannon_index(mean, base=2.0)
        E = evenness(H, len(TARGET_COLS), base=2.0)
        row: Dict[str, str] = {
            "partner_vollname": partner,
            "n_briefe": str(n),
            "n_von_heine": str(counts_dir.get((partner, "VON_HEINE"), 0)),
            "n_an_heine": str(counts_dir.get((partner, "AN_HEINE"), 0)),
            "shannon_H": f"{H:.6f}",
            "evenness_E": f"{E:.6f}",
        }
        for k, v in zip(TARGET_COLS, mean):
            row[k] = f"{v:.10f}".rstrip("0").rstrip(".")
        out_rows.append(row)

    fieldnames = [
        "partner_vollname",
        "n_briefe",
        "n_von_heine",
        "n_an_heine",
        "shannon_H",
        "evenness_E",
    ] + list(TARGET_COLS)
    write_csv(Path(args.out), out_rows, fieldnames)
    print(f"Wrote: {args.out} (rows={len(out_rows)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

