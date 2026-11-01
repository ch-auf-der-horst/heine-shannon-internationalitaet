#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference for all Heine-involved letters, restricted to TOP-100 partners.

Inputs:
  - local/alle_baende_MASTER.csv (edition master file, not published)
  - local/top_100_korrespondenten.csv
  - local/ml_baseline_tfidf_model.joblib

Outputs:
  - data/brief_predictions_top100.csv
"""

from __future__ import annotations
import argparse
import csv
import sys
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from joblib import load

from text_normalize import normalize_unicode_ws

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

def project_to_simplex(yhat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y = np.maximum(yhat, 0.0)
    sums = y.sum(axis=1, keepdims=True)
    zero_mask = sums.squeeze() <= eps
    if np.any(zero_mask):
        y[zero_mask, :] = 1.0 / y.shape[1]
        sums = y.sum(axis=1, keepdims=True)
    return y / sums

def load_top100_names(path: Path) -> Set[str]:
    names: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            n = normalize_unicode_ws((row.get("Vollname") or "").strip())
            if n:
                names.add(n)
    return names

def iter_master_heine_letters(master_csv: Path) -> Iterable[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with master_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            a = normalize_unicode_ws((row.get("absender_vollname") or "").strip())
            b = normalize_unicode_ws((row.get("adressat_vollname") or "").strip())
            # Important: identify Heinrich Heine unambiguously.
            if a == "Heinrich Heine" or b == "Heinrich Heine":
                yield row

def partner_for_row(row: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (partner_vollname, direction)
    direction: VON_HEINE or AN_HEINE
    """
    abs_vollname = normalize_unicode_ws((row.get("absender_vollname") or "").strip())
    if abs_vollname == "Heinrich Heine":
        return normalize_unicode_ws(row.get("adressat_vollname") or ""), "VON_HEINE"
    return normalize_unicode_ws(row.get("absender_vollname") or ""), "AN_HEINE"

def is_placeholder_partner(name: str) -> bool:
    """Exclude synthetic placeholder entities from analysis sample."""
    n = normalize_unicode_ws(name).lower()
    return n in {
        "unbekannter absender",
        "unbekannter adressat",
        "testament (ohne empfänger)",
    }

def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master",
        default=str(_LOCAL / "alle_baende_MASTER.csv"),
    )
    ap.add_argument(
        "--top100",
        default=str(_LOCAL / "top_100_korrespondenten.csv"),
    )
    ap.add_argument(
        "--model",
        default=str(_LOCAL / "ml_baseline_tfidf_model.joblib"),
    )
    ap.add_argument(
        "--out",
        default=str(_DATA / "brief_predictions_top100.csv"),
    )
    args = ap.parse_args(argv)

    top100 = load_top100_names(Path(args.top100))
    model = load(args.model)

    rows_out: List[Dict[str, str]] = []
    texts: List[str] = []
    meta: List[Dict[str, str]] = []

    for row in iter_master_heine_letters(Path(args.master)):
        partner, direction = partner_for_row(row)
        if is_placeholder_partner(partner):
            continue
        if partner not in top100:
            continue
        txt = row.get("brieftext") or ""
        texts.append(txt)
        meta.append(
            {
                "letterid": row.get("letterid") or "",
                "partner_vollname": partner,
                "richtung": direction,
                "sprache": row.get("sprache") or "",
                "datum_iso": row.get("datum_iso") or "",
                "datum_gephi": row.get("datum_gephi") or "",
                "band": row.get("band") or "",
            }
        )

    if not texts:
        print("No TOP-100 letters found (check name matching).", file=sys.stderr)
        return 2

    yhat = model.predict(texts)
    yhat = project_to_simplex(np.asarray(yhat, dtype=float))

    for m, vec in zip(meta, yhat):
        out = dict(m)
        for k, v in zip(TARGET_COLS, vec):
            out[k] = f"{float(v):.10f}".rstrip("0").rstrip(".")
        rows_out.append(out)

    fieldnames = list(meta[0].keys()) + list(TARGET_COLS)
    write_csv(Path(args.out), rows_out, fieldnames)
    print(f"Wrote: {args.out} (rows={len(rows_out)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

