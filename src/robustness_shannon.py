#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness analysis for correspondent-level profiles and Shannon/Evenness.

Computes multiple aggregation variants from letter-level predictions:
  A) unweighted mean over letters (status quo)
  B) length-weighted mean over letters (weights = brieftext length)
  C) unweighted mean after trimming the longest 1% letters globally
  (optional) D) length-weighted + trimmed (also computed)

Inputs:
  - data/brief_predictions_top100.csv
  - Master CSV (for brieftext length per letterid)

Outputs:
  - data/robustness_shannon_comparison.csv
  - data/robustness_shannon_report.md
"""

from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

def rankdata(values: Dict[str, float]) -> Dict[str, float]:
    """
    Compute average ranks (1..n) for dict of key->value.
    Ties receive the average rank.
    """
    items = sorted(values.items(), key=lambda kv: kv[1])
    ranks: Dict[str, float] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i
        v = items[i][1]
        while j < n and items[j][1] == v:
            j += 1
        # ranks are 1-indexed
        avg_rank = (i + 1 + j) / 2.0
        for k, _ in items[i:j]:
            ranks[k] = avg_rank
        i = j
    return ranks

def spearman_rho(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a).intersection(b))
    if len(keys) < 2:
        return float("nan")
    ra = rankdata({k: a[k] for k in keys})
    rb = rankdata({k: b[k] for k in keys})
    ma = sum(ra[k] for k in keys) / len(keys)
    mb = sum(rb[k] for k in keys) / len(keys)
    num = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    den_a = math.sqrt(sum((ra[k] - ma) ** 2 for k in keys))
    den_b = math.sqrt(sum((rb[k] - mb) ** 2 for k in keys))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)

def load_lengths_from_master(master_csv: Path) -> Dict[str, int]:
    csv.field_size_limit(sys.maxsize)
    out: Dict[str, int] = {}
    with master_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            lid = (row.get("letterid") or "").strip()
            if not lid:
                continue
            out[lid] = len(row.get("brieftext") or "")
    return out

class PredRow(Tuple[str, str, List[float]]):
    pass

def iter_predictions(pred_csv: Path) -> Iterable[Tuple[str, str, List[float]]]:
    """
    Yield (letterid, partner_vollname, vec[10]) for each predicted letter.
    """
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            lid = (row.get("letterid") or "").strip()
            partner = (row.get("partner_vollname") or "").strip()
            if not lid or not partner:
                continue
            vec: List[float] = []
            ok = True
            for k in TARGET_COLS:
                v = (row.get(k) or "").strip()
                if v == "":
                    ok = False
                    break
                vec.append(float(v))
            if not ok:
                continue
            yield lid, partner, vec

def aggregate_profiles(
    rows: List[Tuple[str, str, List[float]]],
    lengths: Dict[str, int],
    *,
    weighted: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    sums: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(TARGET_COLS))
    denom: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)

    for lid, partner, vec in rows:
        w = float(lengths.get(lid, 0) if weighted else 1.0)
        if weighted and w <= 0:
            continue
        counts[partner] += 1
        denom[partner] += w
        for i, x in enumerate(vec):
            sums[partner][i] += w * x

    profiles: Dict[str, List[float]] = {}
    for partner in counts:
        d = denom[partner]
        if d <= 0:
            continue
        mean = [s / d for s in sums[partner]]
        # normalize numerical drift
        ssum = sum(mean)
        if ssum > 0:
            mean = [x / ssum for x in mean]
        profiles[partner] = mean
    return profiles, counts

def compute_shannon(profiles: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: shannon_index(v, base=2.0) for k, v in profiles.items()}

def compute_evenness(profiles: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in profiles.items():
        H = shannon_index(v, base=2.0)
        out[k] = evenness(H, len(TARGET_COLS), base=2.0)
    return out

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
        "--master_csv",
        default=str(_LOCAL / "alle_baende_MASTER.csv"),
    )
    ap.add_argument(
        "--trim_quantile",
        type=float,
        default=0.01,
        help="Global trim proportion for longest letters (e.g. 0.01 = top 1%%).",
    )
    ap.add_argument(
        "--out_csv",
        default=str(_DATA / "robustness_shannon_comparison.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(_DATA / "robustness_shannon_report.md"),
    )
    args = ap.parse_args(argv)

    pred_csv = Path(args.pred_csv)
    master_csv = Path(args.master_csv)

    lengths = load_lengths_from_master(master_csv)

    rows_all = list(iter_predictions(pred_csv))
    if not rows_all:
        raise SystemExit("No rows in predictions CSV.")

    # Determine trim threshold globally
    trim_p = float(args.trim_quantile)
    n_trim = int(math.ceil(trim_p * len(rows_all))) if trim_p > 0 else 0
    rows_sorted = sorted(rows_all, key=lambda t: lengths.get(t[0], 0), reverse=True)
    trimmed_out = set([lid for lid, _, _ in rows_sorted[:n_trim]])
    rows_trim = [t for t in rows_all if t[0] not in trimmed_out]

    prof_A, cnt_A = aggregate_profiles(rows_all, lengths, weighted=False)
    prof_B, cnt_B = aggregate_profiles(rows_all, lengths, weighted=True)
    prof_C, cnt_C = aggregate_profiles(rows_trim, lengths, weighted=False)
    prof_D, cnt_D = aggregate_profiles(rows_trim, lengths, weighted=True)

    H_A = compute_shannon(prof_A)
    H_B = compute_shannon(prof_B)
    H_C = compute_shannon(prof_C)
    H_D = compute_shannon(prof_D)

    E_A = compute_evenness(prof_A)
    E_B = compute_evenness(prof_B)
    E_C = compute_evenness(prof_C)
    E_D = compute_evenness(prof_D)

    keys = sorted(set(H_A).intersection(H_B).intersection(H_C).intersection(H_D))

    out_rows: List[Dict[str, str]] = []
    for k in keys:
        out_rows.append(
            {
                "partner_vollname": k,
                "n_briefe": str(cnt_A.get(k, 0)),
                "H_unweighted": f"{H_A[k]:.6f}",
                "H_len_weighted": f"{H_B[k]:.6f}",
                "H_trim_unweighted": f"{H_C[k]:.6f}",
                "H_trim_len_weighted": f"{H_D[k]:.6f}",
                "E_unweighted": f"{E_A[k]:.6f}",
                "E_len_weighted": f"{E_B[k]:.6f}",
                "E_trim_unweighted": f"{E_C[k]:.6f}",
                "E_trim_len_weighted": f"{E_D[k]:.6f}",
            }
        )

    fieldnames = [
        "partner_vollname",
        "n_briefe",
        "H_unweighted",
        "H_len_weighted",
        "H_trim_unweighted",
        "H_trim_len_weighted",
        "E_unweighted",
        "E_len_weighted",
        "E_trim_unweighted",
        "E_trim_len_weighted",
    ]
    write_csv(Path(args.out_csv), out_rows, fieldnames)

    # Rank correlations (Shannon + Evenness)
    md_lines: List[str] = []
    md_lines.append("## Robustheitsanalyse: Shannon/Evenness (TOP100)\n\n")
    md_lines.append(f"- **Input predictions**: `{pred_csv}`\n")
    md_lines.append(f"- **Input master (lengths)**: `{master_csv}`\n")
    md_lines.append(f"- **Letters total**: {len(rows_all)}\n")
    md_lines.append(f"- **Trim proportion**: {trim_p:.3%} (removed {n_trim} letters)\n")
    md_lines.append(f"- **Correspondents compared**: {len(keys)}\n\n")

    def fmt(x: float) -> str:
        return "nan" if (x != x) else f"{x:.4f}"

    md_lines.append("### Spearman-Rangkorrelationen (H)\n\n")
    md_lines.append(f"- H(unweighted) vs H(len_weighted): **{fmt(spearman_rho(H_A, H_B))}**\n")
    md_lines.append(f"- H(unweighted) vs H(trim_unweighted): **{fmt(spearman_rho(H_A, H_C))}**\n")
    md_lines.append(f"- H(unweighted) vs H(trim_len_weighted): **{fmt(spearman_rho(H_A, H_D))}**\n")
    md_lines.append("\n### Spearman-Rangkorrelationen (Evenness)\n\n")
    md_lines.append(f"- E(unweighted) vs E(len_weighted): **{fmt(spearman_rho(E_A, E_B))}**\n")
    md_lines.append(f"- E(unweighted) vs E(trim_unweighted): **{fmt(spearman_rho(E_A, E_C))}**\n")
    md_lines.append(f"- E(unweighted) vs E(trim_len_weighted): **{fmt(spearman_rho(E_A, E_D))}**\n")

    Path(args.out_md).write_text("".join(md_lines), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

