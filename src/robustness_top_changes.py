#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process robustness_shannon_comparison.csv to identify largest changes
between aggregation variants (delta + rank shifts).

Outputs:
  - data/robustness_shannon_top_changes.csv
  - data/robustness_shannon_top_changes.md
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, List, Optional, Sequence, Tuple

def rank_map(items: Dict[str, float], *, reverse: bool) -> Dict[str, int]:
    """
    Dense rank with tie-handling by value equality:
    1 = highest (if reverse=True) / lowest (if reverse=False).
    """
    ordered = sorted(items.items(), key=lambda kv: kv[1], reverse=reverse)
    ranks: Dict[str, int] = {}
    rank = 0
    last_v = None
    for i, (k, v) in enumerate(ordered):
        if last_v is None or v != last_v:
            rank = i + 1
            last_v = v
        ranks[k] = rank
    return ranks

def ffloat(s: str) -> float:
    return float((s or "").strip())

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default=str(_DATA / "robustness_shannon_comparison.csv"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(_DATA / "robustness_shannon_top_changes.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(_DATA / "robustness_shannon_top_changes.md"),
    )
    ap.add_argument("--top_n", type=int, default=15)
    args = ap.parse_args(argv)

    rows: List[Dict[str, str]] = []
    with Path(args.in_csv).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)

    def build_metric_map(col: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for row in rows:
            k = row["partner_vollname"]
            out[k] = ffloat(row[col])
        return out

    H_u = build_metric_map("H_unweighted")
    H_w = build_metric_map("H_len_weighted")
    H_tu = build_metric_map("H_trim_unweighted")
    H_tw = build_metric_map("H_trim_len_weighted")

    E_u = build_metric_map("E_unweighted")
    E_w = build_metric_map("E_len_weighted")
    E_tu = build_metric_map("E_trim_unweighted")
    E_tw = build_metric_map("E_trim_len_weighted")

    # ranks: higher H/E = higher rank (1 is best/highest)
    rHu, rHw, rHtu, rHtw = (rank_map(H_u, reverse=True), rank_map(H_w, reverse=True), rank_map(H_tu, reverse=True), rank_map(H_tw, reverse=True))
    rEu, rEw, rEtu, rEtw = (rank_map(E_u, reverse=True), rank_map(E_w, reverse=True), rank_map(E_tu, reverse=True), rank_map(E_tw, reverse=True))

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        k = row["partner_vollname"]
        n = row.get("n_briefe", "")
        dH_w = H_w[k] - H_u[k]
        dH_tu = H_tu[k] - H_u[k]
        dH_tw = H_tw[k] - H_u[k]
        dE_w = E_w[k] - E_u[k]
        dE_tu = E_tu[k] - E_u[k]
        dE_tw = E_tw[k] - E_u[k]

        out_rows.append(
            {
                "partner_vollname": k,
                "n_briefe": n,
                "H_unweighted": f"{H_u[k]:.6f}",
                "H_len_weighted": f"{H_w[k]:.6f}",
                "delta_H_len_weighted": f"{dH_w:+.6f}",
                "rank_H_unweighted": str(rHu[k]),
                "rank_H_len_weighted": str(rHw[k]),
                "delta_rank_H_len_weighted": str(rHw[k] - rHu[k]),
                "H_trim_unweighted": f"{H_tu[k]:.6f}",
                "delta_H_trim_unweighted": f"{dH_tu:+.6f}",
                "rank_H_trim_unweighted": str(rHtu[k]),
                "delta_rank_H_trim_unweighted": str(rHtu[k] - rHu[k]),
                "H_trim_len_weighted": f"{H_tw[k]:.6f}",
                "delta_H_trim_len_weighted": f"{dH_tw:+.6f}",
                "rank_H_trim_len_weighted": str(rHtw[k]),
                "delta_rank_H_trim_len_weighted": str(rHtw[k] - rHu[k]),
                "E_unweighted": f"{E_u[k]:.6f}",
                "E_len_weighted": f"{E_w[k]:.6f}",
                "delta_E_len_weighted": f"{dE_w:+.6f}",
                "rank_E_unweighted": str(rEu[k]),
                "rank_E_len_weighted": str(rEw[k]),
                "delta_rank_E_len_weighted": str(rEw[k] - rEu[k]),
                "E_trim_unweighted": f"{E_tu[k]:.6f}",
                "delta_E_trim_unweighted": f"{dE_tu:+.6f}",
                "rank_E_trim_unweighted": str(rEtu[k]),
                "delta_rank_E_trim_unweighted": str(rEtu[k] - rEu[k]),
                "E_trim_len_weighted": f"{E_tw[k]:.6f}",
                "delta_E_trim_len_weighted": f"{dE_tw:+.6f}",
                "rank_E_trim_len_weighted": str(rEtw[k]),
                "delta_rank_E_trim_len_weighted": str(rEtw[k] - rEu[k]),
            }
        )

    # write full table
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # top lists (by absolute delta)
    top_n = int(args.top_n)

    def top_abs(col: str) -> List[Dict[str, str]]:
        return sorted(out_rows, key=lambda r: abs(ffloat(r[col])), reverse=True)[:top_n]

    def top_rankshift(col: str) -> List[Dict[str, str]]:
        return sorted(out_rows, key=lambda r: abs(int((r[col] or "0").strip())), reverse=True)[:top_n]

    md: List[str] = []
    md.append("## Robustheit: größte Änderungen (TOP100)\n\n")
    md.append(f"- Input: `{args.in_csv}`\n")
    md.append(f"- Output CSV: `{args.out_csv}`\n\n")

    def md_table(title: str, subset: List[Dict[str, str]], cols: List[str]) -> None:
        md.append(f"### {title}\n\n")
        md.append("| " + " | ".join(cols) + " |\n")
        md.append("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in subset:
            md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        md.append("\n")

    md_table(
        f"Top {top_n} nach |ΔH| (längen-gewichtet vs ungewichtet)",
        top_abs("delta_H_len_weighted"),
        ["partner_vollname", "n_briefe", "H_unweighted", "H_len_weighted", "delta_H_len_weighted", "delta_rank_H_len_weighted"],
    )
    md_table(
        f"Top {top_n} nach |ΔH| (1%-Trim ungewichtet vs ungewichtet)",
        top_abs("delta_H_trim_unweighted"),
        ["partner_vollname", "n_briefe", "H_unweighted", "H_trim_unweighted", "delta_H_trim_unweighted", "delta_rank_H_trim_unweighted"],
    )
    md_table(
        f"Top {top_n} nach |Rangshift| H (längen-gewichtet vs ungewichtet)",
        top_rankshift("delta_rank_H_len_weighted"),
        ["partner_vollname", "n_briefe", "rank_H_unweighted", "rank_H_len_weighted", "delta_rank_H_len_weighted", "delta_H_len_weighted"],
    )

    md_table(
        f"Top {top_n} nach |ΔE| (längen-gewichtet vs ungewichtet)",
        top_abs("delta_E_len_weighted"),
        ["partner_vollname", "n_briefe", "E_unweighted", "E_len_weighted", "delta_E_len_weighted", "delta_rank_E_len_weighted"],
    )
    md_table(
        f"Top {top_n} nach |ΔE| (1%-Trim ungewichtet vs ungewichtet)",
        top_abs("delta_E_trim_unweighted"),
        ["partner_vollname", "n_briefe", "E_unweighted", "E_trim_unweighted", "delta_E_trim_unweighted", "delta_rank_E_trim_unweighted"],
    )
    md_table(
        f"Top {top_n} nach |Rangshift| E (längen-gewichtet vs ungewichtet)",
        top_rankshift("delta_rank_E_len_weighted"),
        ["partner_vollname", "n_briefe", "rank_E_unweighted", "rank_E_len_weighted", "delta_rank_E_len_weighted", "delta_E_len_weighted"],
    )

    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

