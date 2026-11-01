#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute global (ego-network level) Shannon/Evenness for TOP100 predictions,
stratified by:
  - phase (EARLY/MIDDLE/LATE) using month cutoffs
  - direction (VON_HEINE / AN_HEINE)

Two global aggregation schemes:
  A) brief-weighted: aggregate weights across letters (each letter counts equally)
  B) partner-weighted: first aggregate per partner (equal letters per partner),
     then aggregate across partners (each partner counts equally)

Inputs:
  - Auswertungen/brief_predictions_top100.csv

Outputs:
  - Auswertungen/global_shannon_top100_by_phase_direction.csv
  - Auswertungen/global_shannon_top100_by_phase_direction.md
  - Auswertungen/plot_global_shannon_by_phase_direction.png/.pdf
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

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None  # type: ignore[assignment]
    HAVE_MPL = False

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

def parse_year_month(datum_iso: str) -> Optional[Tuple[int, int]]:
    """
    Accepts YYYY-MM-DD or YYYY or empty. Returns (year, month) if possible.
    - If only YYYY given, assume month=7 (mid-year) for phase mapping.
    """
    s = (datum_iso or "").strip()
    if not s:
        return None
    if len(s) >= 7 and s[4] == "-" and s[7:8] != "":
        try:
            y = int(s[0:4])
            m = int(s[5:7])
            return y, m
        except Exception:
            return None
    if len(s) == 4 and s.isdigit():
        return int(s), 7
    return None

def phase_for_year_month(y: int, m: int) -> Optional[str]:
    """
    User-defined phases:
      EARLY:  up to May 1831 (inclusive)
      MIDDLE: Jun 1831 through May 1848 (inclusive)
      LATE:   Jun 1848 through Feb 1856 (inclusive)
    """
    if y < 1831 or (y == 1831 and m <= 5):
        return "EARLY"
    if (y == 1831 and m >= 6) or (1832 <= y <= 1847) or (y == 1848 and m <= 5):
        return "MIDDLE"
    if (y == 1848 and m >= 6) or (1849 <= y <= 1855) or (y == 1856 and m <= 2):
        return "LATE"
    return None

def normalize_vec(v: List[float]) -> List[float]:
    s = sum(v)
    if s <= 0:
        return [1.0 / len(v)] * len(v)
    return [x / s for x in v]

def aggregate_brief_weighted(rows: List[Dict[str, str]]) -> Tuple[List[float], int]:
    sums = [0.0] * len(TARGET_COLS)
    n = 0
    for r in rows:
        n += 1
        for i, k in enumerate(TARGET_COLS):
            sums[i] += float((r.get(k) or "0").strip() or "0")
    if n == 0:
        return [0.0] * len(TARGET_COLS), 0
    mean = [x / n for x in sums]
    return normalize_vec(mean), n

def aggregate_partner_weighted(rows: List[Dict[str, str]]) -> Tuple[List[float], int, int]:
    # per partner mean (unweighted over letters), then mean over partners
    per_partner_sums: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(TARGET_COLS))
    per_partner_n: Dict[str, int] = defaultdict(int)
    for r in rows:
        p = (r.get("partner_vollname") or "").strip()
        if not p:
            continue
        per_partner_n[p] += 1
        for i, k in enumerate(TARGET_COLS):
            per_partner_sums[p][i] += float((r.get(k) or "0").strip() or "0")

    partners = [p for p in per_partner_n.keys() if per_partner_n[p] > 0]
    if not partners:
        return [0.0] * len(TARGET_COLS), 0, 0

    # mean profile per partner
    partner_profiles: List[List[float]] = []
    for p in partners:
        n = per_partner_n[p]
        mean = [x / n for x in per_partner_sums[p]]
        partner_profiles.append(normalize_vec(mean))

    # mean across partners
    agg = [0.0] * len(TARGET_COLS)
    for prof in partner_profiles:
        for i, x in enumerate(prof):
            agg[i] += x
    agg = [x / len(partner_profiles) for x in agg]
    agg = normalize_vec(agg)
    return agg, len(partners), sum(per_partner_n.values())

def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def save_fig(base: Path, fig) -> None:
    if not HAVE_MPL:
        return
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_csv",
        default=str(_DATA / "brief_predictions_top100.csv"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(_DATA / "global_shannon_top100_by_phase_direction.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(_DATA / "global_shannon_top100_by_phase_direction.md"),
    )
    ap.add_argument(
        "--out_plot_base",
        default=str(_DATA / "plot_global_shannon_by_phase_direction"),
    )
    args = ap.parse_args(argv)

    # load + stratify
    strata: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)  # (phase, direction) -> rows
    strata_all: Dict[str, List[Dict[str, str]]] = defaultdict(list)  # e.g. "ALL|VON_HEINE", "EARLY|ALL"

    with Path(args.pred_csv).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            direction = (row.get("richtung") or "").strip() or "UNKNOWN"
            ym = parse_year_month(row.get("datum_iso") or "")
            if not ym:
                continue
            ph = phase_for_year_month(ym[0], ym[1])
            if not ph:
                continue
            strata[(ph, direction)].append(row)
            strata_all[f"ALL|{direction}"].append(row)
            strata_all[f"{ph}|ALL"].append(row)
            strata_all["ALL|ALL"].append(row)

    def compute_block(label_phase: str, label_dir: str, rows_block: List[Dict[str, str]]) -> Dict[str, str]:
        bw_prof, n_letters = aggregate_brief_weighted(rows_block)
        pw_prof, n_partners, n_letters_pw = aggregate_partner_weighted(rows_block)

        H_bw = shannon_index(bw_prof, base=2.0)
        E_bw = evenness(H_bw, len(TARGET_COLS), base=2.0)
        H_pw = shannon_index(pw_prof, base=2.0)
        E_pw = evenness(H_pw, len(TARGET_COLS), base=2.0)

        out: Dict[str, str] = {
            "phase": label_phase,
            "richtung": label_dir,
            "n_letters": str(n_letters),
            "n_partners": str(n_partners),
            "H_brief_weighted": f"{H_bw:.6f}",
            "E_brief_weighted": f"{E_bw:.6f}",
            "H_partner_weighted": f"{H_pw:.6f}",
            "E_partner_weighted": f"{E_pw:.6f}",
        }
        # include aggregate distributions (optional but useful)
        for k, v in zip(TARGET_COLS, bw_prof):
            out[f"P_{k}_brief"] = f"{v:.10f}".rstrip("0").rstrip(".")
        for k, v in zip(TARGET_COLS, pw_prof):
            out[f"P_{k}_partner"] = f"{v:.10f}".rstrip("0").rstrip(".")
        return out

    out_rows: List[Dict[str, str]] = []

    # core strata (phase x direction)
    for ph in ("EARLY", "MIDDLE", "LATE"):
        for d in ("VON_HEINE", "AN_HEINE"):
            rows_block = strata.get((ph, d), [])
            out_rows.append(compute_block(ph, d, rows_block))

    # margins
    for k in ("ALL|VON_HEINE", "ALL|AN_HEINE", "EARLY|ALL", "MIDDLE|ALL", "LATE|ALL", "ALL|ALL"):
        ph, d = k.split("|")
        out_rows.append(compute_block(ph, d, strata_all.get(k, [])))

    fieldnames = [
        "phase",
        "richtung",
        "n_letters",
        "n_partners",
        "H_brief_weighted",
        "E_brief_weighted",
        "H_partner_weighted",
        "E_partner_weighted",
    ] + [f"P_{k}_brief" for k in TARGET_COLS] + [f"P_{k}_partner" for k in TARGET_COLS]

    write_csv(Path(args.out_csv), out_rows, fieldnames)

    # short markdown
    md: List[str] = []
    md.append("## Globaler Shannon‑Index (TOP‑100) nach Phase und Richtung\n\n")
    md.append(f"- **Input**: `{args.pred_csv}`\n")
    md.append("- **Phasen**: EARLY (bis Mai 1831), MIDDLE (Jun 1831–Mai 1848), LATE (Jun 1848–Feb 1856)\n")
    md.append("- **Gewichtungen**:\n")
    md.append("  - **brief‑gewichtet**: jeder Brief zählt gleich (Ereignis‑Perspektive)\n")
    md.append("  - **partner‑gewichtet**: erst je Partner mitteln, dann Partner gleichgewichten (Kanten‑Perspektive)\n\n")

    md.append("### Kernergebnisse (H)\n\n")
    md.append("| Phase | Richtung | n Briefe | n Partner | H (brief‑gew.) | H (partner‑gew.) |\n")
    md.append("|---|---|---:|---:|---:|---:|\n")
    for r in out_rows:
        if r["phase"] in ("EARLY", "MIDDLE", "LATE") and r["richtung"] in ("VON_HEINE", "AN_HEINE"):
            md.append(
                f"| {r['phase']} | {r['richtung']} | {r['n_letters']} | {r['n_partners']} | {r['H_brief_weighted']} | {r['H_partner_weighted']} |\n"
            )
    md.append("\n")
    md.append("### Marginalien (zusammengefasst)\n\n")
    md.append("| Phase | Richtung | n Briefe | n Partner | H (brief‑gew.) | H (partner‑gew.) |\n")
    md.append("|---|---|---:|---:|---:|---:|\n")
    for r in out_rows:
        if not (r["phase"] in ("EARLY", "MIDDLE", "LATE") and r["richtung"] in ("VON_HEINE", "AN_HEINE")):
            md.append(
                f"| {r['phase']} | {r['richtung']} | {r['n_letters']} | {r['n_partners']} | {r['H_brief_weighted']} | {r['H_partner_weighted']} |\n"
            )

    Path(args.out_md).write_text("".join(md), encoding="utf-8")

    # plot: grouped bars for H (brief-weighted vs partner-weighted) per stratum
    if HAVE_MPL:
        plt.style.use("seaborn-v0_8-whitegrid")
        labels: List[str] = []
        H_bw_list: List[float] = []
        H_pw_list: List[float] = []
        for r in out_rows:
            if r["phase"] in ("EARLY", "MIDDLE", "LATE") and r["richtung"] in ("VON_HEINE", "AN_HEINE"):
                labels.append(f"{r['phase']}\n{r['richtung']}")
                H_bw_list.append(float(r["H_brief_weighted"]))
                H_pw_list.append(float(r["H_partner_weighted"]))

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(labels)))
        width = 0.38
        ax.bar([i - width / 2 for i in x], H_bw_list, width=width, label="brief‑gewichtet", color="#2a6fbb")
        ax.bar([i + width / 2 for i in x], H_pw_list, width=width, label="partner‑gewichtet", color="#d14b4b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Shannon H (Basis 2)")
        ax.set_title("Globaler Shannon‑Index nach Phase und Richtung (TOP‑100)")
        ax.legend()
        save_fig(Path(args.out_plot_base), fig)
        plt.close(fig)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    if HAVE_MPL:
        print(f"Wrote: {args.out_plot_base}.png/.pdf")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

