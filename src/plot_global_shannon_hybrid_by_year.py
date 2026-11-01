#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid-Plot der globalen Shannon-Diversitaet:
- Kurven folgen den tatsaechlichen Jahresdaten (datennahe Entwicklung)
- EARLY/MIDDLE/LATE bleiben als explizite Stuetzpunkte markiert

Input:
  - data/brief_predictions_top100.csv

Output:
  - data/plot_global_shannon_hybrid_by_year.png/.pdf

Fuer Spyder:
  - Skript direkt ausfuehren (Run)
  - Optional unten DEFAULT_* Konstanten anpassen
"""

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# Phasendefinition konsistent zur bestehenden Makro-Auswertung
PHASES: Tuple[str, ...] = ("EARLY", "MIDDLE", "LATE")
PHASE_BOUNDS = {
    "EARLY": ((-10_000, 1), (1831, 5)),
    "MIDDLE": ((1831, 6), (1848, 5)),
    "LATE": ((1848, 6), (1856, 2)),
}

DEFAULT_PRED_CSV = Path(
    str(_DATA / "brief_predictions_top100.csv")
)
DEFAULT_OUT_BASE = Path(
    str(_DATA / "plot_global_shannon_hybrid_by_year")
)

def parse_year_month(datum_iso: str) -> Optional[Tuple[int, int]]:
    s = (datum_iso or "").strip()
    if not s:
        return None
    if len(s) >= 7 and s[4] == "-":
        try:
            return int(s[0:4]), int(s[5:7])
        except Exception:
            return None
    if len(s) == 4 and s.isdigit():
        return int(s), 7
    return None

def phase_for_year_month(y: int, m: int) -> Optional[str]:
    if y < 1831 or (y == 1831 and m <= 5):
        return "EARLY"
    if (y == 1831 and m >= 6) or (1832 <= y <= 1847) or (y == 1848 and m <= 5):
        return "MIDDLE"
    if (y == 1848 and m >= 6) or (1849 <= y <= 1855) or (y == 1856 and m <= 2):
        return "LATE"
    return None

def normalize_vec(v: Iterable[float]) -> List[float]:
    vals = [float(x) for x in v]
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [x / s for x in vals]

def shannon_base2(probs: Sequence[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * (math.log(p) / math.log(2.0))
    return h

def aggregate_brief_weighted(rows: Sequence[Dict[str, str]]) -> Tuple[List[float], int]:
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

def aggregate_partner_weighted(rows: Sequence[Dict[str, str]]) -> Tuple[List[float], int]:
    per_partner_sum: Dict[str, List[float]] = defaultdict(lambda: [0.0] * len(TARGET_COLS))
    per_partner_n: Dict[str, int] = defaultdict(int)
    for r in rows:
        p = (r.get("partner_vollname") or "").strip()
        if not p:
            continue
        per_partner_n[p] += 1
        for i, k in enumerate(TARGET_COLS):
            per_partner_sum[p][i] += float((r.get(k) or "0").strip() or "0")

    partners = [p for p in per_partner_n if per_partner_n[p] > 0]
    if not partners:
        return [0.0] * len(TARGET_COLS), 0

    profs = []
    for p in partners:
        n = per_partner_n[p]
        profs.append(normalize_vec([x / n for x in per_partner_sum[p]]))

    agg = [0.0] * len(TARGET_COLS)
    for prof in profs:
        for i, x in enumerate(prof):
            agg[i] += x
    agg = [x / len(profs) for x in agg]
    return normalize_vec(agg), len(partners)

def rolling_mean(xs: List[float], win: int = 3) -> List[float]:
    if win <= 1 or len(xs) <= 2:
        return list(xs)
    out = []
    half = win // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(float(np.mean(xs[lo:hi])))
    return out

def linear_trend(years: List[float], vals: List[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(years) < 2 or len(vals) < 2:
        return None
    x = np.asarray(years, dtype=float)
    y = np.asarray(vals, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    coef = np.polyfit(x, y, deg=1)
    y_fit = np.polyval(coef, x)
    return x, y_fit

def phase_mid_decimal(phase: str) -> float:
    (y0, m0), (y1, m1) = PHASE_BOUNDS[phase]
    if y0 < 0:
        y0, m0 = 1820, 1
    a = y0 + (m0 - 0.5) / 12.0
    b = y1 + (m1 - 0.5) / 12.0
    return (a + b) / 2.0

def save_fig(base: Path, fig) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

def main(pred_csv: Path = DEFAULT_PRED_CSV, out_base: Path = DEFAULT_OUT_BASE, smooth_window: int = 3) -> int:
    by_year_dir: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    by_phase_dir: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            direction = (row.get("richtung") or "").strip()
            if direction not in {"VON_HEINE", "AN_HEINE"}:
                continue
            ym = parse_year_month(row.get("datum_iso") or "")
            if not ym:
                continue
            y, m = ym
            phase = phase_for_year_month(y, m)
            if not phase:
                continue
            by_year_dir[(y, direction)].append(row)
            by_phase_dir[(phase, direction)].append(row)

    dirs = ("VON_HEINE", "AN_HEINE")
    colors = {"VON_HEINE": "#1f77b4", "AN_HEINE": "#d62728"}
    labels = {"VON_HEINE": "VON HEINE", "AN_HEINE": "AN HEINE"}

    yearly_brief: Dict[str, Tuple[List[float], List[float]]] = {}
    yearly_partner: Dict[str, Tuple[List[float], List[float]]] = {}
    all_h_vals: List[float] = []

    years_all = sorted({y for (y, _d) in by_year_dir.keys()})
    if not years_all:
        raise RuntimeError("Keine gueltigen Jahresdaten in brief_predictions_top100.csv gefunden.")

    for d in dirs:
        years = sorted([y for y in years_all if (y, d) in by_year_dir])
        h_b = []
        h_p = []
        for y in years:
            prof_b, n_b = aggregate_brief_weighted(by_year_dir[(y, d)])
            if n_b == 0:
                continue
            prof_p, n_partners = aggregate_partner_weighted(by_year_dir[(y, d)])
            if n_partners == 0:
                continue
            h_b.append(shannon_base2(prof_b))
            h_p.append(shannon_base2(prof_p))
        yearly_brief[d] = (years, rolling_mean(h_b, win=smooth_window))
        yearly_partner[d] = (years, rolling_mean(h_p, win=smooth_window))
        all_h_vals.extend(h_b)
        all_h_vals.extend(h_p)

    phase_points_brief: Dict[str, Tuple[List[float], List[float]]] = {}
    phase_points_partner: Dict[str, Tuple[List[float], List[float]]] = {}
    for d in dirs:
        xs = []
        yb = []
        yp = []
        for ph in PHASES:
            rows = by_phase_dir.get((ph, d), [])
            prof_b, n_b = aggregate_brief_weighted(rows)
            prof_p, n_p = aggregate_partner_weighted(rows)
            if n_b == 0 or n_p == 0:
                continue
            xs.append(phase_mid_decimal(ph))
            yb.append(shannon_base2(prof_b))
            yp.append(shannon_base2(prof_p))
        phase_points_brief[d] = (xs, yb)
        phase_points_partner[d] = (xs, yp)

    y_min = min(all_h_vals) - 0.004
    y_max = max(all_h_vals) + 0.004

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.5), sharey=True)
    fig.suptitle(
        "Makro-Diversitaet im Ego-Netzwerk: datennahe Jahreskurven mit Phasen-Stuetzpunkten",
        fontsize=13,
    )
    panel_meta = [
        ("Globales H (brief-gewichtet)", yearly_brief, phase_points_brief),
        ("Globales H (partner-gewichtet)", yearly_partner, phase_points_partner),
    ]

    for ax, (title, yearly_src, phase_src) in zip(axes, panel_meta):
        # Phase-Baender
        ax.axvspan(1820.0, 1831.42, color="#f3f6fb", alpha=0.8, zorder=0)
        ax.axvspan(1831.42, 1848.42, color="#eef3fa", alpha=0.7, zorder=0)
        ax.axvspan(1848.42, 1856.20, color="#f3f6fb", alpha=0.8, zorder=0)
        ax.axvline(1831.42, color="#9aa0a6", lw=0.8, ls="--", alpha=0.8, zorder=1)
        ax.axvline(1848.42, color="#9aa0a6", lw=0.8, ls="--", alpha=0.8, zorder=1)

        for d in dirs:
            years, hs = yearly_src[d]
            if not years:
                continue
            ax.plot(years, hs, color=colors[d], lw=2.0, alpha=0.95, label=labels[d], zorder=3)
            tr = linear_trend(years, hs)
            if tr is not None:
                x_tr, y_tr = tr
                # Gepunktete Trendlinie = Gesamtentwicklung ueber den Zeitraum
                ax.plot(x_tr, y_tr, color=colors[d], lw=1.35, ls=":", alpha=0.9, zorder=2)
            xs, ys = phase_src[d]
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=62,
                    color=colors[d],
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=5,
                )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Jahr (durchgezogen = Jahresverlauf, gepunktet = linearer Trend)", fontsize=9.5)
        ax.grid(alpha=0.28, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(1820, 1856.3)
        ax.set_ylim(y_min, y_max)
        # Phase-Labels oben
        ax.text(1825.9, y_max - 0.00045, "EARLY", ha="center", va="top", fontsize=8.5, color="#4d5968")
        ax.text(1839.9, y_max - 0.00045, "MIDDLE", ha="center", va="top", fontsize=8.5, color="#4d5968")
        ax.text(1852.0, y_max - 0.00045, "LATE", ha="center", va="top", fontsize=8.5, color="#4d5968")

    axes[0].set_ylabel("Shannon H (Basis 2)", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True, fontsize=8.8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))

    save_fig(out_base, fig)
    plt.close(fig)

    print(f"Wrote: {out_base.with_suffix('.png')}")
    print(f"Wrote: {out_base.with_suffix('.pdf')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

