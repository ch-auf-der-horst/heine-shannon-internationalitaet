#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisierungen zur Internationalität (Codebook: Nationalität) × Phase
auf Basis von data/shannon_H_by_metadata_phase.csv.

Aggregationen:
  (A) Wie im Report shannon_H_by_metadata_phase_report.md: Mittelwert von H über
      alle Zeilen (partner × phase × richtung) je (Phase, Nationalität).
  (B) Partner-zentriert: pro (partner_vollname, phase, Nationalität) Mittel über
      Richtungen, dann Verteilungsplots (Violin/Strip).

Outputs (data/):
  - plot_internationality_H_de_fr.png/.pdf
  - plot_internationality_heatmap.png/.pdf
  - plot_internationality_partner_violin_de_fr.png/.pdf
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
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PHASES: Tuple[str, ...] = ("EARLY", "MIDDLE", "LATE")

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))

def save_fig(base: Path, fig) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def sem(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(v / n)

def aggregate_report_style(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """(phase, nat) -> (mean_H, sem, n) über alle Zeilen wie im Markdown-Report."""
    by: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        nat = (r.get("Nationalität") or "").strip()
        if not nat:
            continue
        ph = (r.get("phase") or "").strip()
        try:
            h = float((r.get("shannon_H") or "0").strip() or "0")
        except ValueError:
            continue
        by[(ph, nat)].append(h)
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for k, xs in by.items():
        m = sum(xs) / len(xs)
        out[k] = (m, sem(xs), len(xs))
    return out

def partner_collapsed(
    rows: List[Dict[str, str]], nationalities: Sequence[str]
) -> Dict[Tuple[str, str, str], float]:
    """(partner, phase, nat) -> Mittel-H über Richtungen, gefiltert auf nationalities."""
    want = set(nationalities)
    acc: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        nat = (r.get("Nationalität") or "").strip()
        if nat not in want:
            continue
        ph = (r.get("phase") or "").strip()
        partner = (r.get("partner_vollname") or "").strip()
        if not partner:
            continue
        try:
            h = float((r.get("shannon_H") or "0").strip() or "0")
        except ValueError:
            continue
        acc[(partner, ph, nat)].append(h)
    return {k: sum(v) / len(v) for k, v in acc.items()}

def heatmap_matrix(
    agg: Dict[Tuple[str, str], Tuple[float, float, int]], nationalities: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Werte-Matrix und Mask (True = leer)."""
    mat = np.full((len(nationalities), len(PHASES)), np.nan, dtype=float)
    mask = np.ones_like(mat, dtype=bool)
    for i, nat in enumerate(nationalities):
        for j, ph in enumerate(PHASES):
            t = agg.get((ph, nat))
            if t is None:
                continue
            m, _, n = t
            if n <= 0:
                continue
            mat[i, j] = m
            mask[i, j] = False
    return mat, mask

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plots: Internationalität × Shannon H")
    ap.add_argument(
        "--in_csv",
        type=Path,
        default=_DATA / "shannon_H_by_metadata_phase.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_DATA,
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    rows = read_rows(args.in_csv)
    agg = aggregate_report_style(rows)

    # Alle Nationalitäten mit mindestens einem Eintrag
    nats_set = {k[1] for k in agg.keys()}
    # Sortierung: deutsch, französisch zuerst, dann alphabetisch
    primary = ["deutsch", "französisch"]
    rest = sorted(n for n in nats_set if n not in primary)
    nationalities_order = [n for n in primary if n in nats_set] + rest

    plt.style.use("seaborn-v0_8-whitegrid")

    # ---------- Plot 1: Deutschland vs Frankreich (Hauptgruppen) ----------
    col_de = "#1B4965"
    col_fr = "#C1121F"
    fig1, ax1 = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x = np.arange(len(PHASES))
    w = 0.36

    def bar_vals(nat: str) -> Tuple[List[float], List[float], List[int]]:
        means, ses, ns = [], [], []
        for ph in PHASES:
            t = agg.get((ph, nat))
            if t is None:
                means.append(float("nan"))
                ses.append(0.0)
                ns.append(0)
            else:
                m, s, n = t
                means.append(m)
                ses.append(s)
                ns.append(n)
        return means, ses, ns

    m_de, se_de, n_de = bar_vals("deutsch")
    m_fr, se_fr, n_fr = bar_vals("französisch")

    ax1.bar(x - w / 2, m_de, width=w, yerr=se_de, capsize=3, color=col_de, alpha=0.92, label="deutsch", edgecolor="white", linewidth=0.8)
    ax1.bar(x + w / 2, m_fr, width=w, yerr=se_fr, capsize=3, color=col_fr, alpha=0.92, label="französisch", edgecolor="white", linewidth=0.8)

    ymax = max(
        [v for v in m_de + m_fr if not math.isnan(v)] + [2.85],
        default=3.0,
    )
    ymin = min([v for v in m_de + m_fr if not math.isnan(v)] + [2.85], default=2.8)
    pad = 0.06 * (ymax - ymin + 1e-6)
    ax1.set_ylim(ymin - pad, ymax + pad)

    for i, ph in enumerate(PHASES):
        if n_de[i] > 0:
            ax1.text(
                i - w / 2,
                m_de[i] + se_de[i] + pad * 0.35,
                f"n={n_de[i]}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=col_de,
            )
        if n_fr[i] > 0:
            ax1.text(
                i + w / 2,
                m_fr[i] + se_fr[i] + pad * 0.35,
                f"n={n_fr[i]}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=col_fr,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(PHASES)
    ax1.set_ylabel("Mittleres Shannon-H (Funktionsdiversität)")
    ax1.set_xlabel("Phase")
    ax1.set_title(
        "Internationalität (Nationalität): mittleres H nach Phase\n"
        "(Hauptgruppen deutsch vs. französisch; Fehlerbalken: Standardfehler des Mittelwerts)"
    )
    ax1.legend(loc="lower right", frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    save_fig(args.out_dir / "plot_internationality_H_de_fr", fig1)

    # ---------- Plot 2: Heatmap alle Nationalitäten ----------
    mat, mask = heatmap_matrix(agg, nationalities_order)
    fig2, ax2 = plt.subplots(figsize=(8.8, max(5.0, 0.38 * len(nationalities_order) + 1.5)), constrained_layout=True)
    # Masked array für leere Zellen
    plot_data = np.ma.array(mat, mask=np.isnan(mat))
    im = ax2.imshow(plot_data, aspect="auto", cmap="turbo", vmin=2.82, vmax=3.02, interpolation="nearest")
    ax2.set_xticks(range(len(PHASES)))
    ax2.set_xticklabels(PHASES)
    ax2.set_yticks(range(len(nationalities_order)))
    ax2.set_yticklabels(nationalities_order, fontsize=9)
    cbar = fig2.colorbar(im, ax=ax2, location="right", shrink=0.82, pad=0.02)
    cbar.set_label("Mittleres H")
    for i in range(len(nationalities_order)):
        for j in range(len(PHASES)):
            if mask[i, j]:
                ax2.text(j, i, "—", ha="center", va="center", color="#888", fontsize=9)
            else:
                m, s, n = agg[(PHASES[j], nationalities_order[i])]
                ax2.text(j, i, f"{m:.2f}\n(n={n})", ha="center", va="center", fontsize=7, color="white" if m > 2.94 else "black")
    ax2.set_title("Internationalität: mittleres Shannon-H (Phase × Nationalität)\n(leere Felder: keine Fälle im TOP-100-Stratum)")
    save_fig(args.out_dir / "plot_internationality_heatmap", fig2)

    # ---------- Plot 3: Partner-zentrierte Verteilung deutsch vs. französisch ----------
    pc = partner_collapsed(rows, ("deutsch", "französisch"))
    by_ph: Dict[str, Dict[str, List[float]]] = {ph: {"deutsch": [], "französisch": []} for ph in PHASES}
    for (_partner, ph, nat), h in pc.items():
        by_ph[ph][nat].append(h)

    fig3, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)

    def plot_phase_violin(ax, ph: str) -> None:
        d_de = by_ph[ph]["deutsch"]
        d_fr = by_ph[ph]["französisch"]
        nd, nf = len(d_de), len(d_fr)
        if nd == 0 and nf == 0:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{ph}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["deutsch", "französisch"])
            return
        datasets: List[List[float]] = []
        positions: List[int] = []
        colors: List[str] = []
        if nd > 0:
            datasets.append(d_de)
            positions.append(0)
            colors.append(col_de)
        if nf > 0:
            datasets.append(d_fr)
            positions.append(1)
            colors.append(col_fr)
        parts = ax.violinplot(
            datasets,
            positions=positions,
            showmeans=True,
            showmedians=False,
            widths=0.72,
        )
        for b, c in zip(parts["bodies"], colors):
            b.set_facecolor(c)
            b.set_alpha(0.78)
            b.set_edgecolor("white")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["deutsch", "französisch"])
        ax.set_title(f"{ph}\n(Partner: deutsch n={nd}, franz. n={nf})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax, ph in zip(axes, PHASES):
        plot_phase_violin(ax, ph)
        if ph == "EARLY":
            ax.set_ylabel("Shannon-H (Mittel je Partner über Richtungen)")
    fig3.suptitle(
        "Verteilung der Funktionsdiversität: Partner-Mittelwerte (deutsch vs. französisch)",
        y=1.03,
    )
    save_fig(args.out_dir / "plot_internationality_partner_violin_de_fr", fig3)

    print(f"Wrote plots in {args.out_dir}:")
    print("  plot_internationality_H_de_fr.png/.pdf")
    print("  plot_internationality_heatmap.png/.pdf")
    print("  plot_internationality_partner_violin_de_fr.png/.pdf")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
