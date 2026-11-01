#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis + visualization for correspondent-level Shannon/Evenness results.

Inputs:
  - Auswertungen/korrespondenten_profile_pred_top100.csv
Optional:
  - top_100_korrespondenten_zum_labeln.csv (adds role/group columns if present)

Outputs (in Auswertungen/):
  - shannon_summary.md
  - shannon_top_bottom.csv
  - plot_shannon_hist.png/.pdf
  - plot_shannon_top_bottom.png/.pdf
  - plot_shannon_scatter_nbriefe.png/.pdf
  - plot_category_heatmap.png/.pdf (10-category profile heatmap)
"""

from __future__ import annotations
import argparse
import csv
import math
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
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

def spearman_rho(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(vals: List[float]) -> List[float]:
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(idx):
            j = i
            v = vals[idx[i]]
            while j < len(idx) and vals[idx[j]] == v:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in idx[i:j]:
                r[k] = avg
            i = j
        return r

    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)

def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)
    return rows

def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def maybe_merge_top100(
    core_rows: List[Dict[str, str]], top100_path: Optional[Path]
) -> Tuple[List[Dict[str, str]], List[str]]:
    if not top100_path or not top100_path.exists():
        return core_rows, []

    top_rows = load_csv(top100_path)
    if not top_rows:
        return core_rows, []

    # identify name column in top100 table
    name_col = None
    for cand in ("Vollname", "partner_vollname", "Name", "vollname"):
        if cand in top_rows[0]:
            name_col = cand
            break
    if not name_col:
        return core_rows, []

    # build index by name
    idx: Dict[str, Dict[str, str]] = {}
    for tr in top_rows:
        n = (tr.get(name_col) or "").strip()
        if n:
            idx[n] = tr

    # choose metadata columns to carry over (exclude duplicates)
    core_keys = set(core_rows[0].keys()) if core_rows else set()
    meta_cols = [k for k in top_rows[0].keys() if k not in core_keys and k != name_col]

    merged: List[Dict[str, str]] = []
    for r in core_rows:
        n = (r.get("partner_vollname") or "").strip()
        m = dict(r)
        tr = idx.get(n)
        if tr:
            for k in meta_cols:
                m[k] = tr.get(k, "")
        merged.append(m)
    return merged, meta_cols

def save_fig(base: Path, fig) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles_csv",
        default=str(_DATA / "korrespondenten_profile_pred_top100.csv"),
    )
    ap.add_argument(
        "--top100_csv",
        default=str(_LOCAL / "top_100_korrespondenten.csv"),
        help="Optional: enrich output with columns from this table (matched by Vollname).",
    )
    ap.add_argument(
        "--out_dir",
        default=str(_DATA),
    )
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = load_csv(Path(args.profiles_csv))
    if not core:
        raise SystemExit("No rows in profiles CSV.")

    rows, meta_cols = maybe_merge_top100(core, Path(args.top100_csv) if args.top100_csv else None)

    # parse
    partners: List[str] = []
    n_briefe: List[int] = []
    H: List[float] = []
    E: List[float] = []
    mat: List[List[float]] = []

    for r in rows:
        partners.append((r.get("partner_vollname") or "").strip())
        n_briefe.append(int(float((r.get("n_briefe") or "0").strip() or "0")))
        H.append(float((r.get("shannon_H") or "nan").strip()))
        E.append(float((r.get("evenness_E") or "nan").strip()))
        mat.append([float((r.get(k) or "0").strip() or "0") for k in TARGET_COLS])

    # summary stats
    def q(vals: List[float], p: float) -> float:
        s = sorted(vals)
        if not s:
            return float("nan")
        i = (len(s) - 1) * p
        lo = int(math.floor(i))
        hi = int(math.ceil(i))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - i) + s[hi] * (i - lo)

    spearman_n = spearman_rho([float(x) for x in n_briefe], H)

    # top/bottom by H
    order = sorted(range(len(rows)), key=lambda i: H[i], reverse=True)
    top_n = int(args.top_n)
    top_idx = order[:top_n]
    bottom_idx = list(reversed(order[-top_n:]))

    # write top/bottom table (CSV)
    tb_rows: List[Dict[str, str]] = []
    for label, idxs in (("TOP", top_idx), ("BOTTOM", bottom_idx)):
        for i in idxs:
            row = {
                "group": label,
                "partner_vollname": partners[i],
                "n_briefe": str(n_briefe[i]),
                "shannon_H": f"{H[i]:.6f}",
                "evenness_E": f"{E[i]:.6f}",
            }
            # carry selected metadata if present
            for k in meta_cols:
                row[k] = (rows[i].get(k) or "").strip()
            tb_rows.append(row)

    fieldnames = ["group", "partner_vollname", "n_briefe", "shannon_H", "evenness_E"] + meta_cols
    write_csv(out_dir / "shannon_top_bottom.csv", tb_rows, fieldnames)

    # write summary markdown
    md: List[str] = []
    md.append("## Auswertung: Shannon/Evenness (TOP-100, Korrespondentenebene)\n\n")
    md.append(f"- **Input**: `{args.profiles_csv}`\n")
    md.append(f"- **Anzahl Korrespondenten**: {len(rows)}\n")
    md.append(f"- **Top/Bottom-Listen**: `{out_dir / 'shannon_top_bottom.csv'}`\n\n")

    md.append("### Deskriptive Statistik\n\n")
    md.append(f"- **Shannon H**: Mittel {statistics.mean(H):.4f}, Median {q(H,0.5):.4f}, Q1 {q(H,0.25):.4f}, Q3 {q(H,0.75):.4f}\n")
    md.append(f"- **Evenness E**: Mittel {statistics.mean(E):.4f}, Median {q(E,0.5):.4f}, Q1 {q(E,0.25):.4f}, Q3 {q(E,0.75):.4f}\n")
    md.append(f"- **Briefanzahl n_briefe**: Mittel {statistics.mean(n_briefe):.1f}, Median {q([float(x) for x in n_briefe],0.5):.0f}\n")
    md.append(f"- **Spearman( n_briefe , H )**: {spearman_n:.4f}\n\n")

    md.append("### Interpretation (knapp)\n\n")
    md.append("Hohe H/E-Werte bedeuten, dass das Kategorienprofil eines Korrespondenten **relativ gleichmäßig** über mehrere relationale Funktionskategorien verteilt ist; niedrige Werte deuten auf **stärkere Konzentration** auf wenige Kategorien hin. Der Spearman-Wert zeigt, ob Diversität systematisch mit der Anzahl der Briefe zusammenhängt (Mengenartefakt) oder nicht.\n\n")

    md.append("### Top/Bottom nach Shannon H\n\n")
    md.append(f"Die vollständigen Listen stehen in `{out_dir / 'shannon_top_bottom.csv'}`.\n\n")

    (out_dir / "shannon_summary.md").write_text("".join(md), encoding="utf-8")

    # ---- plots ----
    plt.style.use("seaborn-v0_8-whitegrid")

    # histogram (modernized): distribution + central tendency + illustrative cases
    fig1, ax1 = plt.subplots(figsize=(9.5, 5.2))
    counts, bins, _ = ax1.hist(
        H,
        bins=14,
        color="#4C78A8",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    h_mean = statistics.mean(H)
    h_med = q(H, 0.5)
    ax1.axvline(h_mean, color="#E45756", linestyle="--", linewidth=1.8, label=f"Mittelwert: {h_mean:.3f}")
    ax1.axvline(h_med, color="#2CA02C", linestyle="-.", linewidth=1.8, label=f"Median: {h_med:.3f}")
    # Rug-Plot für Einzelprofile am unteren Rand
    ax1.scatter(H, [-0.35] * len(H), s=12, color="#6B7280", alpha=0.65, zorder=3)
    ax1.set_ylim(-0.8, max(counts) * 1.14 if len(counts) else 1.0)
    # Beispiel-Korrespondenzen: niedrigster, mittlerer, höchster H
    idx_sorted_asc = sorted(range(len(H)), key=lambda i: H[i])
    ex_low, ex_mid, ex_high = idx_sorted_asc[0], idx_sorted_asc[len(idx_sorted_asc) // 2], idx_sorted_asc[-1]
    ex_points = [
        ("niedrig", ex_low, "#B2182B", f"{partners[ex_low]} (niedrig)"),
        ("mittel", ex_mid, "#2166AC", f"{partners[ex_mid]} (mittel)"),
        ("hoch", ex_high, "#1A9850", f"{partners[ex_high]} (hoch)"),
    ]
    for _, i, c, lbl in ex_points:
        ax1.scatter(
            [H[i]],
            [-0.35],
            s=58,
            color=c,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label=lbl,
        )
    ex_text = (
        f"Hoch: {partners[ex_high]} (H={H[ex_high]:.3f})\n"
        f"Mittel: {partners[ex_mid]} (H={H[ex_mid]:.3f})\n"
        f"Niedrig: {partners[ex_low]} (H={H[ex_low]:.3f})"
    )
    ax1.text(
        0.985,
        0.96,
        ex_text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    ax1.set_title("Verteilung des Shannon-Index H (Korrespondentenprofile)")
    ax1.set_xlabel("Shannon H (Basis 2)")
    ax1.set_ylabel("Anzahl Korrespondenten")
    ax1.legend(loc="upper left", frameon=True, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    save_fig(out_dir / "plot_shannon_hist", fig1)
    plt.close(fig1)

    # top/bottom: links H, rechts Kategorienprofile (Abb. 6)
    idxs = bottom_idx + top_idx
    vals = [H[i] for i in idxs]
    n_show = len(idxs)
    y = np.arange(n_show)
    n_bottom = len(bottom_idx)
    highlight_top_bottom = {
        "Mathilde Heine",
        "Charles Magnin",
        "Christine de Belgiojoso",
        "Fanny Lewald",
        "Ferdinand Lassalle",
        "Alfred Meissner",
    }

    fig2, (ax2a, ax2b) = plt.subplots(
        1,
        2,
        figsize=(14.2, 8.6),
        gridspec_kw={"width_ratios": [1.05, 1.45], "wspace": 0.08},
        sharey=True,
    )
    x0 = min(vals) - 0.03
    x1 = max(vals) + 0.02

    # Hintergrund: Bottom- vs. Top-Gruppe; Trennlinie
    ax2a.axhspan(-0.5, n_bottom - 0.5, facecolor="#FEF2F2", alpha=0.45, zorder=0)
    ax2a.axhspan(n_bottom - 0.5, n_show - 0.5, facecolor="#EFF6FF", alpha=0.45, zorder=0)
    ax2a.axhline(n_bottom - 0.5, color="#9CA3AF", linewidth=1.1, zorder=1)
    ax2b.axhspan(-0.5, n_bottom - 0.5, facecolor="#FEF2F2", alpha=0.45, zorder=0)
    ax2b.axhspan(n_bottom - 0.5, n_show - 0.5, facecolor="#EFF6FF", alpha=0.45, zorder=0)
    ax2b.axhline(n_bottom - 0.5, color="#9CA3AF", linewidth=1.1, zorder=1)

    # Lesefälle: dezenter Hintergrund und Rahmen
    for yi, i in enumerate(idxs):
        if partners[i] in highlight_top_bottom:
            for ax in (ax2a, ax2b):
                ax.axhspan(yi - 0.44, yi + 0.44, facecolor="#FEF3C7", alpha=0.55, zorder=1)
            ax2a.barh(
                yi,
                x1 - x0,
                height=0.88,
                left=x0,
                color="none",
                edgecolor="#D97706",
                linewidth=1.1,
                zorder=2,
            )
            ax2b.barh(
                yi,
                1.0,
                height=0.88,
                left=0,
                color="none",
                edgecolor="#D97706",
                linewidth=1.1,
                zorder=2,
            )

    # Linkes Panel: Lollipop-Diagramm
    colors = ["#D95F5F"] * n_bottom + ["#4C78A8"] * (n_show - n_bottom)
    for yi, i, hval, c in zip(y, idxs, vals, colors):
        lw = 2.4 if partners[i] in highlight_top_bottom else 2.0
        ms = 42 if partners[i] in highlight_top_bottom else 34
        ax2a.hlines(yi, x0, hval, color=c, linewidth=lw, alpha=0.9, zorder=3)
        ax2a.scatter(
            hval,
            yi,
            s=ms,
            color=c,
            edgecolor="#D97706" if partners[i] in highlight_top_bottom else "white",
            linewidth=0.9 if partners[i] in highlight_top_bottom else 0.5,
            zorder=4,
        )
        ax2a.text(hval + 0.0015, yi, f"{hval:.3f}", va="center", fontsize=7, color="#374151", zorder=5)
    ax2a.set_yticks(y)
    yticklabels = ax2a.set_yticklabels(
        [f"{partners[i]} (n={n_briefe[i]})" for i in idxs],
        fontsize=7.5,
    )
    for lbl, i in zip(yticklabels, idxs):
        if partners[i] in highlight_top_bottom:
            lbl.set_fontweight("bold")
    ax2a.invert_yaxis()
    ax2a.set_xlim(x0, x1)
    ax2a.set_xlabel("Shannon H (Basis 2)")
    ax2a.set_title(f"Bottom {top_n} / Top {top_n}: Shannon-Diversität", fontsize=11)
    ax2a.grid(axis="x", alpha=0.25, zorder=0)
    ax2a.spines["top"].set_visible(False)
    ax2a.spines["right"].set_visible(False)

    # Rechtes Panel: gestapelte Kategorienprofile
    cmap_cat = plt.get_cmap("turbo")
    turbo_steps = np.linspace(0.08, 0.92, len(TARGET_COLS))
    cat_colors = {cat: cmap_cat(t) for cat, t in zip(TARGET_COLS, turbo_steps)}
    left_stack = np.zeros(n_show, dtype=float)
    for cat in TARGET_COLS:
        cat_vals = np.array([mat[i][TARGET_COLS.index(cat)] for i in idxs], dtype=float)
        ax2b.barh(
            y,
            cat_vals,
            height=0.88,
            left=left_stack,
            color=cat_colors[cat],
            edgecolor="white",
            linewidth=0.35,
            label=cat,
            zorder=3,
        )
        left_stack += cat_vals
    ax2b.set_xlim(0, 1.0)
    ax2b.set_xlabel("Kategorienprofil (Gewichtssumme = 1)")
    ax2b.set_title("Relationale Funktionsprofile", fontsize=11)
    ax2b.grid(axis="x", alpha=0.2, zorder=0)
    ax2b.tick_params(axis="y", left=False, labelleft=False)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(False)

    handles, labels_legend = ax2b.get_legend_handles_labels()
    fig2.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig2.suptitle(
        f"Bottom {top_n} und Top {top_n} Korrespondenten nach Shannon H\n"
        "Links: Diversitätswert, rechts: Zusammensetzung des Kategorienprofils",
        y=0.98,
        fontsize=12,
    )
    fig2.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.14)
    save_fig(out_dir / "plot_shannon_top_bottom", fig2)
    plt.close(fig2)

    # scatter n vs H: Beschriftungen einheitlich rechts vom Punkt, vertikal gestaffelt, dünne Führungslinien
    fig3, ax3 = plt.subplots(figsize=(10.2, 6.8))
    ax3.scatter(n_briefe, H, s=30, alpha=0.75, color="#2a6fbb", edgecolor="white", linewidth=0.4)
    _xmax = max(n_briefe) * 1.14
    ax3.set_xlim(0, _xmax)
    ax3.set_xlabel("n_briefe (Anzahl Briefe im TOP-100-Korpus)")
    ax3.set_ylabel("Shannon H (Basis 2)")
    ax3.set_title(
        "Streudiagramm von Shannon gegen die Briefanzahl pro Korrespondent im TOP-100-Korpus;\n"
        f"der sehr geringe Spearman-Wert ({spearman_n:.4f}) spricht gegen einen reinen Mengeneffekt.",
        fontsize=10,
    )
    # Selektive Labels für Lesbarkeit:
    # 1) größter n_briefe-Ausreißer (rot) + niedrigster Shannon-H (rot)
    # 2) zwei Fälle mit kleiner Briefzahl und hohem H
    # 3) drei Fälle mit kleiner Briefzahl und niedrigem H (inkl. zweiter Punkt unten links)
    # 4) Korrespondenten mit mittlerer bis hoher Briefzahl (ca. 50–120): nur wenige,
    #    damit sich die Beschriftungen nicht überdecken
    idx_sorted_n = sorted(range(len(n_briefe)), key=lambda i: n_briefe[i], reverse=True)
    idx_max_n = idx_sorted_n[0]
    idx_min_h = min(range(len(H)), key=lambda i: H[i])
    highlight_red = {idx_max_n, idx_min_h}
    small_n_idx = [i for i in range(len(n_briefe)) if n_briefe[i] <= 10]
    small_n_high = sorted(small_n_idx, key=lambda i: H[i], reverse=True)[:2]
    small_n_low = sorted(small_n_idx, key=lambda i: H[i])[:3]
    mid_n_lo, mid_n_hi = 50, 120
    mid_band_idx = [
        i
        for i in range(len(n_briefe))
        if mid_n_lo <= n_briefe[i] <= mid_n_hi and i not in highlight_red
    ]
    mid_band_sorted = sorted(mid_band_idx, key=lambda i: n_briefe[i], reverse=True)
    mid_n_labels_max = 4
    mid_band_labeled = mid_band_sorted[:mid_n_labels_max]
    scatter_label_fs = 7.5
    anno_idx: List[int] = []
    for i in [idx_max_n, idx_min_h] + small_n_high + small_n_low + mid_band_labeled:
        if i not in anno_idx:
            anno_idx.append(i)

    def _label_offset(i: int) -> Tuple[float, float, str, str]:
        """Offset points (dx, dy), horizontal and vertical alignment."""
        dx_base = 11.0
        if i == idx_max_n:
            # Rechts am Rand: Beschriftung links vom Punkt (bleibt im Plot)
            return -14.0, 4.0, "right", "center"
        if i == idx_min_h:
            # Unten im Plot: Beschriftung leicht oberhalb des Punkts
            return dx_base, 10.0, "left", "bottom"
        if i in mid_band_labeled:
            k = mid_band_labeled.index(i)
            return dx_base, -42.0 + k * 20.0, "left", "center"
        if i in small_n_high:
            # Fanny Lewald (n=5) und Lassalle (n=9): beide rechts-unten, vertikal gestaffelt
            sk = small_n_high.index(i)
            if sk == 0:
                return 11.0, -14.0, "left", "top"
            return 11.0, -40.0, "left", "top"
        if i in small_n_low:
            # Magnin, de Custine, Quinet: unten links, horizontal/vertikal getrennt
            sk = small_n_low.index(i)
            if sk == 0:
                # Charles Magnin (niedrigstes H bei n≤10): rechts-unten
                return 11.0, -22.0, "left", "top"
            if sk == 1:
                # Astolphe de Custine: rechts-oben
                return 11.0, 14.0, "left", "bottom"
            # Edgar Quinet (dicht bei de Custine): rechts, leicht unterhalb
            return 11.0, -8.0, "left", "center"
        return dx_base, 0.0, "left", "center"

    # Y-Limits mit wenig Rand; Beschriftungen bleiben im Koordinatenkreuz
    _hmin, _hmax = min(H), max(H)
    _hr = max(_hmax - _hmin, 0.15)
    ax3.set_ylim(_hmin - 0.03 * _hr, _hmax + 0.07 * _hr)

    for i in anno_idx:
        ax3.scatter(
            [n_briefe[i]],
            [H[i]],
            s=48,
            color="#E45756" if i in highlight_red else "#4C78A8",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        label = f"{partners[i]} (n={n_briefe[i]}, H={H[i]:.3f})"
        dx, dy, ha, va = _label_offset(i)
        ax3.annotate(
            label,
            xy=(n_briefe[i], H[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=scatter_label_fs,
            ha=ha,
            va=va,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.45,
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#9CA3AF",
                "lw": 0.38,
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    save_fig(out_dir / "plot_shannon_scatter_nbriefe", fig3)
    plt.close(fig3)

    # heatmap of category profiles, sorted by H descending
    # Datenadaptive Skala (2./98. Perzentil): nutzt die volle Colormap für den tatsächlichen
    # Wertebereich und macht kleine Unterschiede sichtbar (statt alles in einem mittleren Grün).
    # Alle Namen: Einrückung per X-Position in Achsenanteilen (zuverlässiger als Leerzeichen/Monospace).
    # Höhere Figur + größere Schrift, damit 98 Zeilen lesbar bleiben.
    fig4, ax4 = plt.subplots(figsize=(12.5, 15.5))
    sorted_mat = [mat[i] for i in order]
    sorted_names = [partners[i] for i in order]
    arr = np.asarray(sorted_mat, dtype=float)
    flat = arr.ravel()
    p_lo, p_hi = 2.0, 98.0
    vmin = float(np.percentile(flat, p_lo))
    vmax = float(np.percentile(flat, p_hi))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(flat.min()), float(flat.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax4.imshow(
        sorted_mat,
        aspect="auto",
        interpolation="nearest",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax4.set_title(
        "Kategorienprofile (modellbasierte Mittelwerte pro Korrespondent)\n"
        "sortiert nach absteigendem Shannon H",
        fontsize=11,
    )
    ax4.set_xlabel(
        "Farbskala „turbo“: kalt/blau = niedriges Gewicht, warm/rot-gelb = hohes Gewicht "
        f"(Normierung: Perzentile {p_lo:.0f}–{p_hi:.0f} der Matrix → {vmin:.3f} bis {vmax:.3f})",
        fontsize=9,
        labelpad=8,
    )
    ax4.set_xticks(range(len(TARGET_COLS)))
    ax4.set_xticklabels(TARGET_COLS, rotation=45, ha="right", fontsize=9)
    n_rows = len(sorted_names)
    ax4.set_yticks(range(n_rows))
    ax4.set_yticklabels([])
    ax4.tick_params(axis="y", length=0)
    # Dreifach zyklisch: rechter Textanker wandert nach links (sichtbare Stufen).
    y_label_trans = mtransforms.blended_transform_factory(ax4.transAxes, ax4.transData)
    name_fontsize = 10
    stagger = (0.0, 0.030, 0.060)  # Achsenanteile: drei sichtbare Stufen (geometrisch, nicht Leerzeichen)
    for i, name in enumerate(sorted_names):
        x_axes = -0.008 - stagger[i % 3]
        ax4.text(
            x_axes,
            i,
            name,
            transform=y_label_trans,
            va="center",
            ha="right",
            fontsize=name_fontsize,
            clip_on=False,
        )
    n_ticks = 9
    cbar_ticks = np.linspace(vmin, vmax, n_ticks)
    cbar = fig4.colorbar(im, ax=ax4, fraction=0.03, pad=0.02, ticks=cbar_ticks)
    cbar.set_label(
        "Gewicht (Anteil im Profil, 0–1; Summe der zehn Kategorien = 1)",
        fontsize=9,
    )
    fig4.tight_layout()
    # Platz links für gestaffelte Namen (negative x in Achsenanteilen).
    fig4.subplots_adjust(left=0.40)
    save_fig(out_dir / "plot_category_heatmap", fig4)
    plt.close(fig4)

    # Beispielwerte: drei Profile – höchstes H, Median, niedrigstes H
    ex_idx = [order[0], order[len(order) // 2], order[-1]]
    ex_labels = ["höchstes H (oberste Zeile)", "Median nach H", "niedrigstes H (unterste Zeile)"]
    ncols = 4 + len(TARGET_COLS)
    sep_row = "|" + "|".join(["---"] * ncols) + "|\n"
    beispiel_lines = [
        "## Beispielwerte zu Abbildung „Kategorienprofile“ (Heatmap)\n\n",
        "Modellbasierte Mittelwerte pro Korrespondent; die zehn Spalten summieren jeweils zu 1.\n\n",
        "| Rang | Partner | n Briefe | H | "
        + " | ".join(TARGET_COLS)
        + " |\n",
        sep_row,
    ]
    for lab, idx in zip(ex_labels, ex_idx):
        r = rows[idx]
        cells = [lab, partners[idx], str(n_briefe[idx]), f"{H[idx]:.4f}"]
        cells += [f"{float((r.get(k) or '0').strip() or '0'):.4f}" for k in TARGET_COLS]
        beispiel_lines.append("| " + " | ".join(cells) + " |\n")
    beispiel_lines.append(
        "\n**Lesart / Farbskala:** Die Heatmap nutzt die Colormap *turbo* mit **datenadaptiver** Normierung "
        f"(Perzentile {p_lo:.0f} bis {p_hi:.0f} über alle Matrixzellen: {vmin:.3f} bis {vmax:.3f}), "
        "damit feine Unterschiede sichtbar werden. **FAMILIAL** und **CONFLICT** sind z. B. typischerweise "
        "~0,19–0,26 bzw. ~0,00–0,03 – die Farbe spiegelt den Anteil im Profil, nicht die „Seltenheit“ der Kategorie "
        "im Gesamtkorpus.\n"
    )
    (out_dir / "plot_category_heatmap_beispielwerte.md").write_text("".join(beispiel_lines), encoding="utf-8")

    print(f"Wrote: {out_dir / 'shannon_summary.md'}")
    print(f"Wrote: {out_dir / 'shannon_top_bottom.csv'}")
    print(f"Wrote: {out_dir / 'plot_category_heatmap_beispielwerte.md'}")
    print(f"Wrote plots: plot_shannon_hist, plot_shannon_top_bottom, plot_shannon_scatter_nbriefe, plot_category_heatmap (.png/.pdf)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

