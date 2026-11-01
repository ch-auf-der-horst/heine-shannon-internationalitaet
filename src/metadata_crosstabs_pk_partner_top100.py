#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partner-gewichtete Metadaten-Kreuztabellen und P_k-Profile (TOP-100-Korpus).

Pro Korrespondent genau ein Eintrag: Mittel der Funktionsgewichte über alle Briefe
in brief_predictions_top100.csv (= partner-gewichtet; jeder Partner zählt gleich).

Outputs (Unterordner Auswertungen/metadata_partner_top100/):
  - partner_stufe_pk_meta.csv — eine Zeile pro Partner mit Mittel-P_k + Metadaten
  - Kreuztabellen je Paar (Nationalität, Beziehungstyp, Beruf, Juedische_Herkunft):
      *_counts.csv, *_row_norm.csv, *_col_norm.csv
  - pk_mittel_pro_<Dimension>.csv — mittleres P_k je Ausprägung
  - pk_kategoriemasse_anteil_pro_<Dimension>.csv — Anteil der Kategoriensumme je Gruppe
  - metadata_partner_top100_report.md
  - PNG/PDF: Heatmaps der Kreuztabellen, P_k-Heatmaps nach Dimension
  - PNG/PDF: Gestapelte Balken (row_norm / col_norm) je Kreuztabelle, ohne „(nicht im Codebook)“

Kein pandas (nur csv + numpy + matplotlib).
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

META_KEYS = ("Nationalität", "Beziehungstyp", "Beruf", "Juedische_Herkunft")

# gesetzt, wenn partner_vollname nach norm_name nicht im Codebook-Index vorkommt
NOT_IN_CODEBOOK = "(nicht im Codebook)"

def label_fuer_achse_und_legende(dim: str) -> str:
    """Anzeigenamen für Diagramm-Achsen und Legenden (nicht für CSV-Spaltennamen)."""
    if dim == "Beruf":
        return "Berufe/Rollen"
    return dim

CROSSTAB_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("Nationalität", "Beziehungstyp", "Nat_Bez"),
    ("Nationalität", "Beruf", "Nat_Beruf"),
    ("Nationalität", "Juedische_Herkunft", "Nat_JH"),
    ("Beziehungstyp", "Beruf", "Bez_Beruf"),
    ("Beziehungstyp", "Juedische_Herkunft", "Bez_JH"),
    ("Beruf", "Juedische_Herkunft", "Beruf_JH"),
)

# Wie im ursprünglichen Plot: matplotlib tab20 (inkl. Grautöne), dezente Wissenschaftsgrafik.
_TAB20 = plt.colormaps["tab20"]

# Wie im Referenzplot (tab20-ähnlich, grau-dominiert): feste Farben je Berufsrolle.
BERUF_ROLE_COLORS: Dict[str, str] = {
    "Adeliger/Emigrant": "#1f77b4",
    "Adeliger/Politiker": "#aec7e8",
    "Arzt": "#ff7f0e",
    "Bankangestellter/Intellektueller": "#ffbb78",
    "Bankier": "#ff7f0e",
    "Beamter": "#98df8a",
    "Gesellschaft": "#2ca02c",
    "Gesellschaft/Salonnière": "#9467bd",
    "Industrieller": "#8c564b",
    "Jurist/Beamter": "#9467bd",
    "Jurist/Politiker": "#c5b0d5",
    "Kaufmann": "#8c564b",
    "Komponist": "#c49c94",
    "Philosoph": "#c5b0d5",
    "Publizist": "#ff9896",
    "Salonnière/Gesellschaft": "#7f7f7f",
    "Schriftsteller": "#c7c7c7",
    "Sekretär": "#bcbd22",
    "Staatsmann": "#dbdb8d",
    "Theologe/Wissenschaftler": "#17becf",
    "Verleger": "#9edae5",
    "eng verwandt": "#1f77b4",
    "verwandt": "#aec7e8",
    "Übersetzer": "#ffbb78",
}

# Reihenfolge der Berufsrollen wie in der Kreuztabelle (alphabetisch sortiert).
BERUF_ROLE_ORDER: Tuple[str, ...] = tuple(sorted(BERUF_ROLE_COLORS.keys()))

def beruf_role_color(label: str, idx: int) -> str:
    return BERUF_ROLE_COLORS.get(label, categorical_color(idx))

def reorder_beruf_columns(
    mat: np.ndarray, col_labels: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Spalten in der Referenz-Aufschlüsselung (alphabetische Berufsrollen)."""
    if mat.size == 0 or not col_labels:
        return mat, col_labels
    order = sorted(
        range(len(col_labels)),
        key=lambda j: (
            BERUF_ROLE_ORDER.index(col_labels[j])
            if col_labels[j] in BERUF_ROLE_ORDER
            else len(BERUF_ROLE_ORDER),
            col_labels[j],
        ),
    )
    return mat[:, order], [col_labels[j] for j in order]

def categorical_color(idx: int) -> str:
    return _TAB20((idx % 20) / 20.0)

def norm_name(s: str) -> str:
    if not s:
        return ""
    return s.replace("\xa0", " ").replace("  ", " ").strip()

def meta_val(s: str) -> str:
    t = (s or "").strip()
    return t if t else "(fehlend)"

def save_fig(base: Path, fig) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def load_codebook(path: Path) -> Dict[str, Dict[str, str]]:
    metadata_cols = [
        "Kategorie 1: Beziehungstyp",
        "Unterkategorie 2: Beruf/Rolle",
        "Nationalität",
        "Juedische_Herkunft",
    ]
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = norm_name(row.get("Vollname", ""))
            if not name:
                continue
            out[name] = {
                "Beziehungstyp": meta_val(row.get("Kategorie 1: Beziehungstyp", "")),
                "Beruf": meta_val(row.get("Unterkategorie 2: Beruf/Rolle", "")),
                "Nationalität": meta_val(row.get("Nationalität", "")),
                "Juedische_Herkunft": meta_val(row.get("Juedische_Herkunft", "")),
            }
    return out

def aggregate_partner_vectors(pred_csv: Path) -> Dict[str, Tuple[np.ndarray, int]]:
    """partner -> (Summenvektor, n_briefe)."""
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = defaultdict(int)
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            partner = norm_name((row.get("partner_vollname") or "").strip('"'))
            if not partner:
                continue
            if partner not in sums:
                sums[partner] = np.zeros(len(TARGET_COLS), dtype=float)
            vec = sums[partner]
            counts[partner] += 1
            for i, col in enumerate(TARGET_COLS):
                try:
                    vec[i] += float((row.get(col) or "0").strip() or "0")
                except ValueError:
                    continue
    out: Dict[str, Tuple[np.ndarray, int]] = {}
    for p, svec in sums.items():
        n = counts[p]
        out[p] = (svec / float(n), n)
    return out

def build_partner_table(
    partner_pk: Dict[str, Tuple[np.ndarray, int]],
    codebook: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for partner in sorted(partner_pk.keys()):
        mean_vec, n_b = partner_pk[partner]
        meta = codebook.get(partner)
        if meta is None:
            meta = {k: NOT_IN_CODEBOOK for k in META_KEYS}
        row: Dict[str, object] = {
            "partner_vollname": partner,
            "n_briefe": n_b,
        }
        for i, col in enumerate(TARGET_COLS):
            row[f"pk_mean_{col}"] = float(mean_vec[i])
        row.update(meta)
        rows.append(row)
    return rows

def write_partner_csv(path: Path, table: List[Dict[str, object]]) -> None:
    fieldnames = ["partner_vollname", "n_briefe"] + [f"pk_mean_{c}" for c in TARGET_COLS] + list(META_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for r in table:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def crosstab_matrix(
    table: List[Dict[str, object]], key_row: str, key_col: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    levels_r: set = set()
    levels_c: set = set()
    for r in table:
        levels_r.add(str(r.get(key_row, "(fehlend)")))
        levels_c.add(str(r.get(key_col, "(fehlend)")))
    rows_l = sorted(levels_r)
    cols_l = sorted(levels_c)
    ri = {x: i for i, x in enumerate(rows_l)}
    cj = {x: j for j, x in enumerate(cols_l)}
    mat = np.zeros((len(rows_l), len(cols_l)), dtype=float)
    for r in table:
        i = ri[str(r.get(key_row, "(fehlend)"))]
        j = cj[str(r.get(key_col, "(fehlend)"))]
        mat[i, j] += 1.0
    return mat, rows_l, cols_l

def crosstab_matrix_brief_weighted(
    table: List[Dict[str, object]], key_row: str, key_col: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Wie crosstab_matrix, gewichtet mit n_briefe statt 1 pro Partner."""
    levels_r: set = set()
    levels_c: set = set()
    for r in table:
        levels_r.add(str(r.get(key_row, "(fehlend)")))
        levels_c.add(str(r.get(key_col, "(fehlend)")))
    rows_l = sorted(levels_r)
    cols_l = sorted(levels_c)
    ri = {x: i for i, x in enumerate(rows_l)}
    cj = {x: j for j, x in enumerate(cols_l)}
    mat = np.zeros((len(rows_l), len(cols_l)), dtype=float)
    for r in table:
        i = ri[str(r.get(key_row, "(fehlend)"))]
        j = cj[str(r.get(key_col, "(fehlend)"))]
        mat[i, j] += float(r.get("n_briefe", 0) or 0)
    return mat, rows_l, cols_l

def drop_not_in_codebook_from_counts(
    mat: np.ndarray,
    rows_l: List[str],
    cols_l: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Entfernt Zeilen/Spalten mit Label „(nicht im Codebook)“ aus der Zählmatrix
    und liefert die reduzierte Matrix (für Balkendiagramme ohne diese Rubrik).
    """
    keep_r = [i for i, r in enumerate(rows_l) if r != NOT_IN_CODEBOOK]
    keep_c = [j for j, c in enumerate(cols_l) if c != NOT_IN_CODEBOOK]
    if not keep_r or not keep_c:
        return np.zeros((0, 0)), [], []
    sub = mat[np.ix_(keep_r, keep_c)].copy()
    return sub, [rows_l[i] for i in keep_r], [cols_l[j] for j in keep_c]

def row_normalize(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        s = mat[i].sum()
        out[i] = mat[i] / s if s > 0 else 0.0
    return out

def col_normalize(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        s = mat[:, j].sum()
        out[:, j] = mat[:, j] / s if s > 0 else 0.0
    return out

def write_matrix_csv(
    path: Path,
    mat: np.ndarray,
    rows: List[str],
    cols: List[str],
    row_label: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([row_label] + cols)
        for i, rn in enumerate(rows):
            w.writerow([rn] + [f"{mat[i, j]:.8f}" for j in range(len(cols))])

def write_matrix_csv_int(
    path: Path,
    mat: np.ndarray,
    rows: List[str],
    cols: List[str],
    row_label: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([row_label] + cols)
        for i, rn in enumerate(rows):
            w.writerow([rn] + [str(int(mat[i, j])) for j in range(len(cols))])

def pk_means_by_dimension(
    table: List[Dict[str, object]], dim: str
) -> Tuple[np.ndarray, List[str]]:
    """Returns matrix (n_levels, n_categories) and level names."""
    levels = sorted({str(r.get(dim, "(fehlend)")) for r in table})
    acc: Dict[str, List[np.ndarray]] = {lv: [] for lv in levels}
    for r in table:
        lv = str(r.get(dim, "(fehlend)"))
        vec = np.array([float(r[f"pk_mean_{c}"]) for c in TARGET_COLS], dtype=float)
        acc[lv].append(vec)
    mat = np.zeros((len(levels), len(TARGET_COLS)), dtype=float)
    for i, lv in enumerate(levels):
        arrs = acc[lv]
        if arrs:
            mat[i] = np.mean(np.stack(arrs, axis=0), axis=0)
    return mat, levels

def category_mass_share(
    table: List[Dict[str, object]], dim: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Pro Funktionskategorie k: Anteil der Summe der mittleren Gewichte über Partner
    in Gruppe g an der Gesamtsumme über alle Partner (jeder Partner gleich gewichtet).
    Shape: (n_categories, n_levels) — Zeilen = TARGET_COLS, Spalten = Ausprägungen.
    """
    levels = sorted({str(r.get(dim, "(fehlend)")) for r in table})
    sum_w: Dict[str, np.ndarray] = {lv: np.zeros(len(TARGET_COLS), dtype=float) for lv in levels}
    for r in table:
        lv = str(r.get(dim, "(fehlend)"))
        for i, c in enumerate(TARGET_COLS):
            sum_w[lv][i] += float(r[f"pk_mean_{c}"])
    total = np.zeros(len(TARGET_COLS), dtype=float)
    for lv in levels:
        total += sum_w[lv]
    mat = np.zeros((len(TARGET_COLS), len(levels)), dtype=float)
    for j, lv in enumerate(levels):
        for i in range(len(TARGET_COLS)):
            mat[i, j] = sum_w[lv][i] / total[i] if total[i] > 1e-15 else 0.0
    return mat, levels

def plot_crosstab_heatmap(
    out_base: Path,
    mat: np.ndarray,
    rows: List[str],
    cols: List[str],
    title: str,
    annotate_counts: bool = True,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(cols) + 3), max(5.0, 0.35 * len(rows) + 2)), constrained_layout=True)
    if mat.size == 0:
        ax.text(0.5, 0.5, "leer", ha="center", va="center")
        save_fig(out_base, fig)
        return
    if vmax is None:
        vmax = max(float(mat.max()), 1.0)
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate_counts:
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.0f}" if v >= 1 else f"{v:.2f}", ha="center", va="center", fontsize=6, color="black")
    elif vmax == 1.0:
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat[i, j]
                if v > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="black")
    save_fig(out_base, fig)

def plot_crosstab_stacked_row_norm(
    out_base: Path,
    mat_row_norm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    row_dim_name: str,
    col_dim_name: str,
) -> None:
    """Horizontale 100 %-Balken: pro Zeile (row_dim) Anteile über Spaltenkategorien."""
    if mat_row_norm.size == 0 or not row_labels or not col_labels:
        fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
        ax.text(0.5, 0.5, "keine Daten", ha="center", va="center")
        save_fig(out_base, fig)
        return
    use_beruf_style = col_dim_name == "Beruf"
    if use_beruf_style:
        mat_row_norm, col_labels = reorder_beruf_columns(mat_row_norm, col_labels)
        title = "Zeilenanteile: Nationalität → Beruf"
    nrows, ncols = mat_row_norm.shape
    fig_h = max(4.5, 0.38 * nrows + 1.8)
    fig_w = max(9.0, 0.12 * ncols + 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    y = np.arange(nrows, dtype=float)
    left = np.zeros(nrows, dtype=float)
    for j in range(ncols):
        w = mat_row_norm[:, j].astype(float)
        if use_beruf_style:
            color = beruf_role_color(col_labels[j], j)
        else:
            color = categorical_color(j)
        ax.barh(y, w, left=left, height=0.72, label=col_labels[j], color=color, edgecolor="white", linewidth=0.4)
        left = left + w
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8)
    if use_beruf_style:
        ax.set_xlabel("Anteil (je Nationalität summiert zu 1)", fontsize=9)
    else:
        ax.set_xlabel(
            "Anteil (je „{}“ summiert zu 1)".format(label_fuer_achse_und_legende(row_dim_name)),
            fontsize=9,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, 1.0)
    if use_beruf_style:
        ax.xaxis.grid(True, color="#e6e6e6", linestyle="-", linewidth=0.6)
        ax.set_axisbelow(True)
    ax.legend(
        title=label_fuer_achse_und_legende(col_dim_name),
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )
    save_fig(out_base, fig)

def _draw_beruf_stacked_row_norm(
    ax,
    mat_row_norm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    *,
    title: str,
    show_ylabel: bool,
) -> None:
    y = np.arange(len(row_labels), dtype=float)
    left = np.zeros(len(row_labels), dtype=float)
    for j in range(len(col_labels)):
        w = mat_row_norm[:, j].astype(float)
        color = beruf_role_color(col_labels[j], j)
        ax.barh(
            y,
            w,
            left=left,
            height=0.72,
            label=col_labels[j],
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        left = left + w
    ax.set_yticks(y)
    if show_ylabel:
        ax.set_yticklabels(row_labels, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Anteil (je Nationalität summiert zu 1)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.grid(True, color="#e6e6e6", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

def plot_nat_beruf_partner_vs_brief_sidebyside(
    out_base: Path,
    mat_partner: np.ndarray,
    mat_brief: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
) -> None:
    """Zwei gestapelte Zeilenanteile nebeneinander: Partner- vs. Brief-gewichtung."""
    if mat_partner.size == 0 or mat_brief.size == 0:
        return
    rn_p = row_normalize(mat_partner)
    rn_b = row_normalize(mat_brief)
    rn_p, cols = reorder_beruf_columns(rn_p, col_labels)
    rn_b, _ = reorder_beruf_columns(rn_b, col_labels)

    nrows = len(row_labels)
    fig_h = max(5.0, 0.38 * nrows + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(17.0, fig_h))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    _draw_beruf_stacked_row_norm(
        axes[0],
        rn_p,
        row_labels,
        cols,
        title="Partner-gewichtet\n(jeder Partner = 1)",
        show_ylabel=True,
    )
    _draw_beruf_stacked_row_norm(
        axes[1],
        rn_b,
        row_labels,
        cols,
        title="Brief-gewichtet\n(Summe der Briefe)",
        show_ylabel=False,
    )

    fig.suptitle(
        "Nationalität → Beruf: Partner- vs. Brief-gewichtete Zeilenanteile (TOP-100)",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        title="Berufe/Rollen",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        frameon=True,
    )
    fig.subplots_adjust(left=0.20, right=0.70, top=0.90, wspace=0.12)
    save_fig(out_base, fig)

def plot_crosstab_stacked_col_norm(
    out_base: Path,
    mat_col_norm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    row_dim_name: str,
    col_dim_name: str,
) -> None:
    """Horizontale 100 %-Balken: pro Spalte (col_dim) Anteile über Zeilenkategorien."""
    if mat_col_norm.size == 0 or not row_labels or not col_labels:
        fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
        ax.text(0.5, 0.5, "keine Daten", ha="center", va="center")
        save_fig(out_base, fig)
        return
    nrows, ncols = mat_col_norm.shape
    fig_h = max(4.5, 0.38 * ncols + 1.8)
    fig_w = max(9.0, 0.12 * nrows + 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    y = np.arange(ncols, dtype=float)
    left = np.zeros(ncols, dtype=float)
    for i in range(nrows):
        w = mat_col_norm[i, :].astype(float)
        color = categorical_color(i)
        ax.barh(y, w, left=left, height=0.72, label=row_labels[i], color=color, edgecolor="white", linewidth=0.4)
        left = left + w
    ax.set_yticks(y)
    ax.set_yticklabels(col_labels, fontsize=8)
    ax.set_xlabel(
        "Anteil (je „{}“ summiert zu 1)".format(label_fuer_achse_und_legende(col_dim_name)),
        fontsize=9,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.legend(
        title=label_fuer_achse_und_legende(row_dim_name),
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )
    save_fig(out_base, fig)

def plot_nat_beruf_de_fr_comparison(
    out_base: Path,
    mat_row_norm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    *,
    min_share: float = 0.025,
    max_roles: int = 14,
) -> None:
    """
    Gruppierte Balken: deutsch vs. französisch für die wichtigsten Berufsrollen.
    Lesbarere Alternative zur vollständigen gestapelten Darstellung (Abb. 16).
    """
    if mat_row_norm.size == 0:
        return
    try:
        i_de = row_labels.index("deutsch")
        i_fr = row_labels.index("französisch")
    except ValueError:
        return

    de = mat_row_norm[i_de, :].astype(float)
    fr = mat_row_norm[i_fr, :].astype(float)
    combined = de + fr
    candidates = [
        j
        for j in range(len(col_labels))
        if combined[j] > 0 and max(de[j], fr[j]) >= min_share
    ]
    if not candidates:
        candidates = list(np.argsort(-combined))[:max_roles]
    else:
        candidates = sorted(candidates, key=lambda j: (-combined[j], col_labels[j]))[:max_roles]

    roles = [col_labels[j] for j in candidates]
    de_vals = [de[j] * 100 for j in candidates]
    fr_vals = [fr[j] * 100 for j in candidates]

    n = len(roles)
    fig_h = max(5.5, 0.42 * n + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), constrained_layout=True)
    y = np.arange(n, dtype=float)
    bar_h = 0.34
    ax.barh(y - bar_h / 2, de_vals, height=bar_h, label="deutsch", color="#1f77b4", alpha=0.92)
    ax.barh(y + bar_h / 2, fr_vals, height=bar_h, label="französisch", color="#ff9896", alpha=0.92)

    for yi, (dv, fv) in enumerate(zip(de_vals, fr_vals)):
        if dv >= 4:
            ax.text(dv + 0.4, yi - bar_h / 2, f"{dv:.0f}%", va="center", fontsize=7, color="#333333")
        if fv >= 4:
            ax.text(fv + 0.4, yi + bar_h / 2, f"{fv:.0f}%", va="center", fontsize=7, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(roles, fontsize=9)
    ax.set_xlabel("Anteil innerhalb der Nationalitätsgruppe (%)", fontsize=9)
    ax.set_title(
        "Berufs- und Rollenprofile: deutsch vs. französisch\n(TOP-100, partner-gewichtet)",
        fontsize=10,
    )
    xmax = max(max(de_vals + fr_vals, default=0), 5) * 1.18
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save_fig(out_base, fig)

def plot_nat_beruf_compact_heatmap(
    out_base: Path,
    mat_row_norm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
) -> None:
    """Kompakte Heatmap nur für Rollen mit Anteil ≥ 3 % in deutsch oder französisch."""
    if mat_row_norm.size == 0:
        return
    focus_rows = [r for r in ("deutsch", "französisch") if r in row_labels]
    if not focus_rows:
        return
    row_idx = [row_labels.index(r) for r in focus_rows]
    sub = mat_row_norm[row_idx, :].astype(float)
    keep = [j for j in range(sub.shape[1]) if sub[:, j].max() >= 0.03]
    if not keep:
        keep = list(np.argsort(-sub.sum(axis=0)))[:12]
    sub = sub[:, keep]
    cols = [col_labels[j] for j in keep]
    totals = sub.sum(axis=0)
    order = sorted(range(len(cols)), key=lambda j: (-totals[j], cols[j]))
    sub = sub[:, order]
    cols = [cols[j] for j in order]

    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(cols) + 2), 3.8), constrained_layout=True)
    im = ax.imshow(sub, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.25, float(sub.max())))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(focus_rows)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(focus_rows, fontsize=9)
    ax.set_title("Rollenanteile: deutsch und französisch (Anteil ≥ 3 %)", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Zeilenanteil", fontsize=8)
    for i in range(len(focus_rows)):
        for j in range(len(cols)):
            v = sub[i, j]
            if v >= 0.02:
                ax.text(
                    j,
                    i,
                    f"{v * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > 0.14 else "#333333",
                )
    save_fig(out_base, fig)

def plot_pk_heatmap(
    out_base: Path,
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cbar_label: str,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(9.0, 0.42 * len(col_labels) + 2), max(5.5, 0.32 * len(row_labels) + 2)), constrained_layout=True)
    vm = vmax if vmax is not None else max(float(mat.max()), 0.05)
    im = ax.imshow(mat, aspect="auto", cmap="turbo", vmin=0.0, vmax=vm)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = mat[i, j]
            if not math.isnan(v) and v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > vm * 0.55 else "black")
    save_fig(out_base, fig)

def main() -> None:
    ap = argparse.ArgumentParser(description="Metadaten-Kreuztabellen und P_k partner-gewichtet TOP-100")
    ap.add_argument(
        "--pred_csv",
        type=Path,
        default=_DATA / "brief_predictions_top100.csv",
    )
    ap.add_argument(
        "--codebook",
        type=Path,
        default=_DATA / "codebook_ready_3.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_DATA / "metadata_partner_top100",
    )
    ap.add_argument(
        "--nur-mit-codebook",
        action="store_true",
        help=(
            "Nur Partner mit exaktem Vollnamen-Treffer im Codebook auswerten; "
            "sonst erscheint die Kategorie „(nicht im Codebook)“ in Kreuztabellen und P_k."
        ),
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook = load_codebook(args.codebook)
    partner_pk = aggregate_partner_vectors(args.pred_csv)
    table = build_partner_table(partner_pk, codebook)

    unmatched = [r for r in table if r.get("Nationalität") == NOT_IN_CODEBOOK]
    pu = out_dir / "partners_nicht_im_codebook.csv"
    if unmatched:
        with pu.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["partner_vollname", "n_briefe"], delimiter=";")
            w.writeheader()
            for r in sorted(unmatched, key=lambda x: str(x.get("partner_vollname", ""))):
                w.writerow({"partner_vollname": r.get("partner_vollname", ""), "n_briefe": r.get("n_briefe", "")})
    elif pu.exists():
        pu.unlink()
    if args.nur_mit_codebook:
        table = [r for r in table if r.get("Nationalität") != NOT_IN_CODEBOOK]
        if not table:
            raise SystemExit("--nur-mit-codebook: nach Filter keine Partner übrig.")

    write_partner_csv(out_dir / "partner_stufe_pk_meta.csv", table)

    plt.style.use("seaborn-v0_8-whitegrid")

    report: List[str] = []
    report.append("# Metadaten-Kreuztabellen und P_k (partner-gewichtet, TOP-100)\n\n")
    report.append(f"- **Input Vorhersagen:** `{args.pred_csv.name}`\n")
    report.append(f"- **Input Codebook:** `{args.codebook.name}`\n")
    report.append("- **Aggregation:** Pro `partner_vollname` Mittel der zehn Funktionsgewichte über alle Briefe; jeder Partner zählt einmal.\n")
    report.append(f"- **Anzahl Partner (Zeilen):** {len(table)}\n")
    if args.nur_mit_codebook:
        report.append("- **Filter:** `--nur-mit-codebook` — ohne Partner ohne Codebook-Match.\n")
    report.append(
        f"- **`(nicht im Codebook)`:** Partner in den Vorhersagen ohne exakten `Vollname`-Treffer; siehe `partners_nicht_im_codebook.csv` (falls vorhanden).\n\n"
    )

    # --- Kreuztabellen ---
    report.append("## Kreuztabellen (Anzahl Partner pro Zelle)\n\n")
    for key_r, key_c, short in CROSSTAB_PAIRS:
        mat, rows_l, cols_l = crosstab_matrix(table, key_r, key_c)
        base = out_dir / f"crosstab_{short}"
        write_matrix_csv_int(base.parent / f"{base.name}_counts.csv", mat, rows_l, cols_l, key_r)
        rn = row_normalize(mat)
        cn = col_normalize(mat)
        write_matrix_csv(base.parent / f"{base.name}_row_norm.csv", rn, rows_l, cols_l, key_r)
        write_matrix_csv(base.parent / f"{base.name}_col_norm.csv", cn, rows_l, cols_l, key_r)

        title = f"Partner je ({key_r} × {key_c}) — Anzahlen"
        plot_crosstab_heatmap(out_dir / f"plot_crosstab_{short}_counts", mat, rows_l, cols_l, title, annotate_counts=True)
        plot_crosstab_heatmap(
            out_dir / f"plot_crosstab_{short}_row_norm",
            rn,
            rows_l,
            cols_l,
            f"Zeilenanteile: {key_r} → {key_c}",
            annotate_counts=False,
            vmax=1.0,
        )
        plot_crosstab_heatmap(
            out_dir / f"plot_crosstab_{short}_col_norm",
            cn,
            rows_l,
            cols_l,
            f"Spaltenanteile: {key_c} → {key_r}",
            annotate_counts=False,
            vmax=1.0,
        )

        mat_fc, rows_fc, cols_fc = drop_not_in_codebook_from_counts(mat, rows_l, cols_l)
        if mat_fc.size > 0 and mat_fc.sum() > 0:
            rn_fc = row_normalize(mat_fc)
            cn_fc = col_normalize(mat_fc)
            plot_crosstab_stacked_row_norm(
                out_dir / f"plot_crosstab_{short}_row_norm_stacked",
                rn_fc,
                rows_fc,
                cols_fc,
                f"Zeilenanteile: {key_r} → {key_c}",
                key_r,
                key_c,
            )
            plot_crosstab_stacked_col_norm(
                out_dir / f"plot_crosstab_{short}_col_norm_stacked",
                cn_fc,
                rows_fc,
                cols_fc,
                f"Spaltenanteile: {key_c} → {key_r}",
                key_r,
                key_c,
            )
            if short == "Nat_Beruf":
                table_cb = [r for r in table if r.get("Nationalität") != NOT_IN_CODEBOOK]
                mat_brief, rows_b, cols_b = crosstab_matrix_brief_weighted(
                    table_cb, key_r, key_c
                )
                mat_bfc, rows_bfc, cols_bfc = drop_not_in_codebook_from_counts(
                    mat_brief, rows_b, cols_b
                )
                write_matrix_csv_int(
                    out_dir / "crosstab_Nat_Beruf_counts_briefe.csv",
                    mat_brief,
                    rows_b,
                    cols_b,
                    key_r,
                )
                write_matrix_csv(
                    out_dir / "crosstab_Nat_Beruf_row_norm_briefe.csv",
                    row_normalize(mat_brief),
                    rows_b,
                    cols_b,
                    key_r,
                )
                if mat_bfc.size > 0 and mat_bfc.sum() > 0:
                    plot_nat_beruf_partner_vs_brief_sidebyside(
                        out_dir / "plot_nat_beruf_partner_vs_brief_sidebyside",
                        mat_fc,
                        mat_bfc,
                        rows_fc,
                        cols_fc,
                    )
                plot_nat_beruf_de_fr_comparison(
                    out_dir / "plot_abb16_nat_beruf_de_fr_comparison",
                    rn_fc,
                    rows_fc,
                    cols_fc,
                )
                plot_nat_beruf_compact_heatmap(
                    out_dir / "plot_abb16_nat_beruf_compact_heatmap",
                    rn_fc,
                    rows_fc,
                    cols_fc,
                )

        report.append(f"### {short}\n\n")
        report.append(f"- `crosstab_{short}_counts.csv`, `_row_norm.csv`, `_col_norm.csv`\n")
        report.append(
            f"- `plot_crosstab_{short}_counts.png/.pdf`, `_row_norm`, `_col_norm`; "
            f"**gestapelt (ohne „{NOT_IN_CODEBOOK}“):** `_row_norm_stacked`, `_col_norm_stacked`\n"
        )
        if short == "Nat_Beruf":
            report.append(
                "- **Partner vs. Brief:** `plot_nat_beruf_partner_vs_brief_sidebyside` (Nebeneinander), "
                "`crosstab_Nat_Beruf_counts_briefe.csv`, `_row_norm_briefe.csv`\n"
            )
            report.append(
                "- **Abb. 16 Alternativen:** `plot_abb16_nat_beruf_de_fr_comparison` (gruppiert, deutsch vs. französisch), "
                "`plot_abb16_nat_beruf_compact_heatmap` (kompakte Heatmap)\n"
            )
        report.append("\n")

    # --- P_k Mittel und Massenanteile pro Dimension (ohne Partner nur in Vorhersagen) ---
    table_pk = [r for r in table if r.get("Nationalität") != NOT_IN_CODEBOOK]
    report.append("## Mittleres P_k pro Metadimension\n\n")
    report.append(
        f"- P_k-Grafiken und zugehörige CSV nutzen nur Partner **mit** Codebook-Zeile "
        f"(ohne „{NOT_IN_CODEBOOK}“).\n\n"
    )
    for dim in META_KEYS:
        mat, levels = pk_means_by_dimension(table_pk, dim)
        path = out_dir / f"pk_mittel_pro_{dim.replace('/', '_')}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([dim] + list(TARGET_COLS))
            for i, lv in enumerate(levels):
                w.writerow([lv] + [f"{mat[i, j]:.8f}" for j in range(len(TARGET_COLS))])
        plot_pk_heatmap(
            out_dir / f"plot_pk_mittel_{dim.replace('/', '_')}",
            mat,
            levels,
            list(TARGET_COLS),
            f"Mittleres P_k (partner-gewichtet) nach {dim}",
            "Mittleres Gewicht",
            vmax=0.35,
        )
        report.append(f"- `{path.name}` — `plot_pk_mittel_{dim.replace('/', '_')}.png/.pdf`\n")

        mass, levs = category_mass_share(table_pk, dim)
        path_m = out_dir / f"pk_kategoriemasse_anteil_pro_{dim.replace('/', '_')}.csv"
        with path_m.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Funktionskategorie"] + levs)
            for i, cat in enumerate(TARGET_COLS):
                w.writerow([cat] + [f"{mass[i, j]:.8f}" for j in range(len(levs))])
        plot_pk_heatmap(
            out_dir / f"plot_pk_kategoriemasse_anteil_{dim.replace('/', '_')}",
            mass,
            list(TARGET_COLS),
            levs,
            f"Anteil der Kategoriensumme je {dim}\n(Zeile = Kategorie; Spalte = Gruppe; Summe je Zeile = 1)",
            "Anteil",
            vmax=1.0,
        )
        report.append(f"- `{path_m.name}` — `plot_pk_kategoriemasse_anteil_{dim.replace('/', '_')}.png/.pdf`\n")
    report.append("\n")

    report.append("## Hinweise\n\n")
    report.append(
        "- `(fehlend)` = leeres Codebook-Feld; `(nicht im Codebook)` = kein Zeilentreffer für `partner_vollname` unter `Vollname` (Tippfehler/abweichende Schreibweise). "
        "Wiederholung ohne diese Zeile: Skript mit `--nur-mit-codebook` ausführen.\n"
    )
    report.append("- **Kategoriemasse:** Für jede Funktion k ist der Anteil pro Gruppe: Summe der mittleren Gewichte über Partner in g / Summe über alle Partner (jeder Partner gleich).\n")

    (out_dir / "metadata_partner_top100_report.md").write_text("".join(report), encoding="utf-8")

    print(f"Wrote outputs under: {out_dir}")
    if unmatched:
        print(f"Partner ohne Codebook-Match: {len(unmatched)} (siehe partners_nicht_im_codebook.csv)")
    if args.nur_mit_codebook:
        print("Filter aktiv: nur Partner mit Codebook-Zeile.")

if __name__ == "__main__":
    main()
