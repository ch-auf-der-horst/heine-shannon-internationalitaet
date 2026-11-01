#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and visualize category distribution shifts (P_k) across strata.

Reads:
  - Auswertungen/global_shannon_top100_by_phase_direction.csv

Produces:
  - Auswertungen/global_pk_differences_top100.csv
  - Auswertungen/global_pk_differences_top100.md
  - Plots (png+pdf):
      plot_pk_heatmap_brief.png/.pdf
      plot_pk_heatmap_partner.png/.pdf
      plot_pk_deltas_time_brief.png/.pdf
      plot_pk_deltas_time_partner.png/.pdf
      plot_pk_deltas_dir_brief.png/.pdf
      plot_pk_deltas_dir_partner.png/.pdf

Core idea:
  - P_k are global category distributions per stratum (phase x direction),
    available as P_<CAT>_brief and P_<CAT>_partner in the input CSV.
  - We compute delta profiles:
      * Time deltas per direction: EARLY->MIDDLE, MIDDLE->LATE, EARLY->LATE
      * Direction deltas per phase: AN - VON for each phase
  - For each delta we report values and top positive/negative shifts.
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

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

PHASES: Tuple[str, ...] = ("EARLY", "MIDDLE", "LATE")
DIRECTIONS: Tuple[str, ...] = ("VON_HEINE", "AN_HEINE")

def load_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)
    return rows

def ffloat(s: str) -> float:
    return float((s or "0").strip() or "0")

def key(phase: str, direction: str) -> str:
    return f"{phase}|{direction}"

def extract_pk(
    by_key: Dict[str, Dict[str, str]],
    phase: str,
    direction: str,
    *,
    scheme: str,  # "brief" or "partner"
) -> List[float]:
    row = by_key[key(phase, direction)]
    suf = "_brief" if scheme == "brief" else "_partner"
    return [ffloat(row.get(f"P_{c}{suf}", "0")) for c in TARGET_COLS]

def delta(a: List[float], b: List[float]) -> List[float]:
    return [bb - aa for aa, bb in zip(a, b)]

def save_fig(base: Path, fig) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

def heatmap(
    mat: List[List[float]],
    row_labels: List[str],
    col_labels: List[str],
    *,
    title: str,
    out_base: Path,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Anteil / ΔAnteil")
    save_fig(out_base, fig)
    plt.close(fig)

def bar_deltas(
    deltas: Dict[str, List[float]],
    *,
    title: str,
    out_base: Path,
) -> None:
    # One subplot per delta profile
    n = len(deltas)
    fig_h = max(3.5, 1.5 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, dvec) in zip(axes, deltas.items()):
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.bar(range(len(TARGET_COLS)), dvec, color=["#2a6fbb" if v >= 0 else "#d14b4b" for v in dvec])
        ax.set_ylabel(label, rotation=0, labelpad=40, va="center")
    axes[-1].set_xticks(range(len(TARGET_COLS)))
    axes[-1].set_xticklabels(TARGET_COLS, rotation=45, ha="right")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_fig(out_base, fig)
    plt.close(fig)

def top_shifts(dvec: List[float], k: int = 3) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    items = list(zip(TARGET_COLS, dvec))
    pos = sorted(items, key=lambda t: t[1], reverse=True)[:k]
    neg = sorted(items, key=lambda t: t[1])[:k]
    return pos, neg

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
        "--in_csv",
        default=str(_DATA / "global_shannon_top100_by_phase_direction.csv"),
    )
    ap.add_argument(
        "--out_csv",
        default=str(_DATA / "global_pk_differences_top100.csv"),
    )
    ap.add_argument(
        "--out_md",
        default=str(_DATA / "global_pk_differences_top100.md"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(_DATA),
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.in_csv))
    by: Dict[str, Dict[str, str]] = {}
    for r in rows:
        ph = (r.get("phase") or "").strip()
        di = (r.get("richtung") or "").strip()
        if ph in PHASES and di in DIRECTIONS:
            by[key(ph, di)] = r
    missing = [key(p, d) for p in PHASES for d in DIRECTIONS if key(p, d) not in by]
    if missing:
        raise SystemExit(f"Missing strata rows: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")

    # ---- Heatmaps of P_k per stratum ----
    row_labels = [f"{p} | {d}" for p in PHASES for d in DIRECTIONS]
    mat_brief = [extract_pk(by, p, d, scheme="brief") for p in PHASES for d in DIRECTIONS]
    mat_partner = [extract_pk(by, p, d, scheme="partner") for p in PHASES for d in DIRECTIONS]

    heatmap(
        mat_brief,
        row_labels,
        list(TARGET_COLS),
        title="Globale Kategorienanteile P_k (brief‑gewichtet) nach Phase × Richtung",
        out_base=out_dir / "plot_pk_heatmap_brief",
        cmap="viridis",
        vmin=0.0,
        vmax=max(max(r) for r in mat_brief) if mat_brief else None,
    )
    heatmap(
        mat_partner,
        row_labels,
        list(TARGET_COLS),
        title="Globale Kategorienanteile P_k (partner‑gewichtet) nach Phase × Richtung",
        out_base=out_dir / "plot_pk_heatmap_partner",
        cmap="viridis",
        vmin=0.0,
        vmax=max(max(r) for r in mat_partner) if mat_partner else None,
    )

    # ---- Delta profiles ----
    # time deltas per direction
    time_pairs = [("EARLY", "MIDDLE"), ("MIDDLE", "LATE"), ("EARLY", "LATE")]
    delta_time_brief: Dict[str, List[float]] = {}
    delta_time_partner: Dict[str, List[float]] = {}
    for d in DIRECTIONS:
        for a, b in time_pairs:
            lab = f"{d}: {a}→{b}"
            da_b = delta(extract_pk(by, a, d, scheme="brief"), extract_pk(by, b, d, scheme="brief"))
            da_p = delta(extract_pk(by, a, d, scheme="partner"), extract_pk(by, b, d, scheme="partner"))
            delta_time_brief[lab] = da_b
            delta_time_partner[lab] = da_p

    bar_deltas(
        delta_time_brief,
        title="ΔP_k über die Zeit (brief‑gewichtet): Phase‑Verschiebungen je Richtung",
        out_base=out_dir / "plot_pk_deltas_time_brief",
    )
    bar_deltas(
        delta_time_partner,
        title="ΔP_k über die Zeit (partner‑gewichtet): Phase‑Verschiebungen je Richtung",
        out_base=out_dir / "plot_pk_deltas_time_partner",
    )

    # direction deltas per phase: AN - VON
    delta_dir_brief: Dict[str, List[float]] = {}
    delta_dir_partner: Dict[str, List[float]] = {}
    for p in PHASES:
        lab = f"{p}: AN−VON"
        d_b = delta(extract_pk(by, p, "VON_HEINE", scheme="brief"), extract_pk(by, p, "AN_HEINE", scheme="brief"))
        d_p = delta(extract_pk(by, p, "VON_HEINE", scheme="partner"), extract_pk(by, p, "AN_HEINE", scheme="partner"))
        delta_dir_brief[lab] = d_b
        delta_dir_partner[lab] = d_p

    bar_deltas(
        delta_dir_brief,
        title="ΔP_k nach Richtung (brief‑gewichtet): AN_HEINE minus VON_HEINE (je Phase)",
        out_base=out_dir / "plot_pk_deltas_dir_brief",
    )
    bar_deltas(
        delta_dir_partner,
        title="ΔP_k nach Richtung (partner‑gewichtet): AN_HEINE minus VON_HEINE (je Phase)",
        out_base=out_dir / "plot_pk_deltas_dir_partner",
    )

    # ---- Write delta CSV (long format) ----
    out_rows: List[Dict[str, str]] = []

    def emit_block(kind: str, scheme: str, label: str, dvec: List[float]) -> None:
        row: Dict[str, str] = {"kind": kind, "scheme": scheme, "comparison": label}
        for c, v in zip(TARGET_COLS, dvec):
            row[f"delta_{c}"] = f"{v:+.6f}"
        pos, neg = top_shifts(dvec, k=3)
        row["top_positive"] = ", ".join([f"{c} {v:+.4f}" for c, v in pos])
        row["top_negative"] = ", ".join([f"{c} {v:+.4f}" for c, v in neg])
        out_rows.append(row)

    for lab, dvec in delta_time_brief.items():
        emit_block("time", "brief", lab, dvec)
    for lab, dvec in delta_time_partner.items():
        emit_block("time", "partner", lab, dvec)
    for lab, dvec in delta_dir_brief.items():
        emit_block("direction", "brief", lab, dvec)
    for lab, dvec in delta_dir_partner.items():
        emit_block("direction", "partner", lab, dvec)

    fieldnames = ["kind", "scheme", "comparison"] + [f"delta_{c}" for c in TARGET_COLS] + ["top_positive", "top_negative"]
    write_csv(Path(args.out_csv), out_rows, fieldnames)

    # ---- Write interpretive markdown ----
    md: List[str] = []
    md.append("## Differenzprofile der Kategorienanteile (P_k) (TOP‑100)\n\n")
    md.append(
        "Diese Auswertung geht einen Schritt über den Shannon‑Index hinaus: Statt nur die **Gleichmäßigkeit** der Verteilung (Entropie) zu betrachten, werden die **globalen Kategorienanteile** P_k pro Stratum (Phase × Richtung) direkt miteinander verglichen.\n\n"
    )
    md.append("- **Input**: `global_shannon_top100_by_phase_direction.csv`\n")
    md.append("- **Output (Deltas)**: `global_pk_differences_top100.csv`\n\n")
    md.append("### Was ist P_k?\n\n")
    md.append(
        "Für jedes Stratum liegt eine globale Verteilung über die zehn Kategorien vor. In der CSV heißen diese Anteile `P_<KATEGORIE>_brief` bzw. `P_<KATEGORIE>_partner`.\n\n"
        "- **brief‑gewichtet**: jeder Brief zählt gleich (Ereignis‑Perspektive)\n"
        "- **partner‑gewichtet**: erst pro Partner mitteln, dann Partner gleichgewichten (Kanten‑Perspektive)\n\n"
    )
    md.append("### Abbildungen (Übersicht)\n\n")
    md.append("![Heatmap P_k (brief)](plot_pk_heatmap_brief.png)\n\n")
    md.append("![Heatmap P_k (partner)](plot_pk_heatmap_partner.png)\n\n")
    md.append("![ΔP_k Zeit (brief)](plot_pk_deltas_time_brief.png)\n\n")
    md.append("![ΔP_k Zeit (partner)](plot_pk_deltas_time_partner.png)\n\n")
    md.append("![ΔP_k Richtung (brief)](plot_pk_deltas_dir_brief.png)\n\n")
    md.append("![ΔP_k Richtung (partner)](plot_pk_deltas_dir_partner.png)\n\n")

    def md_delta_table(title: str, blocks: Dict[str, List[float]]) -> None:
        md.append(f"### {title}\n\n")
        md.append("| Vergleich | Top + (größte Zunahmen) | Top − (größte Abnahmen) |\n")
        md.append("|---|---|---|\n")
        for lab, dvec in blocks.items():
            pos, neg = top_shifts(dvec, k=3)
            md.append(
                f"| {lab} | "
                + ", ".join([f"**{c}** {v:+.4f}" for c, v in pos])
                + " | "
                + ", ".join([f"**{c}** {v:+.4f}" for c, v in neg])
                + " |\n"
            )
        md.append("\n")

    md_delta_table("Zeitliche Verschiebungen (ΔP_k) – brief‑gewichtet", delta_time_brief)
    md_delta_table("Zeitliche Verschiebungen (ΔP_k) – partner‑gewichtet", delta_time_partner)
    md_delta_table("Richtungsunterschiede (ΔP_k = AN − VON) – brief‑gewichtet", delta_dir_brief)
    md_delta_table("Richtungsunterschiede (ΔP_k = AN − VON) – partner‑gewichtet", delta_dir_partner)

    md.append("### Wie man das literaturwissenschaftlich liest\n\n")
    md.append(
        "Die Deltas (ΔP_k) sind **inhaltliche Verschiebungen** im globalen Funktionsmix. Sie beantworten Fragen wie: „Nimmt der Anteil editoraler Kommunikation im späten Briefwechsel zu?“, „Ist die eingehende Korrespondenz (AN Heine) in einer Phase stärker transaktional als die ausgehende (VON Heine)?“.\n\n"
        "Besonders aussagekräftig sind Effekte, die **in beiden Gewichtungsschemata** (brief‑ und partner‑gewichtet) in dieselbe Richtung zeigen: Dann sind sie weniger wahrscheinlich allein durch einzelne Vielschreiber‑Beziehungen getrieben.\n"
    )

    Path(args.out_md).write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")
    print("Wrote plots: plot_pk_heatmap_brief/partner, plot_pk_deltas_time_*, plot_pk_deltas_dir_* (.png/.pdf)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

