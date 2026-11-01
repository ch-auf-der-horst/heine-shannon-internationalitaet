#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korrespondenten-spezifische Shannon-H/Evenness-Werte pro Phase (EARLY/MIDDLE/LATE),
basierend auf brief_predictions_top100.csv, verknüpft mit Codebook-Metadaten.

Die Logik orientiert sich an:
- ml/global_shannon_by_phase_direction.py (Phasen-Definition)
- ml/join_profile_codebook_h_by_metadata.py (Metadaten-Join)

Output:
  - Auswertungen/shannon_H_by_metadata_phase.csv
      Eine Zeile pro (partner_vollname, phase) mit:
        partner_vollname, phase, richtung, n_briefe,
        shannon_H, evenness_E,
        Beziehungstyp, Beruf, Nationalität, Juedische_Herkunft
  - Auswertungen/shannon_H_by_metadata_phase_report.md
      Deskriptive Tabellen:
        H nach Phase × Nationalität
        H nach Phase × Beziehungstyp
        H nach Phase × Beruf
        H nach Phase × Juedische_Herkunft
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
from typing import Dict, List, Optional, Tuple

def norm_name(s: str) -> str:
    if not s:
        return ""
    return s.replace("\xa0", " ").replace("  ", " ").strip()

def parse_year_month(datum_iso: str) -> Optional[Tuple[int, int]]:
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
    Gleiche Phasen wie in global_shannon_by_phase_direction.py
      EARLY:  bis Mai 1831 (inkl.)
      MIDDLE: Jun 1831 bis Mai 1848 (inkl.)
      LATE:   Jun 1848 bis Feb 1856 (inkl.)
    """
    if y < 1831 or (y == 1831 and m <= 5):
        return "EARLY"
    if (y == 1831 and m >= 6) or (1832 <= y <= 1847) or (y == 1848 and m <= 5):
        return "MIDDLE"
    if (y == 1848 and m >= 6) or (1849 <= y <= 1855) or (y == 1856 and m <= 2):
        return "LATE"
    return None

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

def normalize_vec(v: List[float]) -> List[float]:
    s = sum(v)
    if s <= 0:
        return [1.0 / len(v)] * len(v)
    return [x / s for x in v]

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-spezifische H-Werte pro Korrespondent + Metadaten")
    ap.add_argument(
        "--pred_csv",
        type=Path,
        default=_DATA / "brief_predictions_top100.csv",
        help="Brief-basierte Vorhersagen mit Funktionsgewichten",
    )
    ap.add_argument(
        "--codebook",
        type=Path,
        default=_DATA / "codebook_ready_3.csv",
        help="Codebook mit Metadaten (Beziehungstyp, Beruf, Nationalität, Juedische_Herkunft)",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=_DATA / "shannon_H_by_metadata_phase.csv",
        help="Output CSV (Korrespondent × Phase × Richtung mit H und Metadaten)",
    )
    ap.add_argument(
        "--out_report",
        type=Path,
        default=_DATA / "shannon_H_by_metadata_phase_report.md",
        help="Output Report (Markdown)",
    )
    args = ap.parse_args()

    # Metadaten aus Codebook
    metadata_cols = [
        "Kategorie 1: Beziehungstyp",
        "Unterkategorie 2: Beruf/Rolle",
        "Nationalität",
        "Juedische_Herkunft",
    ]
    codebook_rows: Dict[str, Dict[str, str]] = {}
    with args.codebook.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = norm_name(row.get("Vollname", ""))
            if not name:
                continue
            codebook_rows[name] = {k: row.get(k, "") for k in metadata_cols}

    # Akkumulation: (partner, phase, richtung) -> Summenvektor + n_briefe
    sums: Dict[Tuple[str, str, str], List[float]] = defaultdict(lambda: [0.0] * len(TARGET_COLS))
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

    with args.pred_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            partner = norm_name((row.get("partner_vollname") or "").strip('"'))
            if not partner:
                continue

            ym = parse_year_month(row.get("datum_iso") or "")
            if not ym:
                continue
            phase = phase_for_year_month(ym[0], ym[1])
            if not phase:
                continue

            direction = (row.get("richtung") or "").strip() or "UNKNOWN"
            key = (partner, phase, direction)
            counts[key] += 1
            vec = sums[key]
            for i, col in enumerate(TARGET_COLS):
                try:
                    vec[i] += float((row.get(col) or "0").strip() or "0")
                except ValueError:
                    continue

    # Zeilen bauen: pro (partner, phase, richtung) H/E berechnen und Metadaten joinen
    out_rows: List[Dict[str, str]] = []
    for (partner, phase, direction), n_briefe in sorted(counts.items()):
        vec = sums[(partner, phase, direction)]
        mean_vec = [x / n_briefe for x in vec]
        probs = normalize_vec(mean_vec)
        H = shannon_index(probs, base=2.0)
        E = evenness(H, len(TARGET_COLS), base=2.0)

        meta = codebook_rows.get(partner, {})
        row_out: Dict[str, str] = {
            "partner_vollname": partner,
            "phase": phase,
            "richtung": direction,
            "n_briefe": str(n_briefe),
            "shannon_H": f"{H:.6f}",
            "evenness_E": f"{E:.6f}",
            "Beziehungstyp": meta.get("Kategorie 1: Beziehungstyp", ""),
            "Beruf": meta.get("Unterkategorie 2: Beruf/Rolle", ""),
            "Nationalität": meta.get("Nationalität", ""),
            "Juedische_Herkunft": meta.get("Juedische_Herkunft", ""),
        }
        out_rows.append(row_out)

    # CSV schreiben
    fieldnames = [
        "partner_vollname",
        "phase",
        "richtung",
        "n_briefe",
        "shannon_H",
        "evenness_E",
        "Beziehungstyp",
        "Beruf",
        "Nationalität",
        "Juedische_Herkunft",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(out_rows)

    # Report: H nach Phase × Metadaten
    # Nur Zeilen mit numerischem H verwenden
    def parse_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    rows_with_H = [r for r in out_rows if parse_float(r.get("shannon_H", "")) is not None]

    by_phase_nat: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_phase_bez: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_phase_beruf: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_phase_jh: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for r in rows_with_H:
        H_val = parse_float(r["shannon_H"])
        if H_val is None:
            continue
        ph = r["phase"]

        nat = r.get("Nationalität", "")
        if nat:
            by_phase_nat[(ph, nat)].append(H_val)

        bez = r.get("Beziehungstyp", "")
        if bez:
            by_phase_bez[(ph, bez)].append(H_val)

        beruf = r.get("Beruf", "")
        if beruf:
            by_phase_beruf[(ph, beruf)].append(H_val)

        jh = r.get("Juedische_Herkunft", "")
        if jh:
            by_phase_jh[(ph, jh)].append(H_val)

    def mean(arr: List[float]) -> float:
        return sum(arr) / len(arr) if arr else float("nan")

    lines: List[str] = []
    lines.append("# Phase-spezifischer Shannon H nach Metadaten\n\n")
    lines.append(f"- **Input** Vorhersagen: `{args.pred_csv.name}`\n")
    lines.append(f"- **Input** Codebook: `{args.codebook.name}` (Metadaten)\n")
    lines.append("- **Output** Tabelle: `shannon_H_by_metadata_phase.csv`\n\n")

    # Phase × Nationalität
    lines.append("## H nach Phase × Nationalität\n\n")
    lines.append("| Phase | Nationalität | n | H_mean |\n")
    lines.append("|---|---|---:|---:|\n")
    for ph in ("EARLY", "MIDDLE", "LATE"):
        keys = sorted([k for k in by_phase_nat.keys() if k[0] == ph], key=lambda x: x[1])
        for (phase, nat) in keys:
            arr = by_phase_nat[(phase, nat)]
            lines.append(f"| {phase} | {nat} | {len(arr)} | {mean(arr):.4f} |\n")
    lines.append("\n")

    # Phase × Beziehungstyp
    lines.append("## H nach Phase × Beziehungstyp\n\n")
    lines.append("| Phase | Beziehungstyp | n | H_mean |\n")
    lines.append("|---|---|---:|---:|\n")
    for ph in ("EARLY", "MIDDLE", "LATE"):
        keys = sorted([k for k in by_phase_bez.keys() if k[0] == ph], key=lambda x: x[1])
        for (phase, bez) in keys:
            arr = by_phase_bez[(phase, bez)]
            lines.append(f"| {phase} | {bez} | {len(arr)} | {mean(arr):.4f} |\n")
    lines.append("\n")

    # Phase × Beruf
    lines.append("## H nach Phase × Beruf/Rolle\n\n")
    lines.append("| Phase | Beruf/Rolle | n | H_mean |\n")
    lines.append("|---|---|---:|---:|\n")
    for ph in ("EARLY", "MIDDLE", "LATE"):
        keys = sorted([k for k in by_phase_beruf.keys() if k[0] == ph], key=lambda x: x[1])
        for (phase, beruf) in keys:
            arr = by_phase_beruf[(phase, beruf)]
            lines.append(f"| {phase} | {beruf} | {len(arr)} | {mean(arr):.4f} |\n")
    lines.append("\n")

    # Phase × Juedische_Herkunft
    lines.append("## H nach Phase × Juedischer Herkunft\n\n")
    lines.append("| Phase | Juedische_Herkunft | n | H_mean |\n")
    lines.append("|---|---|---:|---:|\n")
    for ph in ("EARLY", "MIDDLE", "LATE"):
        keys = sorted([k for k in by_phase_jh.keys() if k[0] == ph], key=lambda x: x[1])
        for (phase, jh) in keys:
            arr = by_phase_jh[(phase, jh)]
            lines.append(f"| {phase} | {jh} | {len(arr)} | {mean(arr):.4f} |\n")
    lines.append("\n")

    args.out_report.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_report}")

if __name__ == "__main__":
    main()

