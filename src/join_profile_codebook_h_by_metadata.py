#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Join korrespondenten_profile_pred_top100.csv with codebook_ready_3.csv (metadata only)
for H-by-Nationalität / Beziehungstyp / Beruf / Juedische_Herkunft analyses.

Uses only metadata from codebook (Beziehungstyp, Beruf, Nationalität, Juedische_Herkunft).
Category weights and shannon_H come exclusively from the profile (ML pipeline).

Outputs:
  - Auswertungen/shannon_H_by_metadata.csv (joined table)
  - Auswertungen/shannon_H_by_metadata_report.md (summary statistics)
"""

from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, Optional

def norm_name(s: str) -> str:
    """Normalize name: replace non-breaking spaces, collapse multiple spaces."""
    if not s:
        return ""
    return s.replace("\xa0", " ").replace("  ", " ").strip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Join profile + codebook for H-by-metadata analysis")
    parser.add_argument(
        "--profile",
        type=Path,
        default=_DATA / "korrespondenten_profile_pred_top100.csv",
        help="Profile CSV (shannon_H, etc.)",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=_DATA / "codebook_ready_3.csv",
        help="Codebook CSV (metadata only: Beziehungstyp, Beruf, Nationalität, Juedische_Herkunft)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_DATA / "shannon_H_by_metadata.csv",
        help="Output joined CSV",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=_DATA / "shannon_H_by_metadata_report.md",
        help="Output report MD",
    )
    args = parser.parse_args()

    # Load profile
    profile_rows: Dict[str, dict] = {}
    with open(args.profile, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = norm_name(row.get("partner_vollname", "").strip('"'))
            profile_rows[name] = row

    # Load codebook (metadata only)
    metadata_cols = [
        "Kategorie 1: Beziehungstyp",
        "Unterkategorie 2: Beruf/Rolle",
        "Nationalität",
        "Juedische_Herkunft",
    ]
    codebook_rows: Dict[str, dict] = {}
    with open(args.codebook, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = norm_name(row.get("Vollname", ""))
            # Extract only metadata; ignore INTELLECTUAL etc. (flawed procedure)
            codebook_rows[name] = {k: row.get(k, "") for k in metadata_cols}

    # Join
    joined: list[dict] = []
    matched_names: set[str] = set()
    for pname, prow in profile_rows.items():
        cb_row = codebook_rows.get(pname)
        if cb_row is None:
            continue
        matched_names.add(pname)
        # Build joined row: profile fields + metadata
        n_briefe = prow.get("n_briefe", "").strip('"')
        try:
            shannon_H = float(prow.get("shannon_H", "").strip('"'))
        except (ValueError, TypeError):
            shannon_H = ""
        evenness_E = prow.get("evenness_E", "").strip('"')
        joined.append({
            "partner_vollname": pname,
            "n_briefe": n_briefe,
            "shannon_H": shannon_H if isinstance(shannon_H, float) else "",
            "evenness_E": evenness_E,
            "Beziehungstyp": cb_row.get("Kategorie 1: Beziehungstyp", ""),
            "Beruf": cb_row.get("Unterkategorie 2: Beruf/Rolle", ""),
            "Nationalität": cb_row.get("Nationalität", ""),
            "Juedische_Herkunft": cb_row.get("Juedische_Herkunft", ""),
        })

    # Write joined CSV
    out_cols = [
        "partner_vollname", "n_briefe", "shannon_H", "evenness_E",
        "Beziehungstyp", "Beruf", "Nationalität", "Juedische_Herkunft",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, delimiter=";")
        writer.writeheader()
        writer.writerows(joined)

    # Summary statistics for report
    with_h = [r for r in joined if isinstance(r["shannon_H"], (int, float))]
    if not with_h:
        report_lines = ["## Keine gültigen H-Werte für Report.\n"]
    else:
        by_nat: Dict[str, list[float]] = defaultdict(list)
        by_beziehung: Dict[str, list[float]] = defaultdict(list)
        by_juedisch: Dict[str, list[float]] = defaultdict(list)
        for r in with_h:
            h = r["shannon_H"]
            if r["Nationalität"]:
                by_nat[r["Nationalität"]].append(h)
            if r["Beziehungstyp"]:
                by_beziehung[r["Beziehungstyp"]].append(h)
            if r["Juedische_Herkunft"]:
                by_juedisch[r["Juedische_Herkunft"]].append(h)

        def stats(arr: list[float]) -> str:
            if not arr:
                return "—"
            n = len(arr)
            mean_h = sum(arr) / n
            return f"n={n}, H_mean={mean_h:.4f}"

        report_lines = [
            "# Shannon H by Metadata – Auswertung",
            "",
            f"**Dateien:**",
            f"- Profil: `{args.profile.name}`",
            f"- Codebook: `{args.codebook.name}` (nur Metadaten)",
            "",
            f"**Gematchte Korrespondenten:** {len(joined)} von {len(profile_rows)} im Profil",
            "",
            "## H nach Nationalität",
            "",
            "| Nationalität | n | H_mean |",
            "|--------------|---|--------|",
        ]
        for nat in sorted(by_nat.keys()):
            arr = by_nat[nat]
            mean_h = sum(arr) / len(arr)
            report_lines.append(f"| {nat} | {len(arr)} | {mean_h:.4f} |")

        report_lines.extend([
            "",
            "## H nach Beziehungstyp",
            "",
            "| Beziehungstyp | n | H_mean |",
            "|---------------|---|--------|",
        ])
        for typ in sorted(by_beziehung.keys()):
            arr = by_beziehung[typ]
            mean_h = sum(arr) / len(arr)
            report_lines.append(f"| {typ} | {len(arr)} | {mean_h:.4f} |")

        report_lines.extend([
            "",
            "## H nach Jüdischer Herkunft",
            "",
            "| Juedische_Herkunft | n | H_mean |",
            "|--------------------|---|--------|",
        ])
        for j in sorted(by_juedisch.keys()):
            arr = by_juedisch[j]
            mean_h = sum(arr) / len(arr)
            report_lines.append(f"| {j} | {len(arr)} | {mean_h:.4f} |")

        report_lines.extend([
            "",
            "## Nicht gematchte Profil-Namen (ohne Codebook-Metadaten)",
            "",
        ])
        unmatched = sorted(set(profile_rows.keys()) - matched_names)
        if unmatched:
            for u in unmatched:
                report_lines.append(f"- {u}")
        else:
            report_lines.append("- (alle gematcht)")

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Joined: {len(joined)} Korrespondenten")
    print(f"✓ Output: {args.out_csv}")
    print(f"✓ Report: {args.out_report}")

if __name__ == "__main__":
    main()
