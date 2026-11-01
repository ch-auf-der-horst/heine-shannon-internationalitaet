#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a labeled training dataset by joining manually coded letters with the
Heine master CSV via letter_id.

Design goals:
- No external dependencies (no pandas needed).
- Robust to delimiter differences in manual CSVs (';' vs ',').
- Produces one combined CSV suitable for later ML steps.

Default paths are set to your project layout, but can be overridden via CLI.
"""

from __future__ import annotations
import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
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

MASTER_KEEP_COLS: Tuple[str, ...] = (
    "letterid",
    "sprache",
    "datum_iso",
    "datum_gephi",
    "absender_vollname",
    "absender_name",
    "adressat_vollname",
    "adressat_name",
    "brieftext",
)

DEFAULT_MAPPING_CSV = str(_LOCAL / "brief_id_seiten_mapping.csv")

def normalize_unicode_ws(s: str) -> str:
    # Normalize NBSP etc. to plain spaces and collapse whitespace.
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    return " ".join(s.split())

def sniff_delimiter(path: Path, fallback: str = ";") -> str:
    sample = path.read_text(encoding="utf-8", errors="replace")[:20_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        return fallback

def parse_float_maybe(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    # handle "7.2." or decimal comma
    if x.endswith(".") and x.count(".") >= 2:
        # e.g. "7.2." is likely a data issue; keep as None to be safe
        x = x.strip(".")
    x = x.replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None

def extract_year_from_manual_date(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    # look for 4-digit year 18xx/19xx
    digits = []
    for i in range(len(s) - 3):
        chunk = s[i : i + 4]
        if chunk.isdigit():
            y = int(chunk)
            if 1800 <= y <= 1900:
                return y
    return None

def year_fits_phase(year: int, phase: str) -> bool:
    phase = (phase or "").strip().upper()
    if not phase:
        return True
    # Use your project convention (seen in validate scripts):
    # EARLY: 1816-1831, MIDDLE: 1832-1848, LATE: 1849-1856
    if phase == "EARLY":
        return 1812 <= year <= 1831
    if phase == "MIDDLE":
        return 1832 <= year <= 1848
    if phase == "LATE":
        return 1849 <= year <= 1856
    return True

@dataclass(frozen=True)
class ManualRow:
    source_file: str
    letter_id: str
    raw: Dict[str, str]

def read_manual_csv(path: Path) -> List[ManualRow]:
    delim = sniff_delimiter(path)
    rows: List[ManualRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            return rows
        # Normalize fieldnames (strip spaces)
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        # Re-wrap reader to apply normalized fieldnames
        reader.fieldnames = fieldnames

        for r in reader:
            # common keys: letter_id / letterid
            letter_id = (r.get("letter_id") or r.get("letterid") or "").strip()
            if not letter_id:
                continue
            letter_id = normalize_unicode_ws(letter_id)
            rows.append(ManualRow(source_file=path.name, letter_id=letter_id, raw=r))
    return rows

def load_master_index(master_csv: Path) -> Dict[str, Dict[str, str]]:
    # Increase CSV field size limit for long texts.
    csv.field_size_limit(sys.maxsize)
    idx: Dict[str, Dict[str, str]] = {}
    with master_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            letterid = (r.get("letterid") or "").strip()
            if not letterid:
                continue
            idx[normalize_unicode_ws(letterid)] = r
    return idx

def load_lokal_to_master_mapping(mapping_csv: Path) -> Dict[str, List[str]]:
    """
    Loads mapping from XML-local ids (e.g. L247) to master ids (e.g. W20B0247).
    Returns: { "L247": ["W20B0247", "W24B0247", ...], ... }
    """
    csv.field_size_limit(sys.maxsize)
    out: Dict[str, List[str]] = {}
    with mapping_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            master_id = normalize_unicode_ws((r.get("Brief_ID") or "").strip())
            lokal_id = normalize_unicode_ws((r.get("Brief_ID_Lokal") or "").strip())
            if not master_id or not lokal_id:
                continue
            out.setdefault(lokal_id, []).append(master_id)
    return out

def infer_partner_hint_from_filename(filename: str) -> str:
    """
    Infer a (very rough) partner hint from a manual CSV file name.
    Example: 'julius_campe_kodierung_manuell_....csv' -> 'julius campe'
             'j_homberg_co_1_kodierung_manuell_....csv' -> 'j homberg co'
    """
    base = filename
    if base.endswith(".csv"):
        base = base[:-4]
    base = base.split("_kodierung_manuell_")[0]
    # strip a trailing counter like _1
    parts = base.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    hint = " ".join(parts)
    hint = normalize_unicode_ws(hint).lower()
    return hint

def infer_phase_and_year_from_manual_row(mr: ManualRow) -> Tuple[Optional[str], Optional[int]]:
    phase = (mr.raw.get("phase") or mr.raw.get("PHASE") or "").strip()
    year = extract_year_from_manual_date(mr.raw.get("date") or mr.raw.get("datum") or "")
    return (phase or None, year)

def master_year(m: Dict[str, str]) -> Optional[int]:
    d = (m.get("datum_iso") or "").strip()
    if len(d) >= 4 and d[:4].isdigit():
        return int(d[:4])
    y = (m.get("jahr") or "").strip()
    if y.isdigit():
        return int(y)
    return None

def score_candidate(
    *,
    master_row: Dict[str, str],
    partner_hint: str,
    manual_phase: Optional[str],
    manual_year: Optional[int],
) -> int:
    score = 0
    if partner_hint:
        a = normalize_unicode_ws((master_row.get("absender_vollname") or "")).lower()
        b = normalize_unicode_ws((master_row.get("adressat_vollname") or "")).lower()
        if partner_hint in a or partner_hint in b:
            score += 10
    y = master_year(master_row)
    if y is not None:
        if manual_year is not None and y == manual_year:
            score += 4
        if manual_phase and year_fits_phase(y, manual_phase):
            score += 3
    return score

def search_master_by_nr(
    master_idx: Dict[str, Dict[str, str]],
    nr_value: int,
) -> List[str]:
    hits: List[str] = []
    for mid, m in master_idx.items():
        nr = (m.get("nr") or "").strip()
        if nr.isdigit() and int(nr) == nr_value:
            hits.append(mid)
    return hits

def expand_candidates_for_lokal_id(
    letter_id: str,
    master_idx: Dict[str, Dict[str, str]],
    lokal_to_master: Dict[str, List[str]],
) -> List[str]:
    """
    Sammelt alle Master-IDs zu einer XML-L-ID (z. B. L2 -> W20B0002, W24B0002, W29B0002).
    Die Mapping-CSV ist oft unvollständig (fehlende Nachtragsbände).
    """
    cands = list(lokal_to_master.get(letter_id, []))
    match = re.match(r"L(\d+)([a-z])?", letter_id, re.I)
    if match:
        num = int(match.group(1))
        suffix = (match.group(2) or "").upper()
        for mid in master_idx:
            mm = re.match(r"W(\d+)B(\d+)([A-Z])?", mid)
            if not mm or int(mm.group(2)) != num:
                continue
            if (mm.group(3) or "").upper() == suffix and mid not in cands:
                cands.append(mid)
    if not cands and letter_id.startswith("L") and letter_id[1:].isdigit():
        cands = search_master_by_nr(master_idx, int(letter_id[1:]))
    return cands

def pick_master_id_for_manual_row(
    mr: ManualRow,
    master_idx: Dict[str, Dict[str, str]],
    lokal_to_master: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (master_letterid, warning_or_none).
    Handles two cases:
    - mr.letter_id is already a master id (starts with 'W')
    - mr.letter_id is a lokal xml id (starts with 'L') and needs mapping
    Prefer master ids (W20B0251) in manual CSVs — L-ids collide across bands.
    """
    letter_id = mr.letter_id
    if letter_id.startswith("W"):
        return (letter_id, None) if letter_id in master_idx else (None, f"Missing in master: {letter_id}")

    cands = expand_candidates_for_lokal_id(letter_id, master_idx, lokal_to_master)
    if not cands:
        return None, f"Missing lokal->master mapping for {letter_id} (from {mr.source_file})"

    # If only one candidate, accept it.
    if len(cands) == 1:
        return cands[0], None

    # Disambiguate using partner hint from filename + phase/year hints.
    hint = infer_partner_hint_from_filename(mr.source_file)
    manual_phase, manual_year = infer_phase_and_year_from_manual_row(mr)

    scored: List[Tuple[int, str]] = []
    for mid in cands:
        m = master_idx.get(mid)
        if not m:
            continue
        scored.append(
            (
                score_candidate(
                    master_row=m,
                    partner_hint=hint,
                    manual_phase=manual_phase,
                    manual_year=manual_year,
                ),
                mid,
            )
        )
    scored.sort(reverse=True)
    if not scored:
        return None, f"Ambiguous mapping for {letter_id}: {cands} (from {mr.source_file})"
    best_score, best_id = scored[0]
    # Require some evidence if ambiguous; if scores tie, reject.
    if len(scored) > 1 and scored[1][0] == best_score:
        return None, f"Ambiguous mapping for {letter_id}: {cands} (hint='{hint}', phase='{manual_phase}', year='{manual_year}')"
    if best_score == 0:
        return None, f"Ambiguous mapping for {letter_id}: {cands} (no hint matched)"
    return best_id, None

def build_training_rows(
    manual_rows: Sequence[ManualRow],
    master_idx: Dict[str, Dict[str, str]],
    lokal_to_master: Dict[str, List[str]],
    *,
    sum_tolerance: float = 0.02,
    normalize_targets: bool = True,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Returns (rows, warnings).
    Each output row contains:
      - letter_id
      - source_file
      - 10 target columns (as strings)
      - selected master columns (brieftext + key metadata)
      - qa flags
    """
    out_rows: List[Dict[str, str]] = []
    warns: List[str] = []

    for mr in manual_rows:
        master_id, w = pick_master_id_for_manual_row(mr, master_idx, lokal_to_master)
        if w:
            warns.append(w)
        if not master_id:
            continue
        m = master_idx.get(master_id)
        if m is None:
            warns.append(f"Missing in master after mapping: {mr.letter_id} -> {master_id} (from {mr.source_file})")
            continue

        targets: Dict[str, Optional[float]] = {k: parse_float_maybe(mr.raw.get(k, "")) for k in TARGET_COLS}
        missing_targets = [k for k, v in targets.items() if v is None]
        if missing_targets:
            warns.append(f"Missing targets for {mr.letter_id} in {mr.source_file}: {', '.join(missing_targets)}")

        present_vals = [v for v in targets.values() if v is not None]
        s = sum(present_vals) if present_vals else 0.0

        qa_sum_ok = ""
        if present_vals:
            qa_sum_ok = "OK" if abs(s - 1.0) <= sum_tolerance else "NOT_OK"
        else:
            qa_sum_ok = "NO_TARGETS"

        if present_vals and normalize_targets:
            if s > 0 and abs(s - 1.0) > sum_tolerance:
                # Renormalize to sum 1 using only present targets.
                for k in TARGET_COLS:
                    if targets[k] is not None:
                        targets[k] = targets[k] / s
                warns.append(f"Renormalized targets for {mr.letter_id} (sum was {s:.4f})")

        row: Dict[str, str] = {
            "letter_id": mr.letter_id,
            "master_letterid": master_id,
            "source_file": mr.source_file,
            "qa_target_sum": f"{s:.6f}" if present_vals else "",
            "qa_target_sum_status": qa_sum_ok,
        }

        for k in TARGET_COLS:
            v = targets[k]
            row[k] = "" if v is None else f"{v:.10f}".rstrip("0").rstrip(".")

        for k in MASTER_KEEP_COLS:
            row[k] = m.get(k, "")

        out_rows.append(row)

    return out_rows, warns

def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            delimiter=";",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Join manual coded letters with master CSV to build ML training set.")
    parser.add_argument(
        "--master",
        default=str(_LOCAL / "alle_baende_MASTER.csv"),
        help="Path to master CSV (semicolon-delimited).",
    )
    parser.add_argument(
        "--manual_dir",
        default=str(_LOCAL / "manual_coding_results"),
        help="Directory containing *_kodierung_manuell_*.csv files.",
    )
    parser.add_argument(
        "--mapping",
        default=DEFAULT_MAPPING_CSV,
        help="Path to brief_id_seiten_mapping.csv (maps L-ids to W-ids).",
    )
    parser.add_argument(
        "--output",
        default=str(_DATA / "train_labeled_letters_weights_only.csv"),
        help="Output training CSV path.",
    )
    parser.add_argument(
        "--sum_tolerance",
        type=float,
        default=0.02,
        help="Tolerance for target sum check around 1.0.",
    )
    parser.add_argument(
        "--no_normalize_targets",
        action="store_true",
        help="Do not renormalize targets to sum=1 when outside tolerance.",
    )
    args = parser.parse_args(argv)

    master_csv = Path(args.master)
    manual_dir = Path(args.manual_dir)
    out_csv = Path(args.output)

    manual_files = sorted(manual_dir.glob("*_kodierung_manuell_*.csv"))
    if not manual_files:
        print(f"No manual files found in {manual_dir}", file=sys.stderr)
        return 2

    print(f"Master: {master_csv}")
    print(f"Manual dir: {manual_dir} ({len(manual_files)} files)")

    manual_rows: List[ManualRow] = []
    for p in manual_files:
        rs = read_manual_csv(p)
        manual_rows.extend(rs)
    print(f"Loaded manual rows: {len(manual_rows)}")

    master_idx = load_master_index(master_csv)
    print(f"Loaded master rows: {len(master_idx)}")

    mapping_csv = Path(args.mapping)
    lokal_to_master = load_lokal_to_master_mapping(mapping_csv)
    print(f"Loaded mapping rows: {sum(len(v) for v in lokal_to_master.values())} (unique lokal ids={len(lokal_to_master)})")

    rows, warns = build_training_rows(
        manual_rows,
        master_idx,
        lokal_to_master,
        sum_tolerance=args.sum_tolerance,
        normalize_targets=not args.no_normalize_targets,
    )

    fieldnames = (
        ["letter_id", "master_letterid", "source_file"]
        + list(TARGET_COLS)
        + ["qa_target_sum", "qa_target_sum_status"]
        + list(MASTER_KEEP_COLS)
    )
    write_csv(out_csv, rows, fieldnames)

    print(f"Wrote training set: {out_csv} (rows={len(rows)})")
    if warns:
        warn_path = out_csv.with_suffix(".warnings.txt")
        warn_path.write_text("\n".join(warns) + "\n", encoding="utf-8")
        print(f"Wrote warnings: {warn_path} (n={len(warns)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

