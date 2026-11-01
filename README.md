# Heine correspondence: Shannon index and internationality

Reproduction materials for a study of **functional diversity** in Heinrich Heine’s letter network (TOP-100 correspondents, DHA vols. 20–38).

> **Important:** This repository does **not** contain letter full texts from the Düsseldorf Heine Edition. It provides pipeline code, aggregated metrics (category weights, Shannon *H*), a metadata codebook, and publication figures.

## What is measured

Letters and correspondences are represented as distributions over **ten relational function categories**. The **Shannon index** summarises these profiles as a measure of **functional breadth** (not letter volume). Internationality is treated as a **relational quality**: how broad or concentrated are partner profiles coded as German vs French in the codebook (phases EARLY / MIDDLE / LATE)?

## Corpus snapshot (manuscript)

| Quantity | Value |
|---|---|
| Stratum | TOP-100 partners, bidirectional |
| Partners | 96 (codebook complete) |
| Letters (inference) | ca. 2394 |
| Training annotation | 80 letters / 16 correspondences |
| Master file | local only (not in this repo) |

## Repository layout

- `src/` — Python pipeline (TF-IDF + Ridge → aggregation → Shannon → internationality plots)
- `data/` — aggregated CSVs, codebook, redacted training weights (no `brieftext`)
- `figures/` — figures (incl. nationality × phase)
- `docs/` — short method notes
- `local/` — placeholder for private edition files / model (not published)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Published **result snapshots** are under `data/` and `figures/`. Full re-inference needs a local master CSV (see `docs/REPRODUCTION.md`).

Path defaults in `src/` are **relative** to the repository root (`data/`, `local/`).

## Core pipeline

1. Training: `train_baseline_tfidf.py` *(requires local texts + writes local `.joblib`)*
2. Inference TOP-100: `infer_top100.py` *(local master + model)*
3. Partner profiles + Shannon: `aggregate_shannon.py`
4. Descriptives / plots: `analyze_and_visualize_shannon.py`
5. Phase × metadata: `shannon_by_metadata_phase.py`
6. Internationality figures: `plot_internationality_shannon.py`
7. Crosstabs: `metadata_crosstabs_pk_partner_top100.py`

## Rights / privacy

- **Code:** [license TBD, e.g. MIT]
- **Aggregated research data:** [e.g. CC-BY 4.0] — without edition full texts
- **DHA letter texts:** not included; rights remain with the edition
- **Trained `.joblib` model:** not included (may embed text-derived vocabulary); available on request

## Citation

Cite this repository / release / DOI once published. See `CITATION.cff`.

## Repository

- GitHub user: [`ch-auf-der-horst`](https://github.com/ch-auf-der-horst)
- Name: `heine-shannon-internationalitaet` (public)
- URL (after first push): https://github.com/ch-auf-der-horst/heine-shannon-internationalitaet

## Author

Christoph Aufderhorst — [affiliation TBD]
