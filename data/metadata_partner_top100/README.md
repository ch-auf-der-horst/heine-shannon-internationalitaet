# Metadaten Kreuztabellen & P_k (partner-gewichtet, TOP-100)

Dieser Ordner wird durch das Skript `test/ml/metadata_crosstabs_pk_partner_top100.py` befüllt.

## Ausführung (lokal im Projekt `test/`)

```bash
cd test
MPLBACKEND=Agg .venv_ml/bin/python ml/metadata_crosstabs_pk_partner_top100.py
```

**Ohne die Rubrik „(nicht im Codebook)“** in den Grafiken (nur Partner mit Zeile im Codebook):

```bash
MPLBACKEND=Agg .venv_ml/bin/python ml/metadata_crosstabs_pk_partner_top100.py --nur-mit-codebook
```

Trefferlisten ohne Codebook-Match werden bei Bedarf als `partners_nicht_im_codebook.csv` geschrieben (zur Korrektur von `Vollname` / `partner_vollname`).

Optional:

```bash
.venv_ml/bin/python ml/metadata_crosstabs_pk_partner_top100.py \
  --pred_csv Auswertungen/brief_predictions_top100.csv \
  --codebook Aufsatz_Sammelband/codebook_ready_3.csv \
  --out_dir Auswertungen/metadata_partner_top100
```

## Erzeugte Dateien

- `partner_stufe_pk_meta.csv` — eine Zeile pro Korrespondent: Mittel-P_k über alle Briefe + Metadaten
- `crosstab_<Paar>_counts|row_norm|col_norm.csv` — sechs Metadaten-Paare (Nationalität, Beziehungstyp, Beruf, Jüdische Herkunft)
- `pk_mittel_pro_<Dimension>.csv` — mittleres P_k je Ausprägung
- `pk_kategoriemasse_anteil_pro_<Dimension>.csv` — Anteil der summierten Kategoriengewichte je Gruppe (Zeilen summieren zu 1)
- `metadata_partner_top100_report.md` — Dateiindex
- Zu jedem Datensatz passende **PNG/PDF**-Heatmaps
- Zusätzlich **gestapelte 100-%-Balken** (`plot_crosstab_*_row_norm_stacked`, `plot_crosstab_*_col_norm_stacked`): dieselben Kreuztabellen wie die Heatmaps zu Zeilen-/Spaltenanteilen, aber ohne die Rubrik **„(nicht im Codebook)“** (Zählmatrix vorher reduziert, Anteile neu normiert).

Die **P_k**-CSV und **P_k**-Heatmaps beziehen sich ebenfalls nur auf Partner **mit** Codebook-Zeile (ohne „nicht im Codebook“); `partner_stufe_pk_meta.csv` enthält weiter alle Partner, sofern nicht `--nur-mit-codebook` gesetzt ist.

Aggregation: **partner-gewichtet** = pro Partner Mittel der zehn Vorhersagegewichte über alle Briefe in `brief_predictions_top100.csv`; jeder Partner zählt gleich oft.

**Lesarten zu allen Grafiken:** siehe `Grafiken_Beschreibung_und_Interpretation.md` im gleichen Ordner.
