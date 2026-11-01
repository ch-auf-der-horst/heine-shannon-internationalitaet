# Metadaten-Kreuztabellen und P_k (partner-gewichtet, TOP-100)

- **Input Vorhersagen:** `brief_predictions_top100.csv`
- **Input Codebook:** `codebook_ready_3.csv`
- **Aggregation:** Pro `partner_vollname` Mittel der zehn Funktionsgewichte über alle Briefe; jeder Partner zählt einmal.
- **Anzahl Partner (Zeilen):** 96
- **`(nicht im Codebook)`:** Partner in den Vorhersagen ohne exakten `Vollname`-Treffer; siehe `partners_nicht_im_codebook.csv` (falls vorhanden).

## Kreuztabellen (Anzahl Partner pro Zelle)

### Nat_Bez

- `crosstab_Nat_Bez_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Nat_Bez_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`

### Nat_Beruf

- `crosstab_Nat_Beruf_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Nat_Beruf_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`
- **Partner vs. Brief:** `plot_nat_beruf_partner_vs_brief_sidebyside` (Nebeneinander), `crosstab_Nat_Beruf_counts_briefe.csv`, `_row_norm_briefe.csv`
- **Abb. 16 Alternativen:** `plot_abb16_nat_beruf_de_fr_comparison` (gruppiert, deutsch vs. französisch), `plot_abb16_nat_beruf_compact_heatmap` (kompakte Heatmap)

### Nat_JH

- `crosstab_Nat_JH_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Nat_JH_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`

### Bez_Beruf

- `crosstab_Bez_Beruf_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Bez_Beruf_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`

### Bez_JH

- `crosstab_Bez_JH_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Bez_JH_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`

### Beruf_JH

- `crosstab_Beruf_JH_counts.csv`, `_row_norm.csv`, `_col_norm.csv`
- `plot_crosstab_Beruf_JH_counts.png/.pdf`, `_row_norm`, `_col_norm`; **gestapelt (ohne „(nicht im Codebook)“):** `_row_norm_stacked`, `_col_norm_stacked`

## Mittleres P_k pro Metadimension

- P_k-Grafiken und zugehörige CSV nutzen nur Partner **mit** Codebook-Zeile (ohne „(nicht im Codebook)“).

- `pk_mittel_pro_Nationalität.csv` — `plot_pk_mittel_Nationalität.png/.pdf`
- `pk_kategoriemasse_anteil_pro_Nationalität.csv` — `plot_pk_kategoriemasse_anteil_Nationalität.png/.pdf`
- `pk_mittel_pro_Beziehungstyp.csv` — `plot_pk_mittel_Beziehungstyp.png/.pdf`
- `pk_kategoriemasse_anteil_pro_Beziehungstyp.csv` — `plot_pk_kategoriemasse_anteil_Beziehungstyp.png/.pdf`
- `pk_mittel_pro_Beruf.csv` — `plot_pk_mittel_Beruf.png/.pdf`
- `pk_kategoriemasse_anteil_pro_Beruf.csv` — `plot_pk_kategoriemasse_anteil_Beruf.png/.pdf`
- `pk_mittel_pro_Juedische_Herkunft.csv` — `plot_pk_mittel_Juedische_Herkunft.png/.pdf`
- `pk_kategoriemasse_anteil_pro_Juedische_Herkunft.csv` — `plot_pk_kategoriemasse_anteil_Juedische_Herkunft.png/.pdf`

## Hinweise

- `(fehlend)` = leeres Codebook-Feld; `(nicht im Codebook)` = kein Zeilentreffer für `partner_vollname` unter `Vollname` (Tippfehler/abweichende Schreibweise). Wiederholung ohne diese Zeile: Skript mit `--nur-mit-codebook` ausführen.
- **Kategoriemasse:** Für jede Funktion k ist der Anteil pro Gruppe: Summe der mittleren Gewichte über Partner in g / Summe über alle Partner (jeder Partner gleich).
