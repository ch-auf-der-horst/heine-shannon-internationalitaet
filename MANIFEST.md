# GitHub-Ablage: Manifest (Heine / Shannon / Internationalität)

Stand: 2026-10-31 · Pipeline-Referenz: Master `alle_baende_20-38_MASTER_VOLLSTÄNDIG_20260619_144717.csv` · TOP-100: **96 Partner**, ~2394 Briefe

Dieses Manifest definiert, was in ein öffentliches Reproduktions-Repository gehört — **nicht** den gesamten Sciebo-Arbeitsordner.

---

## Empfohlene Repo-Struktur

```text
heine-shannon-internationalitaet/
├── README.md
├── CITATION.cff                 # optional, für Zitation
├── LICENSE                     # Code: z. B. MIT
├── LICENSE-DATA.md             # Daten: Rechtehinweis / CC-BY o. ä.
├── requirements.txt
├── src/                        # Pipeline-Skripte (aus ml/)
├── data/
│   ├── codebook_ready_3.csv
│   ├── train_labeled_letters_weights_only.csv   # OHNE brieftext
│   ├── brief_predictions_top100.csv             # Gewichte, kein Volltext
│   ├── korrespondenten_profile_pred_top100.csv
│   ├── shannon_*.csv / *.md
│   └── metadata_partner_top100/                 # Kreuztabellen + Plots
├── figures/                    # Publikationsabbildungen (PNG/PDF)
└── docs/                       # kurze Methoden-/Abbildungsnotes
```

---

## EINBEZIEHEN (öffentlich, reproduktionsrelevant)

### A. Pipeline-Code (`ml/` → `src/`)

| Datei | Rolle |
|---|---|
| `text_normalize.py` | Textnormalisierung |
| `build_training_set.py` | Trainingskorpus (lokal mit Master) |
| `train_baseline_tfidf.py` | TF-IDF + Ridge |
| `infer_top100.py` | Inferenz TOP-100 |
| `aggregate_shannon.py` | Partnerprofile + Shannon H |
| `analyze_and_visualize_shannon.py` | Deskriptiva, Histogramm, Top/Bottom |
| `shannon_by_metadata_phase.py` | H × Phase × Metadaten |
| `plot_internationality_shannon.py` | Abb. 12–13 |
| `metadata_crosstabs_pk_partner_top100.py` | Abb. 14–16, Nat×Beruf |
| `analyze_global_pk_differences.py` | ΔPₖ global |
| `global_shannon_by_phase_direction.py` | Makro-H Phase×Richtung |
| `plot_global_shannon_hybrid_by_year.py` | Abb. 10 |
| `robustness_shannon.py` | Robustheit |
| `robustness_top_changes.py` | Rangverschiebungen |
| `join_profile_codebook_h_by_metadata.py` | H×Metadaten Join |
| `export_sammelband_abbildungen_600dpi.py` | optional: Export 600 dpi |

**Optional (wenn der Aufsatz LDA/Word2Vec mitführt):** `lda_*.py` — sonst weglassen, um den Fokus scharf zu halten.

### B. Daten ohne Briefvolltexte

| Datei / Ordner | Hinweis |
|---|---|
| `Aufsatz_Sammelband/codebook_ready_3.csv` | Partner-Metadaten + manuelle Profilgewichte |
| `Auswertungen/brief_predictions_top100.csv` | **sicher** (nur letterid + Gewichte) |
| `Auswertungen/korrespondenten_profile_pred_top100.csv` | Partnerprofile + H |
| `github_deposit/data/train_labeled_letters_weights_only.csv` | 80 Briefe, **ohne** `brieftext` |
| `Auswertungen/shannon_H_by_metadata_phase.csv` | Zellen für Abb. 12 |
| `Auswertungen/shannon_H_by_metadata_phase_report.md` | Report |
| `Auswertungen/shannon_top_bottom.csv` | Extremgruppen |
| `Auswertungen/shannon_bootstrap_ci.csv` | Bootstrap-CIs |
| `Auswertungen/global_shannon_top100_by_phase_direction.csv` | Makro |
| `Auswertungen/global_pk_differences_top100.csv` | ΔPₖ |
| `Auswertungen/robustness_shannon_comparison.csv` | Robustheit |
| `Auswertungen/metadata_partner_top100/` | Kreuztabellen + Abb. 14–16 |
| `Auswertungen/ml_baseline_tfidf_model.joblib` | optional: fertiges Modell (Reproduktion ohne Retrain) |
| `Auswertungen/ml_baseline_tfidf_report.md` | CV-Report |

### C. Abbildungen (Kern des Aufsatzes)

| Abbildung | Datei(en) |
|---|---|
| Abb. 5–8 (Shannon mikro) | `plot_shannon_hist`, `plot_shannon_top_bottom`, `plot_shannon_forest_ci`, `plot_category_heatmap`, `plot_shannon_scatter_nbriefe` |
| Abb. 10 (Makro) | `plot_global_shannon_hybrid_by_year` |
| Abb. 12–13 (Internationalität) | `plot_internationality_heatmap`, `plot_internationality_partner_violin_de_fr` (+ optional `plot_internationality_H_de_fr`) |
| Abb. 14–16 | unter `metadata_partner_top100/` (`plot_pk_mittel_Nationalität`, `plot_crosstab_Nat_Bez_*`, `plot_crosstab_Nat_Beruf_*`, Abb.-16-Varianten) |

Jeweils **PNG + PDF**, wenn vorhanden.

### D. Dokumentation (kurz)

| Inhalt | Quelle |
|---|---|
| README (dieses Repo) | neu schreiben |
| Pipeline-Stand / Master-Hinweis | `MASTER_DATEI_DOKUMENTATION.md` (gekürzt) |
| Optional: Entwürfe 5.3.x | `entwurf_abschnitt_5_3_*.md` — nur wenn Sie Methodennotizen mitliefern wollen |

---

## NICHT einbeziehen (Rechte / Rauschen)

| Pfad / Typ | Grund |
|---|---|
| `alle_baende_20-38_MASTER_*.csv` | Briefvolltexte / Editionsmaterial |
| `Auswertungen/train_labeled_letters.csv` (Original) | enthält Spalte **`brieftext`** |
| XML-Bände, Rohordner `Briefe_*`, `Nachtragsbände` | Editionsdaten |
| `.venv*`, `__pycache__`, `.DS_Store` | Umgebung |
| Gesamter Sciebo-`test/`-Baum | Arbeitsablage, nicht Reproduktion |
| `*_Kopie.png`, Zwischenplots, Folien-Exports | Rauschen |
| Persönliche Beleg-Excel unter `Aufsatz_Sammelband/*BELEGE*` | Arbeitsnotizen |
| LDA/Word2Vec-Korpora mit Volltext | nur bei klarer Rechtefreigabe |

---

## Rechte-Stufen (Empfehlung)

| Stufe | Inhalt | Wann |
|---|---|---|
| **A – öffentlich ohne Klärung** | Code + aggregierte H/Pₖ + Predictions ohne Text + redigiertes Training | Standard für Aufsatz-Link |
| **B – nach Rechteklärung** | Brieftexte / Masterauszug / Trainingsvolltexte | nur mit DHA/Verlag |
| **C – auf Anfrage** | reproduzierbare Private-Kopie für Gutachter | Data Availability „available upon request“ |

---

## Reproduktionspfad (nach Ablage)

```text
1. train_baseline_tfidf.py          # oder fertiges .joblib laden
2. infer_top100.py                  # braucht lokalen Master (nicht im Repo)
3. aggregate_shannon.py
4. analyze_and_visualize_shannon.py
5. shannon_by_metadata_phase.py
6. plot_internationality_shannon.py
7. metadata_crosstabs_pk_partner_top100.py
8. analyze_global_pk_differences.py / plot_global_shannon_hybrid_by_year.py
```

**Hinweis:** Schritt 2 setzt die DHA-Masterdatei lokal voraus. Das öffentliche Repo dokumentiert das und liefert die **bereits berechneten** `brief_predictions_top100.csv` / Partnerprofile als Snapshot zum Manuskriptstand.

---

## Aufsatz-Formulierung (Data Availability)

> Reproduktionsmaterial (Pipeline-Code, aggregierte Kategorien- und Shannon-Kennzahlen, Abbildungen; ohne Briefvolltexte der Edition) ist unter  
> `https://github.com/<user>/heine-shannon-internationalitaet`  
> als Release `v1.0` abgelegt. [Optional: DOI via Zenodo.]

---

## Nächste Schritte

1. GitHub-Repo anlegen (öffentlich)  
2. Dateien laut diesem Manifest kopieren (`prepare_repo.sh` in diesem Ordner)  
3. README anpassen (Autor, Affiliation, Lizenz)  
4. Release `v1.0` taggen → optional Zenodo → DOI in Fußnote  
5. Link im Aufsatz eintragen
