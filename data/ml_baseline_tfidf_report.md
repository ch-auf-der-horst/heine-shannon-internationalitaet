## ML Baseline Report: TF-IDF (word+char) + Ridge (multi-output)
- **Rows**: 80
- **Groups (source_file)**: 16
- **CV splits**: 5

### Vergleich GroupKFold (gleiche Folds, gleiche Metriken)

Pooled über alle Test-Briefe; **±** = Standardabweichung der **Fold-Mittelwerte** (zwischen den Folds; nicht SEM einzelner Briefe).

| Methode | MAE (macro) | Cosine mean | Cosine p50 |
|---|---|---|---|
| Uniform (1/K) | 0.1422 ± 0.0076 | 0.4877 ± 0.0297 | 0.4813 ± 0.0278 |
| Train-Mittel (Fold) | 0.1269 ± 0.0056 | 0.5394 ± 0.0409 | 0.5898 ± 0.0464 |
| **Ridge + TF-IDF** | **0.1203 ± 0.0066** | **0.5871 ± 0.0456** | **0.6615 ± 0.0205** |

*Uniform / Train-Mittel: keine Textmerkmale; Train-Mittel = Spaltenmittel nur aus Trainingsbriefen des Folds.*

- **Mean |sum(pred)-1| (Ridge)**: 0.000000

### MAE pro Kategorie (Mittel ± SD über Folds)

| Kategorie | Uniform | Train-Mittel | Ridge |
|---|---|---|---|
| INTELLECTUAL | 0.1350 ± 0.0359 | 0.1313 ± 0.0320 | **0.1315 ± 0.0400** |
| NETWORK | 0.1187 ± 0.0284 | 0.0890 ± 0.0368 | **0.0932 ± 0.0396** |
| POLITICAL | 0.0981 ± 0.0088 | 0.0467 ± 0.0166 | **0.0466 ± 0.0192** |
| PATRONAGE | 0.1344 ± 0.0406 | 0.1122 ± 0.0426 | **0.1048 ± 0.0519** |
| AFFECTIVE | 0.1369 ± 0.0561 | 0.1524 ± 0.0388 | **0.1363 ± 0.0461** |
| INSTITUTIONAL | 0.1453 ± 0.0872 | 0.1240 ± 0.0738 | **0.1003 ± 0.0894** |
| EDITORIAL | 0.1762 ± 0.0652 | 0.2039 ± 0.0282 | **0.1922 ± 0.0345** |
| FAMILIAL | 0.1996 ± 0.0505 | 0.1970 ± 0.0217 | **0.1847 ± 0.0188** |
| TRANSACTIONAL | 0.1753 ± 0.0299 | 0.1891 ± 0.0184 | **0.1894 ± 0.0217** |
| CONFLICT | 0.1022 ± 0.0107 | 0.0229 ± 0.0150 | **0.0244 ± 0.0154** |
