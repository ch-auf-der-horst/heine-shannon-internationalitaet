# Reproduction notes

## What works without edition full texts

With the files under `data/` you can inspect:

- correspondent profiles and Shannon-*H* distributions
- internationality analyses (nationality × phase)
- metadata crosstabs and mean *Pₖ*
- robustness comparison tables (where CSV is shipped)

Scripts under `src/` use **repo-relative** defaults (`data/`, `local/`).

## What stays local (`local/`)

Not published (place here on your machine if you re-run inference/training):

- DHA master CSV (letter full texts)
- trained TF-IDF model (`.joblib`)
- training CSV **with** `brieftext`
- TOP-100 name list used for inference filtering

See `local/README.md`.

## Training weights in this repo

`data/train_labeled_letters_weights_only.csv` contains category weights and metadata for the 80 pilot letters **without** the `brieftext` column. Re-training TF-IDF requires attaching letter texts locally via letter IDs.

## Trained model

The `.joblib` model is **not** included. It may embed vocabulary derived from edition texts and is available on request / for local use with edition access.
