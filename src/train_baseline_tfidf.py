#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline training: TF-IDF (word + char ngrams) -> multi-output regression for
10 category weights.

Inputs:
  - local/train_labeled_letters.csv (not published; contains letter texts)

Outputs:
  - local/ml_baseline_tfidf_model.joblib (not published)
  - data/ml_baseline_tfidf_report.md
  - data/ml_baseline_tfidf_metrics.json
    (Ridge + Baselines Uniform/Train-Mittel; je Methode: pooled Metriken, SD über Folds,
     fold_level, MAE/Kategorie-Std über Folds)
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = _REPO_ROOT / "data"
_LOCAL = _REPO_ROOT / "local"
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

from text_normalize import text_preprocess

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

def load_training_csv(path: Path) -> Tuple[List[str], np.ndarray, List[str], List[Dict[str, str]]]:
    csv.field_size_limit(sys.maxsize)
    texts: List[str] = []
    y_list: List[List[float]] = []
    groups: List[str] = []
    raw_rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            raw_rows.append(r)
            txt = r.get("brieftext") or ""
            texts.append(txt)

            # Group by source file (proxy for correspondent, avoids leakage)
            groups.append((r.get("source_file") or "UNKNOWN").strip())

            row_vals: List[float] = []
            for k in TARGET_COLS:
                v = (r.get(k) or "").strip()
                if v == "":
                    row_vals.append(float("nan"))
                else:
                    row_vals.append(float(v))
            y_list.append(row_vals)

    y = np.array(y_list, dtype=float)
    return texts, y, groups, raw_rows

def impute_and_renormalize(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    - Impute missing targets with 0.0 (conservative).
    - Renormalize each row to sum to 1 if sum > 0.
    """
    info: Dict[str, int] = {"rows": int(y.shape[0]), "nan_cells": int(np.isnan(y).sum())}
    y2 = y.copy()
    y2[np.isnan(y2)] = 0.0
    row_sums = y2.sum(axis=1)
    for i, s in enumerate(row_sums):
        if s > 0:
            y2[i, :] = y2[i, :] / s
    return y2, info

def project_to_simplex(yhat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Project predictions to a simple probability simplex:
    - clip negatives to 0
    - renormalize rows to sum to 1 (if sum>0 else uniform)
    """
    y = np.maximum(yhat, 0.0)
    sums = y.sum(axis=1, keepdims=True)
    zero_mask = sums.squeeze() <= eps
    if np.any(zero_mask):
        y[zero_mask, :] = 1.0 / y.shape[1]
        sums = y.sum(axis=1, keepdims=True)
    return y / sums

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + eps
    return num / den

def _cv_metrics_from_folds(
    maes: List[np.ndarray],
    cosines: List[np.ndarray],
    sum_abs_errs: List[float],
) -> Dict[str, object]:
    mae_mean = np.mean(np.vstack(maes), axis=0)
    cosine_all = np.concatenate(cosines)
    return {
        "mae_per_category": {k: float(v) for k, v in zip(TARGET_COLS, mae_mean)},
        "mae_macro": float(np.mean(mae_mean)),
        "cosine_mean": float(np.mean(cosine_all)),
        "cosine_p25": float(np.quantile(cosine_all, 0.25)),
        "cosine_p50": float(np.quantile(cosine_all, 0.50)),
        "cosine_p75": float(np.quantile(cosine_all, 0.75)),
        "sum_abs_err_mean": float(np.mean(sum_abs_errs)),
    }

def _std_across_folds(maes: List[np.ndarray], cosines: List[np.ndarray]) -> Dict[str, object]:
    """
    Streuung **zwischen** Folds: Pro Fold ein Mittelwert (MAE macro, Cosinus-Mittel, Cosinus-Median);
    SD über diese Fold-Statistiken. Pro Kategorie: SD der foldweisen MAE je Dimension.
    """
    n = len(maes)
    ddof = 1 if n > 1 else 0

    def _sd(a: List[float]) -> float:
        return float(np.std(np.array(a, dtype=float), ddof=ddof)) if n else 0.0

    fold_macro = [float(np.mean(m)) for m in maes]
    fold_cos_mean = [float(np.mean(c)) for c in cosines]
    fold_cos_p50 = [float(np.quantile(c, 0.5)) for c in cosines]

    stack = np.vstack(maes)
    mae_cat_std = {
        k: float(np.std(stack[:, j], ddof=ddof)) for j, k in enumerate(TARGET_COLS)
    }

    return {
        "mae_macro_std_across_folds": _sd(fold_macro),
        "cosine_mean_std_across_folds": _sd(fold_cos_mean),
        "cosine_p50_std_across_folds": _sd(fold_cos_p50),
        "mae_per_category_std_across_folds": mae_cat_std,
        "fold_level": {
            "mae_macro": fold_macro,
            "cosine_mean": fold_cos_mean,
            "cosine_p50": fold_cos_p50,
        },
    }

def _merge_pooled_and_fold_std(
    pooled: Dict[str, object],
    maes: List[np.ndarray],
    cosines: List[np.ndarray],
) -> Dict[str, object]:
    out = dict(pooled)
    out.update(_std_across_folds(maes, cosines))
    return out

def build_model(random_state: int = 0) -> Pipeline:
    # Word ngrams
    word = TfidfVectorizer(
        preprocessor=text_preprocess,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=150_000,
    )
    # Char ngrams (robust to historical spelling variation)
    char = TfidfVectorizer(
        preprocessor=text_preprocess,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=250_000,
    )

    feats = FeatureUnion([("word", word), ("char", char)])
    reg = MultiOutputRegressor(Ridge(alpha=5.0, random_state=random_state))

    return Pipeline([("features", feats), ("reg", reg)])

def group_cv_report(
    model: Pipeline,
    X: List[str],
    y: np.ndarray,
    groups: List[str],
    n_splits: int = 5,
) -> Dict[str, object]:
    uniq_groups = sorted(set(groups))
    n_splits = min(n_splits, len(uniq_groups))
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 groups for CV; got {len(uniq_groups)}")

    gkf = GroupKFold(n_splits=n_splits)
    k_dim = y.shape[1]
    uni = np.full((1, k_dim), 1.0 / k_dim)

    maes_r: List[np.ndarray] = []
    cos_r: List[np.ndarray] = []
    err_r: List[float] = []

    maes_u: List[np.ndarray] = []
    cos_u: List[np.ndarray] = []
    err_u: List[float] = []

    maes_m: List[np.ndarray] = []
    cos_m: List[np.ndarray] = []
    err_m: List[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        Xtr = [X[i] for i in train_idx]
        Xte = [X[i] for i in test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]
        n_te = len(yte)

        # --- Ridge (TF-IDF + multi-output); predictions projected to simplex ---
        model.fit(Xtr, ytr)
        yhat_r = project_to_simplex(model.predict(Xte))
        maes_r.append(np.mean(np.abs(yhat_r - yte), axis=0))
        cos_r.append(cosine_sim(yhat_r, yte))
        err_r.append(float(np.mean(np.abs(yhat_r.sum(axis=1) - 1.0))))

        # --- Uniform: constant (1/K,...,1/K) per row ---
        yhat_u = np.repeat(uni, n_te, axis=0)
        maes_u.append(np.mean(np.abs(yhat_u - yte), axis=0))
        cos_u.append(cosine_sim(yhat_u, yte))
        err_u.append(float(np.mean(np.abs(yhat_u.sum(axis=1) - 1.0))))

        # --- Train mean: single vector = column mean of y_train (no test leakage) ---
        if ytr.size == 0 or float(np.sum(ytr)) < 1e-15:
            yhat_m = np.repeat(uni, n_te, axis=0)
        else:
            mu = ytr.mean(axis=0, keepdims=True)
            yhat_m = np.repeat(mu, n_te, axis=0)
        maes_m.append(np.mean(np.abs(yhat_m - yte), axis=0))
        cos_m.append(cosine_sim(yhat_m, yte))
        err_m.append(float(np.mean(np.abs(yhat_m.sum(axis=1) - 1.0))))

    ridge_metrics = _merge_pooled_and_fold_std(
        _cv_metrics_from_folds(maes_r, cos_r, err_r), maes_r, cos_r
    )
    uniform_metrics = _merge_pooled_and_fold_std(
        _cv_metrics_from_folds(maes_u, cos_u, err_u), maes_u, cos_u
    )
    train_mean_metrics = _merge_pooled_and_fold_std(
        _cv_metrics_from_folds(maes_m, cos_m, err_m), maes_m, cos_m
    )

    out: Dict[str, object] = {
        "n_rows": int(len(X)),
        "n_groups": int(len(uniq_groups)),
        "n_splits": int(n_splits),
        # Backward-compatible top-level keys = Ridge CV (inkl. *_std_across_folds, fold_level)
        **ridge_metrics,
        "baseline_uniform": uniform_metrics,
        "baseline_train_mean": train_mean_metrics,
    }
    return out

def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"

def write_report_md(path: Path, metrics: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("## ML Baseline Report: TF-IDF (word+char) + Ridge (multi-output)\n")
    lines.append(f"- **Rows**: {metrics['n_rows']}\n")
    lines.append(f"- **Groups (source_file)**: {metrics['n_groups']}\n")
    lines.append(f"- **CV splits**: {metrics['n_splits']}\n")
    lines.append("\n### Vergleich GroupKFold (gleiche Folds, gleiche Metriken)\n\n")
    lines.append(
        "Pooled über alle Test-Briefe; **±** = Standardabweichung der **Fold-Mittelwerte** "
        "(zwischen den Folds; nicht SEM einzelner Briefe).\n\n"
    )
    lines.append("| Methode | MAE (macro) | Cosine mean | Cosine p50 |\n")
    lines.append("|---|---|---|---|\n")
    u = metrics["baseline_uniform"]
    m = metrics["baseline_train_mean"]
    lines.append(
        "| Uniform (1/K) | "
        f"{_fmt_pm(u['mae_macro'], u['mae_macro_std_across_folds'])} | "
        f"{_fmt_pm(u['cosine_mean'], u['cosine_mean_std_across_folds'])} | "
        f"{_fmt_pm(u['cosine_p50'], u['cosine_p50_std_across_folds'])} |\n"
    )
    lines.append(
        "| Train-Mittel (Fold) | "
        f"{_fmt_pm(m['mae_macro'], m['mae_macro_std_across_folds'])} | "
        f"{_fmt_pm(m['cosine_mean'], m['cosine_mean_std_across_folds'])} | "
        f"{_fmt_pm(m['cosine_p50'], m['cosine_p50_std_across_folds'])} |\n"
    )
    lines.append(
        "| **Ridge + TF-IDF** | "
        f"**{_fmt_pm(metrics['mae_macro'], metrics['mae_macro_std_across_folds'])}** | "
        f"**{_fmt_pm(metrics['cosine_mean'], metrics['cosine_mean_std_across_folds'])}** | "
        f"**{_fmt_pm(metrics['cosine_p50'], metrics['cosine_p50_std_across_folds'])}** |\n"
    )
    lines.append("\n*Uniform / Train-Mittel: keine Textmerkmale; Train-Mittel = Spaltenmittel nur aus Trainingsbriefen des Folds.*\n")
    lines.append(f"\n- **Mean |sum(pred)-1| (Ridge)**: {metrics['sum_abs_err_mean']:.6f}\n")

    lines.append("\n### MAE pro Kategorie (Mittel ± SD über Folds)\n\n")
    lines.append("| Kategorie | Uniform | Train-Mittel | Ridge |\n")
    lines.append("|---|---|---|---|\n")
    u_mae = u["mae_per_category"]
    u_sd = u["mae_per_category_std_across_folds"]
    m_mae = m["mae_per_category"]
    m_sd = m["mae_per_category_std_across_folds"]
    r_mae = metrics["mae_per_category"]
    r_sd = metrics["mae_per_category_std_across_folds"]
    for k in TARGET_COLS:
        lines.append(
            f"| {k} | {_fmt_pm(u_mae[k], u_sd[k])} | {_fmt_pm(m_mae[k], m_sd[k])} | "
            f"**{_fmt_pm(r_mae[k], r_sd[k])}** |\n"
        )

    path.write_text("".join(lines), encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_csv",
        default=str(_LOCAL / "train_labeled_letters.csv"),
    )
    ap.add_argument(
        "--model_out",
        default=str(_LOCAL / "ml_baseline_tfidf_model.joblib"),
    )
    ap.add_argument(
        "--report_md",
        default=str(_DATA / "ml_baseline_tfidf_report.md"),
    )
    ap.add_argument(
        "--metrics_json",
        default=str(_DATA / "ml_baseline_tfidf_metrics.json"),
    )
    args = ap.parse_args(argv)

    train_csv = Path(args.train_csv)
    texts, y_raw, groups, _ = load_training_csv(train_csv)
    y, info = impute_and_renormalize(y_raw)

    model = build_model()
    metrics = group_cv_report(model, texts, y, groups)
    metrics["imputation_info"] = info

    # Fit on full data for downstream inference
    model.fit(texts, y)
    dump(model, args.model_out)

    Path(args.metrics_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_md(Path(args.report_md), metrics)

    print(f"Wrote model: {args.model_out}")
    print(f"Wrote report: {args.report_md}")
    print(f"Wrote metrics: {args.metrics_json}")
    print(
        f"Ridge — MAE macro: {metrics['mae_macro']:.4f} ± {metrics['mae_macro_std_across_folds']:.4f} "
        f"| cosine mean: {metrics['cosine_mean']:.4f} ± {metrics['cosine_mean_std_across_folds']:.4f}"
    )
    u, m = metrics["baseline_uniform"], metrics["baseline_train_mean"]
    print(
        f"Uniform — MAE: {u['mae_macro']:.4f} ± {u['mae_macro_std_across_folds']:.4f} "
        f"|| Train-mean — MAE: {m['mae_macro']:.4f} ± {m['mae_macro_std_across_folds']:.4f}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

