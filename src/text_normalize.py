#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared text normalization helpers for ML scripts.

Important: keep functions importable (not defined in __main__) so that
joblib-persisted sklearn pipelines can be loaded from other scripts.
"""

from __future__ import annotations

import unicodedata


def normalize_unicode_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ")
    return " ".join(s.split())


def text_preprocess(s: str) -> str:
    s = normalize_unicode_ws(s)
    return s.lower()

