"""The shipped v1 detector: score every function in a binary and select the cryptographic ones with FDR control.

Runtime is CPU only and offline. The model and the per-ISA calibration scores are files in model/, exported
by research/experiments/export_v1.py from exactly the configuration evaluated in
research/results/v1_1_detector.json. Nothing is trained here.

For one binary:
    1. find functions (engine.binary_ml.functions): symbols when present, call targets when stripped;
    2. describe each one (engine.binary_ml.features) and score it with the model;
    3. turn scores into conformal p-values against the non-cryptographic calibration functions *of the same
       ISA* (v1.1: calibrating on another ISA broke the guarantee), then apply Benjamini-Hochberg at alpha.
The result states what was selected, each selected function's q-value, and anything that stopped the
detector (unsupported ISA, too many functions), so a skipped binary is reported, never silently clean.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine.binary_ml.conformal import bh, bh_qvalues, conformal_pvalues
from engine.binary_ml.features import FEATURES, MIN_INSNS, function_vector
from engine.binary_ml.functions import Function

MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_ALPHA = float(os.environ.get("VERA_DETECTOR_ALPHA", "0.1"))
MAX_FUNCTIONS = int(os.environ.get("VERA_DETECTOR_MAX_FUNCTIONS", "3000"))


@dataclass(frozen=True)
class Selected:
    address: int
    name: str
    score: float
    q_value: float


@dataclass
class Detection:
    arch: str
    boundary: str
    functions_scored: int
    alpha: float
    selected: list[Selected] = field(default_factory=list)
    min_q: float | None = None
    skipped: str | None = None
    model: str = ""


@lru_cache(maxsize=1)
def _load():
    import joblib   # imported lazily: a scan that never meets a binary never loads the model
    meta = json.loads((MODEL_DIR / "v1_calibration.json").read_text())
    if meta["meta"]["features"] != list(FEATURES):
        raise RuntimeError("detector model was exported with a different feature list; re-run export_v1.py")
    model = joblib.load(MODEL_DIR / "v1_model.joblib")
    nulls = {arch: np.asarray(v, dtype=float) for arch, v in meta["null_scores"].items()}
    return model, nulls, meta["meta"]


def available() -> bool:
    return (MODEL_DIR / "v1_model.joblib").exists() and (MODEL_DIR / "v1_calibration.json").exists()


def detect(funcs: list[Function], alpha: float = DEFAULT_ALPHA) -> Detection | None:
    """Score and select; None when there is nothing to score."""
    if not funcs:
        return None
    arch, boundary = funcs[0].arch, funcs[0].boundary
    model, nulls, meta = _load()
    result = Detection(arch=arch, boundary=boundary, functions_scored=0, alpha=alpha, model=meta["model"])
    if arch not in nulls:
        result.skipped = f"no calibration for ISA {arch}"
        return result
    if len(funcs) > MAX_FUNCTIONS:
        result.skipped = f"{len(funcs)} functions exceeds VERA_DETECTOR_MAX_FUNCTIONS={MAX_FUNCTIONS}"
        return result
    rows, kept = [], []
    for fn in funcs:
        vec, n = function_vector(fn.code, fn.arch, fn.address)
        if n >= MIN_INSNS:
            rows.append(vec)
            kept.append(fn)
    result.functions_scored = len(kept)
    if not kept:
        return result
    scores = model.predict_proba(np.asarray(rows))[:, 1]
    p = conformal_pvalues(scores, nulls[arch])
    q = bh_qvalues(p)
    sel = bh(p, alpha)
    result.min_q = float(q.min())
    order = np.argsort(q)
    result.selected = [Selected(kept[i].address, kept[i].name, round(float(scores[i]), 4), round(float(q[i]), 4))
                       for i in order if sel[i]]
    return result
