"""Distribution-free false-discovery control for "this function is cryptographic".

Vendored from engine/binary_ml/conformal.py (rung fusion omitted); the scorer needs only selection.

THE CLAIM IT SUPPORTS
---------------------
Given detector scores for every function in the binaries under test, and scores for a *calibration*
set of functions known to be non-cryptographic, the functions this module selects contain, in
expectation, at most a fraction ``alpha`` of non-cryptographic ones:

    E[ #false selections / max(1, #selections) ] <= alpha

The only assumption is that the non-cryptographic test functions and the calibration functions are
exchangeable (drawn the same way). Nothing is assumed about the detector: it can be any score, and a
bad score costs power (fewer true finds), never validity. When exchangeability fails, for example
calibrating on x86-64 and testing on ARM, the bound can fail; research/ measures exactly that.

TWO PROCEDURES
--------------
conformal p-values + Benjamini-Hochberg
    p_j = (1 + #{calibration nulls scoring >= s_j}) / (n0 + 1). The p-values are PRDS, so BH at level
    alpha controls FDR (Bates, Candes, Lei, Romano, Sesia, Ann. Statist. 2023; the selection form is
    Jin & Candes, JMLR 2023).
conformal e-values + e-BH
    e_j = (n0 + 1) * 1{s_j >= T} / (1 + #{calibration nulls >= T}), with T the conformal-BH threshold
    (Bashari, Epstein, Romano, Sesia, NeurIPS 2023). e-BH (Wang & Ramdas, JRSS-B 2022) controls FDR
    under *any* dependence, so e-values from several detectors (ladder rungs, per-architecture models)
    can be averaged and the average is still a valid e-value. That is what lets the ladder add rungs
    without re-deriving the guarantee.

Each finding carries its q-value: the smallest alpha at which it would be selected. That is the number
an analyst can defend in an audit, and it is what the CBOM records.
"""
from __future__ import annotations

import numpy as np


def conformal_pvalues(test_scores: np.ndarray, null_calib_scores: np.ndarray) -> np.ndarray:
    """p_j = (1 + #{null >= s_j}) / (n0 + 1); higher score = more cryptographic."""
    null = np.sort(np.asarray(null_calib_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n0 = null.size
    if n0 == 0:
        raise ValueError("conformal p-values need at least one calibration null")
    ge = n0 - np.searchsorted(null, s, side="left")      # count of nulls >= s
    return (1.0 + ge) / (n0 + 1.0)


def bh(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg selection mask at level alpha."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    thresh = alpha * np.arange(1, m + 1) / m
    passed = np.nonzero(p[order] <= thresh)[0]
    sel = np.zeros(m, dtype=bool)
    if passed.size:
        sel[order[: passed[-1] + 1]] = True
    return sel


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values: the smallest alpha at which each hypothesis is rejected."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def conformal_evalues(test_scores: np.ndarray, null_calib_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Conformal e-values at the BH stopping threshold (Bashari et al. 2023, eq. for FDR control)."""
    null = np.sort(np.asarray(null_calib_scores, dtype=float))
    s = np.asarray(test_scores, dtype=float)
    n0, m = null.size, s.size
    if n0 == 0 or m == 0:
        return np.zeros(m)
    # Candidate thresholds are the observed test scores; T is the smallest t whose estimated FDP <= alpha.
    cands = np.unique(s)
    null_ge = n0 - np.searchsorted(null, cands, side="left")
    test_ge = m - np.searchsorted(np.sort(s), cands, side="left")
    fdp_hat = (m / (n0 + 1.0)) * (1.0 + null_ge) / np.maximum(1, test_ge)
    ok = np.nonzero(fdp_hat <= alpha)[0]
    t = cands[ok[0]] if ok.size else np.inf
    denom = 1.0 + (n0 - np.searchsorted(null, t, side="left") if np.isfinite(t) else 0)
    return np.where(s >= t, (n0 + 1.0) / denom, 0.0)


def ebh(evalues: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH selection mask: reject the k largest e-values, k = max{k : e_(k) >= m / (alpha k)}."""
    e = np.asarray(evalues, dtype=float)
    m = e.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-e, kind="stable")
    k = np.arange(1, m + 1)
    passed = np.nonzero(e[order] >= m / (alpha * k))[0]
    sel = np.zeros(m, dtype=bool)
    if passed.size:
        sel[order[: passed[-1] + 1]] = True
    return sel
