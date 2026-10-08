"""Score a detector on IndiCrypt-Bench.

Input: a table with one row per function and a detector score (higher = more cryptographic).

    required   library, rel, name, arch (x86-64 | aarch64 | arm32), score
    optional   label (1/0 or crypto/noncrypto; derived from the frozen rule when absent), uid

Output: ROC-AUC and PR-AUC overall and per ISA, and the false-discovery proportion (FDP) among flagged
functions at realistic prevalence (natural, 5%, 1% crypto) for the fixed 0.9 cut-off and for conformal
Benjamini-Hochberg selection, following experiments/v1_1_detector.py: the pool is split many times into a
calibration half and a test half *by source function*, calibration nulls are the negatives of the calibration
half, and positives are subsampled on the test side only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .conformal import bh, conformal_pvalues
from .data import ISAS, resolve_labels
from .splits import libraries, split_of

REQUIRED = ("library", "rel", "name", "arch", "score")
ARCH_ALIASES = {"x86_64": "x86-64", "x64": "x86-64", "amd64": "x86-64", "x86-64": "x86-64", "aarch64": "aarch64",
                "arm64": "aarch64", "a64": "aarch64", "arm32": "arm32", "arm": "arm32", "armv7": "arm32"}
SEED = 20260928


def _labels(raw: pd.Series) -> pd.Series:
    def one(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"crypto", "1", "true"}:
                return 1.0
            if s in {"noncrypto", "non-crypto", "0", "false"}:
                return 0.0
            return np.nan
        return float(v) if v in (0, 1) else np.nan
    return raw.map(one)


def prepare(df: pd.DataFrame, split: str = "all", data_dir=None) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(missing)}")
    df = df.copy()
    df["arch"] = df["arch"].astype(str).str.lower().map(ARCH_ALIASES)
    if df["arch"].isna().any():
        raise ValueError("arch must be one of x86-64, aarch64, arm32")
    df["score"] = pd.to_numeric(df["score"], errors="raise")
    df["y"] = _labels(df["label"]) if "label" in df.columns else resolve_labels(df, data_dir)
    if split != "all":
        df = df[df["library"].isin(libraries(split))]
    df = df[df["y"].notna()].copy()
    if "uid" not in df.columns:
        df["uid"] = df["library"] + "/" + df["rel"] + "/" + df["name"].astype(str).str.split(".").str[0]
    return df.reset_index(drop=True)


def _auc(y: np.ndarray, s: np.ndarray) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n": int(len(y))}
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4), "pr_auc": round(float(average_precision_score(y, s)), 4),
            "n": int(len(y))}


def _fdp(sel: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    n = int(sel.sum())
    return float((sel & (y == 0)).sum() / max(1, n)), float((sel & (y == 1)).sum() / max(1, (y == 1).sum())), n


def conformal_study(s: np.ndarray, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                    prevalence: float | None, alpha: float, n_splits: int, threshold: float = 0.9) -> dict:
    codes, uniq = pd.factorize(groups)
    ids = np.arange(len(uniq))
    acc: dict[str, list] = {"bh": [], "fixed": []}
    for _ in range(n_splits):
        cal_ids = rng.choice(ids, size=len(ids) // 2, replace=False)
        in_cal = np.isin(codes, cal_ids)
        cal = in_cal & (y == 0)
        idx = np.nonzero(~in_cal)[0]
        if cal.sum() == 0 or idx.size == 0:
            continue
        if prevalence is not None:
            neg, pos = idx[y[idx] == 0], idx[y[idx] == 1]
            k = int(round(prevalence * len(neg) / (1 - prevalence)))
            idx = np.concatenate([neg, rng.choice(pos, size=min(k, len(pos)), replace=False)])
        st, yt = s[idx], y[idx]
        acc["bh"].append(_fdp(bh(conformal_pvalues(st, s[cal]), alpha), yt))
        acc["fixed"].append(_fdp(st >= threshold, yt))
    out = {}
    for k, rows in acc.items():
        if not rows:
            out[k] = None
            continue
        a = np.array(rows)
        out[k] = {"mean_fdp": round(float(a[:, 0].mean()), 4), "p90_fdp": round(float(np.quantile(a[:, 0], 0.9)), 4),
                  "mean_power": round(float(a[:, 1].mean()), 4), "mean_selected": round(float(a[:, 2].mean()), 1)}
    return out


def score(df: pd.DataFrame, split: str = "all", alpha: float = 0.1, n_splits: int = 200, seed: int = SEED,
          prevalences: tuple = (None, 0.05, 0.01), data_dir=None) -> dict:
    d = prepare(df, split, data_dir)
    if d.empty:
        raise ValueError("no labelled rows to score (check library/rel/name, or pass a label column)")
    y, s = d["y"].to_numpy().astype(int), d["score"].to_numpy(float)
    res: dict = {"split": split, "alpha": alpha, "n_rows": int(len(d)), "n_crypto": int(y.sum()),
                 "natural_prevalence": round(float(y.mean()), 4), "n_splits": n_splits,
                 "overall": _auc(y, s), "per_isa": {}, "fdp": {}}
    for a in ISAS:
        m = (d["arch"] == a).to_numpy()
        if m.any():
            res["per_isa"][a] = _auc(y[m], s[m])
    rng = np.random.default_rng(seed)
    if len(np.unique(y)) == 2:
        for p in prevalences:
            key = "natural" if p is None else f"{p:g}"
            res["fdp"][key] = conformal_study(s, y, d["uid"].to_numpy(), rng, p, alpha, n_splits)
    return res


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def format_report(r: dict) -> str:
    lines = [f"IndiCrypt-Bench score  split={r['split']}  rows={r['n_rows']}  crypto={r['n_crypto']} "
             f"({r['natural_prevalence']:.1%})", "",
             f"{'':10s}{'ROC-AUC':>9s}{'PR-AUC':>9s}{'n':>9s}"]
    lines.append(f"{'all ISAs':10s}{_f(r['overall']['roc_auc']):>9s}{_f(r['overall']['pr_auc']):>9s}{r['overall']['n']:>9d}")
    for a, v in r["per_isa"].items():
        lines.append(f"{a:10s}{_f(v['roc_auc']):>9s}{_f(v['pr_auc']):>9s}{v['n']:>9d}")
    if r["fdp"]:
        lines += ["", f"False-discovery proportion among flagged functions ({r['n_splits']} calibration/test splits)",
                  f"{'crypto share':14s}{'fixed 0.9':>11s}{'conformal BH':>14s}{'power':>9s}"]
        for k, v in r["fdp"].items():
            fx, cf = v["fixed"], v["bh"]
            lines.append(f"{k:14s}{_f(fx['mean_fdp'] if fx else None):>11s}{_f(cf['mean_fdp'] if cf else None):>14s}"
                         f"{_f(cf['mean_power'] if cf else None):>9s}")
        lines.append(f"conformal selection at alpha = {r['alpha']}")
    return "\n".join(lines)
