"""v1.1: the v1 experiment with function-level labels (protocol v2) and a sealed final test set.

WHAT CHANGED FROM v1, AND WHY (the v1 first-run numbers stay in results/v1_detector.json)
    labels   v1 labelled every function in a crypto file as crypto. Mnemocrypt labels at function level,
             and v1's misses were mostly wrappers. Protocol v2 (indicrypt_bench/core_labels.py) keeps a
             crypto-file function only if its source does primitive computation; wrappers are excluded.
    splits   v1's held-out pool was inspected while diagnosing v1, so it is now DEV: model and feature
             choices are made on it. A SEALED set (BearSSL, TweetNaCl, SipHash; brotli, libdeflate,
             yyjson, lodepng) was registered before this script existed and is scored once, at the end.

    train   mbedtls libtomcrypt bcon tiny-aes | lua zstd cjson zlib
    dev     pqclean monocypher micro-ecc      | stb lz4 xxhash kissfft
    sealed  bearssl tweetnacl siphash         | brotli libdeflate yyjson lodepng

Model selection on dev: {random forest, histogram gradient boosting} x {mnemonic, operand, both}, by dev
PR-AUC. The winner is retrained on train only and applied to sealed. Conformal calibration for sealed
uses sealed-pool nulls from a random half of the sealed functions (never the test half), 200 splits.

    python v1_1_detector.py     (CPU only, about two minutes on 6 cores)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_detector import (ALPHAS, BENCH, FEATURE_SETS, OUT, SEED, _commit, conformal_study,  # noqa: E402
                         fdp_power)
from engine.binary_ml.conformal import conformal_evalues, ebh  # noqa: E402
from engine.binary_ml.features import FEATURES  # noqa: E402

TRAIN = {"mbedtls", "libtomcrypt", "bcon", "tiny-aes", "lua", "zstd", "cjson", "zlib"}
DEV = {"pqclean", "monocypher", "micro-ecc", "stb", "lz4", "xxhash", "kissfft"}
SEALED = {"bearssl", "tweetnacl", "siphash", "brotli", "libdeflate", "yyjson", "lodepng"}


def models() -> dict:
    return {
        "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                             n_jobs=6, random_state=SEED),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                      class_weight="balanced", random_state=SEED),
    }


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(BENCH / "functions.csv.gz")
    core = pd.read_csv(BENCH / "core_labels.csv")
    core = core[core.ops >= 0].drop_duplicates(["library", "rel", "name"])
    core_map = {(r.library, r.rel, r.name): int(r.core) for r in core.itertuples()}
    base = df.name.str.split(".").str[0]                  # gcc clones: foo.constprop.0 -> foo
    keys = list(zip(df.library, df.rel, base))
    status = np.array(["noncrypto" if lab == "noncrypto" else
                       ("core" if core_map.get(k) == 1 else "orchestration" if core_map.get(k) == 0 else "unmatched")
                       for lab, k in zip(df.label, keys)])
    counts = {s: int((status == s).sum()) for s in ("core", "orchestration", "unmatched", "noncrypto")}
    df = df[np.isin(status, ("core", "noncrypto"))].copy()
    df["y"] = (df.label == "crypto").astype(int)
    return df, counts


def evaluate(s: np.ndarray, part: pd.DataFrame, rng) -> dict:
    y = part.y.values
    out = {"roc_auc": round(roc_auc_score(y, s), 4), "pr_auc": round(average_precision_score(y, s), 4),
           "per_arch": {a: {"roc_auc": round(roc_auc_score(y[m], s[m]), 4), "pr_auc": round(average_precision_score(y[m], s[m]), 4)}
                        for a in ("x86-64", "aarch64", "arm32") if (m := (part.arch == a).values).any()}}
    f, pw, n = fdp_power(s >= 0.9, y)
    out["fixed@0.9"] = {"fdp": round(f, 4), "tpr": round(pw, 4), "flagged": n}
    groups = part.uid.values
    out["conformal"] = {f"prevalence={'natural' if p is None else p}": conformal_study(s, y, groups, rng, prevalence=p)
                        for p in (None, 0.05, 0.01)}
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df, label_counts = load()
    assert set(df.library) <= TRAIN | DEV | SEALED
    tr, dev, sealed = (df[df.library.isin(s)].reset_index(drop=True) for s in (TRAIN, DEV, SEALED))
    res = {"commit": _commit(), "seed": SEED, "label_protocol": "v2 (core_labels.py, CORE_MIN=3)",
           "label_counts_all_rows": label_counts,
           "rows": {k: {"n": len(v), "crypto": int(v.y.sum()), "noncrypto": int((v.y == 0).sum()),
                        "unique_functions": int(v.uid.nunique())} for k, v in (("train", tr), ("dev", dev), ("sealed", sealed))}}

    # ---- model selection on DEV only
    grid = {}
    for mname, make in models().items():
        for fname, cols in FEATURE_SETS.items():
            clf = make().fit(tr[cols].values, tr.y.values)
            s = clf.predict_proba(dev[cols].values)[:, 1]
            grid[f"{mname}/{fname}"] = {"roc_auc": round(roc_auc_score(dev.y, s), 4),
                                        "pr_auc": round(average_precision_score(dev.y, s), 4),
                                        "per_arch_roc": {a: round(roc_auc_score(dev.y[m], s[m]), 4)
                                                         for a in ("x86-64", "aarch64", "arm32") if (m := (dev.arch == a).values).any()}}
    res["dev_grid"] = grid
    best = max(grid, key=lambda k: grid[k]["pr_auc"])
    res["selected_on_dev"] = best
    mname, fname = best.split("/")
    cols = FEATURE_SETS[fname]
    clf = models()[mname]().fit(tr[cols].values, tr.y.values)
    res["dev"] = evaluate(clf.predict_proba(dev[cols].values)[:, 1], dev, rng)

    # Ablation at the chosen model family: does the operand family add anything on DEV?
    res["dev_ablation_same_model"] = {f: grid[f"{mname}/{f}"] for f in FEATURE_SETS}

    # e-value composition: average the e-values of the mnemonic-only and operand-only detectors.
    # A mean of e-values is an e-value, so e-BH on the average keeps FDR <= alpha under any dependence.
    s_m = models()[mname]().fit(tr[FEATURE_SETS["mnemonic"]].values, tr.y.values).predict_proba(dev[FEATURE_SETS["mnemonic"]].values)[:, 1]
    s_o = models()[mname]().fit(tr[FEATURE_SETS["operand"]].values, tr.y.values).predict_proba(dev[FEATURE_SETS["operand"]].values)[:, 1]
    comp = {}
    y, groups = dev.y.values, dev.uid.values
    uniq = np.unique(groups)
    for a in ALPHAS:
        fd, pw = [], []
        for _ in range(100):
            cal_g = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
            in_cal = np.array([g in cal_g for g in groups])
            cal, test = in_cal & (y == 0), ~in_cal
            e = 0.5 * (conformal_evalues(s_m[test], s_m[cal], a) + conformal_evalues(s_o[test], s_o[cal], a))
            f, p, _ = fdp_power(ebh(e, a), y[test])
            fd.append(f), pw.append(p)
        comp[f"ebh_mean_evalue@{a}"] = {"mean_fdp": round(float(np.mean(fd)), 4), "mean_power": round(float(np.mean(pw)), 4)}
    res["dev_evalue_composition"] = comp

    # Architecture shift on DEV: train on x86-64 only, calibrate on x86-64 nulls, test on each ISA.
    x86 = tr[tr.arch == "x86-64"]
    s_x = models()[mname]().fit(x86[cols].values, x86.y.values).predict_proba(dev[cols].values)[:, 1]
    shift = {}
    for arch in ("x86-64", "aarch64", "arm32"):
        m = (dev.arch == arch).values
        shift[arch] = {"roc_auc": round(roc_auc_score(dev.y[m], s_x[m]), 4),
                       "calibrated_on_x86": conformal_study(s_x, dev.y.values, groups, rng, prevalence=0.05,
                                                            cal_mask_fn=(dev.arch == "x86-64").values, test_mask_fn=m),
                       "recalibrated_on_target": conformal_study(s_x, dev.y.values, groups, rng, prevalence=0.05,
                                                                 cal_mask_fn=m, test_mask_fn=m)}
    res["dev_arch_shift"] = shift

    # ---- SEALED: scored once, with the model chosen above, trained on TRAIN only.
    res["sealed"] = evaluate(clf.predict_proba(sealed[cols].values)[:, 1], sealed, rng)
    sealed_scores = clf.predict_proba(sealed[cols].values)[:, 1]
    sealed = sealed.assign(score=sealed_scores)
    res["sealed_per_library"] = sealed.groupby("library").apply(
        lambda g: pd.Series({"n": len(g), "y": int(g.y.iloc[0]), "mean_score": round(float(g.score.mean()), 3),
                             "at_0.9": int((g.score >= 0.9).sum())}), include_groups=False).reset_index().to_dict(orient="records")
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "v1_1_detector.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("commit", "label_counts_all_rows", "rows", "selected_on_dev", "dev_ablation_same_model", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
