"""v1 experiment: operand-aware statistical detector + conformal FDR control, on IndiCrypt-Bench v0.

PRE-REGISTERED SPLIT (fixed before any model was trained; library-disjoint, so the test measures
generalisation to implementations the model never saw)
    train      crypto: mbedtls, libtomcrypt, bcon, tiny-aes      non-crypto: lua, zstd, cjson, zlib
    held-out   crypto: pqclean (ML-KEM, ML-DSA, HQC, Falcon, SPHINCS+), monocypher, micro-ecc
               non-crypto: stb (JPEG DCT, fonts), lz4, xxhash, kissfft (fixed-point FFT)
The held-out pool is split at random into calibration and test halves, *by source function* (every
compiled variant of one function lands on the same side), 200 times. Calibration never touches training.

QUESTIONS AND THE MEASUREMENT FOR EACH
    Q1 detection   ROC-AUC / PR-AUC on the held-out pool, mnemonic-only vs +operand features (ablation),
                   per architecture, training on all architectures
    Q2 baselines   Findcrypt3 (ELF only) and the Caballero heuristic, at their own operating points;
                   our model at Mnemocrypt's operating point (confidence >= 0.9), reporting the same
                   metric Mnemocrypt reports: the false share among flagged functions (FDP)
    Q3 guarantee   conformal BH and e-BH at alpha 0.05 / 0.1 / 0.2: mean FDP and power over 200 splits,
                   at the pool's natural prevalence and at 5% and 1% crypto prevalence (real binaries
                   are mostly not cryptography), against the fixed 0.9 threshold
    Q4 shift       train and calibrate on x86-64 only, test on AArch64 / ARM32: does FDP <= alpha
                   survive? then recalibrate with target-architecture nulls and measure again

    python v1_detector.py            (about a minute on 6 cores; CPU only)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.binary_ml.conformal import bh, conformal_evalues, conformal_pvalues, ebh  # noqa: E402
from engine.binary_ml.features import FEATURES, MNEMONIC_FEATURES, OPERAND_FEATURES  # noqa: E402

BENCH = ROOT / "indicrypt_bench"
OUT = ROOT / "results"
SEED = 20260928
N_SPLITS = 200
ALPHAS = (0.05, 0.1, 0.2)
TRAIN_LIBS = {"mbedtls", "libtomcrypt", "bcon", "tiny-aes", "lua", "zstd", "cjson", "zlib"}
HELD_LIBS = {"pqclean", "monocypher", "micro-ecc", "stb", "lz4", "xxhash", "kissfft"}
FEATURE_SETS = {"mnemonic": list(MNEMONIC_FEATURES), "operand": list(OPERAND_FEATURES), "mnemonic+operand": list(FEATURES)}


def _commit() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        # "+dirty" means uncommitted changes to what produces numbers (code, rules, the shipped model), not to
        # result files or prose, which a run itself rewrites.
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "*.py", "*.yaml", "engine/binary_ml/model"],
                               cwd=ROOT).returncode != 0
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model() -> RandomForestClassifier:
    # Mnemocrypt: random forest, 100 trees, SMOTE for imbalance. We use class weights instead of
    # SMOTE (no synthetic functions); this is a stated deviation, not a replication.
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                  n_jobs=6, random_state=SEED)


def fdp_power(sel: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    n_sel = int(sel.sum())
    fdp = float((sel & (y == 0)).sum() / max(1, n_sel))
    power = float((sel & (y == 1)).sum() / max(1, (y == 1).sum()))
    return fdp, power, n_sel


def conformal_study(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                    prevalence: float | None = None, cal_mask_fn=None, test_mask_fn=None) -> dict:
    """Mean FDP / power over random calibration-test splits of the held-out pool."""
    uniq = np.unique(groups)
    out = {f"{proc}@{a}": {"fdp": [], "power": [], "n_sel": []} for a in ALPHAS for proc in ("bh", "ebh")}
    out["fixed@0.9"] = {"fdp": [], "power": [], "n_sel": []}
    for _ in range(N_SPLITS):
        cal_groups = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
        in_cal = np.array([g in cal_groups for g in groups])
        cal = in_cal & (y == 0)
        if cal_mask_fn is not None:
            cal &= cal_mask_fn
        test = ~in_cal
        if test_mask_fn is not None:
            test &= test_mask_fn
        idx = np.nonzero(test)[0]
        if prevalence is not None:
            neg = idx[y[idx] == 0]
            pos = idx[y[idx] == 1]
            k = int(round(prevalence * len(neg) / (1 - prevalence)))
            idx = np.concatenate([neg, rng.choice(pos, size=min(k, len(pos)), replace=False)])
        s, yt = scores[idx], y[idx]
        p = conformal_pvalues(s, scores[cal])
        for a in ALPHAS:
            for proc, sel in (("bh", bh(p, a)), ("ebh", ebh(conformal_evalues(s, scores[cal], a), a))):
                f, pw, n = fdp_power(sel, yt)
                r = out[f"{proc}@{a}"]
                r["fdp"].append(f), r["power"].append(pw), r["n_sel"].append(n)
        f, pw, n = fdp_power(s >= 0.9, yt)
        r = out["fixed@0.9"]
        r["fdp"].append(f), r["power"].append(pw), r["n_sel"].append(n)
    summary = {}
    for k, r in out.items():
        fd = np.array(r["fdp"])
        summary[k] = {"mean_fdp": round(float(fd.mean()), 4), "se_fdp": round(float(fd.std() / np.sqrt(len(fd))), 4),
                      "p90_fdp": round(float(np.quantile(fd, 0.9)), 4),
                      "mean_power": round(float(np.mean(r["power"])), 4),
                      "mean_selected": round(float(np.mean(r["n_sel"])), 1)}
    return summary


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(BENCH / "functions.csv.gz")
    df["y"] = (df.label == "crypto").astype(int)
    assert set(df.library) == TRAIN_LIBS | HELD_LIBS, "every library must be assigned to exactly one side"
    train, held = df[df.library.isin(TRAIN_LIBS)], df[df.library.isin(HELD_LIBS)].reset_index(drop=True)
    res: dict = {
        "commit": _commit(), "seed": SEED, "splits": N_SPLITS, "host": platform.processor() or platform.machine(),
        "data": {
            "train_rows": len(train), "held_rows": len(held),
            "train_by_label": train.label.value_counts().to_dict(), "held_by_label": held.label.value_counts().to_dict(),
            "held_unique_functions": int(held.uid.nunique()),
        },
    }

    # Q1: ablation, all architectures in training, per-architecture evaluation.
    scores = {}
    q1 = {}
    for name, cols in FEATURE_SETS.items():
        clf = model().fit(train[cols].values, train.y.values)
        s = clf.predict_proba(held[cols].values)[:, 1]
        scores[name] = s
        per_arch = {}
        for arch in ("x86-64", "aarch64", "arm32"):
            m = (held.arch == arch).values
            per_arch[arch] = {"roc_auc": round(roc_auc_score(held.y[m], s[m]), 4),
                              "pr_auc": round(average_precision_score(held.y[m], s[m]), 4)}
        q1[name] = {"roc_auc": round(roc_auc_score(held.y, s), 4), "pr_auc": round(average_precision_score(held.y, s), 4),
                    "per_arch": per_arch}
        if name == "mnemonic+operand":
            imp = sorted(zip(cols, clf.feature_importances_), key=lambda kv: -kv[1])[:15]
            q1["top_features"] = [(k, round(float(v), 4)) for k, v in imp]
    res["q1_detection"] = q1

    # Per-library view of the full model at the Mnemocrypt operating point (where do mistakes live?).
    full = scores["mnemonic+operand"]
    held["score"] = full
    by_lib = held.groupby("library").apply(
        lambda g: pd.Series({"n": len(g), "label": g.label.iloc[0], "flagged_at_0.9": int((g.score >= 0.9).sum()),
                             "mean_score": round(float(g.score.mean()), 3)}), include_groups=False)
    res["per_library_at_0.9"] = by_lib.reset_index().to_dict(orient="records")

    # Q2: baselines on the same held-out rows.
    y = held.y.values
    q2 = {}
    for name in FEATURE_SETS:
        f, pw, n = fdp_power(scores[name] >= 0.9, y)
        fpr = float(((scores[name] >= 0.9) & (y == 0)).sum() / max(1, (y == 0).sum()))
        q2[f"rf_{name}@0.9"] = {"fdp": round(f, 4), "tpr": round(pw, 4), "fpr": round(fpr, 4), "flagged": n}
    cab = held["caballero_adjusted"].values
    thr = float(np.quantile(train.loc[train.y == 1, "caballero_adjusted"], 0.10))  # keeps 90% of train crypto
    sel = cab >= thr
    f, pw, n = fdp_power(sel, y)
    q2["caballero_adjusted"] = {"threshold_from_train": round(thr, 4), "fdp": round(f, 4), "tpr": round(pw, 4),
                                "fpr": round(float((sel & (y == 0)).sum() / max(1, (y == 0).sum())), 4), "flagged": n}
    fc = pd.read_csv(BENCH / "findcrypt_flags.csv")
    fc_keys = set(zip(fc.toolchain, fc.opt, fc.library, fc.rel, fc.name.str.lstrip("_")))
    elf = held.toolchain != "gcc-x64"
    sel = np.array([(r.toolchain, r.opt, r.library, r.rel, r.name) in fc_keys for r in held.itertuples()])
    f, pw, n = fdp_power(sel[elf.values], y[elf.values])
    q2["findcrypt3_elf_only"] = {"fdp": round(f, 4), "tpr": round(pw, 4),
                                 "fpr": round(float((sel & elf.values & (y == 0)).sum() / max(1, (y[elf.values] == 0).sum())), 4),
                                 "flagged": n, "note": "gcc-x64 (COFF) excluded; attribution by relocation, as Mnemocrypt did"}
    f_rf, pw_rf, n_rf = fdp_power((full >= 0.9)[elf.values], y[elf.values])
    q2["rf_mnemonic+operand@0.9_same_elf_rows"] = {"fdp": round(f_rf, 4), "tpr": round(pw_rf, 4), "flagged": n_rf}
    res["q2_baselines"] = q2

    # Q3: conformal guarantee on the held-out pool, natural and realistic prevalence.
    groups = held.uid.values
    res["q3_conformal"] = {
        f"prevalence={'natural' if p is None else p}": conformal_study(full, y, groups, rng, prevalence=p)
        for p in (None, 0.05, 0.01)
    }
    res["q3_conformal_mnemonic_only_5pct"] = conformal_study(scores["mnemonic"], y, groups, rng, prevalence=0.05)

    # Q4: architecture shift. Train on x86-64 only; calibrate on x86-64 nulls; test on each ISA.
    x86_train = train[train.arch == "x86-64"]
    clf = model().fit(x86_train[list(FEATURES)].values, x86_train.y.values)
    s = clf.predict_proba(held[list(FEATURES)].values)[:, 1]
    q4 = {"train": "x86-64 only (clang + gcc)"}
    for arch in ("x86-64", "aarch64", "arm32"):
        is_arch = (held.arch == arch).values
        q4[arch] = {
            "roc_auc": round(roc_auc_score(y[is_arch], s[is_arch]), 4),
            "calibrated_on_x86": conformal_study(s, y, groups, rng, prevalence=0.05,
                                                 cal_mask_fn=(held.arch == "x86-64").values, test_mask_fn=is_arch),
            "recalibrated_on_target": conformal_study(s, y, groups, rng, prevalence=0.05,
                                                      cal_mask_fn=is_arch, test_mask_fn=is_arch),
        }
    res["q4_arch_shift"] = q4
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v1_detector.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("commit", "data", "q1_detection", "q2_baselines", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
