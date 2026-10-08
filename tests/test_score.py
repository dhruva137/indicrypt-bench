import numpy as np
import pandas as pd
import pytest

from indicrypt_bench import __version__
from indicrypt_bench.cli import main
from indicrypt_bench.conformal import bh, conformal_pvalues
from indicrypt_bench.data import core_labels, file_label, manifest
from indicrypt_bench.score import score
from indicrypt_bench.splits import SPLITS, UNASSIGNED


def test_manifest_counts():
    m = manifest()
    assert len(m) == 30
    assert sum(1 for l in m.values() if l.role == "crypto") == 15
    assert len(SPLITS["sealed"] | SPLITS["sealed-b"] | SPLITS["sealed-c"]) == 14
    assert UNASSIGNED == {"miniz"}


def test_labels_load_and_resolve():
    cl = core_labels()
    assert {"library", "rel", "name", "ops", "core"} <= set(cl.columns)
    assert file_label("zlib", "adler32.c") == "noncrypto"
    assert file_label("tiny-aes", "aes.c") == "crypto"


def test_conformal_pvalues_and_bh():
    p = conformal_pvalues(np.array([0.99, 0.1]), np.linspace(0, 0.5, 50))
    assert p[0] < p[1]
    assert bh(np.array([0.001, 0.9]), 0.1).tolist() == [True, False]


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        crypto = i % 4 == 0
        for arch in ("x86-64", "aarch64"):
            rows.append(dict(library="bearssl" if crypto else "brotli", rel="src/x.c", name=f"f{i}", arch=arch,
                             label=int(crypto), score=float(np.clip(rng.normal(0.8 if crypto else 0.2, 0.15), 0, 1))))
    return pd.DataFrame(rows)


def test_score_synthetic():
    r = score(_synthetic(), n_splits=20)
    assert r["overall"]["roc_auc"] > 0.95
    assert set(r["per_isa"]) == {"x86-64", "aarch64"}
    assert set(r["fdp"]) == {"natural", "0.05", "0.01"}
    assert r["fdp"]["0.05"]["bh"]["mean_fdp"] <= 0.35


def test_missing_columns():
    with pytest.raises(ValueError):
        score(pd.DataFrame({"score": [1.0]}))


def test_cli_info(capsys):
    assert main(["info"]) == 0
    assert "30 libraries" in capsys.readouterr().out
    assert __version__ == "0.1.0"


def test_core_labels_are_numeric():
    cl = core_labels()
    assert set(cl.core.unique()) == {0, 1}
