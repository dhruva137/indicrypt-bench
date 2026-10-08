"""Load the manifest and the labelled function tables, and resolve labels for arbitrary functions."""
from __future__ import annotations

import fnmatch
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .manifest import ALL_LIBRARIES, SKIP, Library

PKG = Path(__file__).resolve().parent
ISAS = ("x86-64", "aarch64", "arm32")


def manifest() -> dict[str, Library]:
    """Library name -> pinned source, commit and per-file labelling rules."""
    return {lib.name: lib for lib in ALL_LIBRARIES}


def core_labels_path(data_dir: str | Path | None = None) -> Path:
    for base in ([Path(data_dir)] if data_dir else []) + [PKG]:
        p = base / "core_labels.csv"
        if p.exists():
            return p
    raise FileNotFoundError("core_labels.csv not found; run `indicrypt-bench fetch`")


@lru_cache(maxsize=4)
def _core_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.ops >= 0].drop_duplicates(["library", "rel", "name"]).copy()  # ops -1 marks preprocess failures
    df["core"] = df["core"].astype(int)
    return df


def core_labels(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Source-derived function labels for every function in a crypto file (columns library, rel, name, ops, core)."""
    return _core_labels(str(core_labels_path(data_dir))).copy()


def _matches(rel: str, glob: str) -> bool:
    if "**/" in glob:
        head, tail = glob.split("**/", 1)
        return rel.startswith(head) and fnmatch.fnmatch(rel[len(head):].rsplit("/", 1)[-1], tail) \
            and fnmatch.fnmatch(rel, glob.replace("**/", "*"))
    return rel.count("/") == glob.count("/") and fnmatch.fnmatch(rel, glob)


def file_label(library: str, rel: str) -> str | None:
    """Manifest label of a source file: crypto, noncrypto, exclude, or None when no rule matches."""
    lib = manifest().get(library)
    if lib is None or SKIP.search("/" + rel):
        return None
    for rule in lib.rules:
        if _matches(rel, rule.glob):
            return rule.label
    return None


@lru_cache(maxsize=4)
def _core_map(path: str) -> dict:
    return {(r.library, r.rel, r.name): int(r.core) for r in _core_labels(path).itertuples()}


def resolve_labels(df: pd.DataFrame, data_dir: str | Path | None = None) -> pd.Series:
    """1 = core crypto, 0 = non-crypto, NaN = orchestration/glue/excluded/unknown (dropped from scoring).

    Frozen rule: a function is crypto if its file implements a primitive and its source body (with the static
    functions it calls) does at least 3 arithmetic or bitwise operations. Functions in non-crypto files are
    negative. gcc clones (foo.constprop.0) are matched through their base name.
    """
    cmap = _core_map(str(core_labels_path(data_dir)))
    base = df["name"].astype(str).str.split(".").str[0]
    out = []
    for lib, rel, nm in zip(df["library"], df["rel"], base):
        lab = file_label(lib, rel)
        if lab == "noncrypto":
            out.append(0.0)
        elif lab == "crypto":
            core = cmap.get((lib, rel, nm))
            out.append(1.0 if core == 1 else float("nan"))
        else:
            out.append(float("nan"))
    return pd.Series(out, index=df.index, name="y")
