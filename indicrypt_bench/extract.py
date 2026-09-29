"""Turn the compiled objects into one labelled function table: indicrypt_bench/functions.csv.gz.

For every object that compiled (objects/index.jsonl), every function is disassembled and described by
engine.binary_ml.features. The label is the source file's label (manifest.py). Dropped, and counted in
extract_report.json rather than silently lost:
    glue        lifecycle/bookkeeping functions inside crypto files (manifest.GLUE_NAME)
    tiny        fewer than features.MIN_INSNS instructions
    duplicate   byte-identical code already seen in the same library, toolchain and opt level
                (static inline helpers compiled into several files)

The function name is kept only as a grouping key ("uid" = library/file/name): all compiled variants of
one source function share a uid, so the experiments can keep them on the same side of every split.

    python extract.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from engine.binary_ml.features import FEATURES, MIN_INSNS, function_vector  # noqa: E402
from engine.binary_ml.functions import functions  # noqa: E402
from manifest import GLUE_NAME  # noqa: E402

ARCH_OF = {"clang-x64": "x86-64", "gcc-x64": "x86-64", "clang-a64": "aarch64", "clang-arm32": "arm32"}
META = ("library", "rel", "label", "family", "toolchain", "opt", "arch", "name", "uid", "n_insns", "code_sha")


def _one(job: dict) -> tuple[list[list], Counter]:
    drops: Counter = Counter()
    rows = []
    try:
        funcs = functions(job["obj"])
    except Exception as exc:  # a parse failure is reported per object, never swallowed silently
        drops[f"parse_error:{type(exc).__name__}"] += 1
        return rows, drops
    for fn in funcs:
        name = fn.name.lstrip("_")
        if job["label"] == "crypto" and GLUE_NAME.search(name):
            drops["glue"] += 1
            continue
        vec, n = function_vector(fn.code, fn.arch, fn.address)
        if n < MIN_INSNS:
            drops["tiny"] += 1
            continue
        sha = hashlib.sha256(fn.code).hexdigest()[:16]
        uid = f"{job['library']}/{job['rel']}/{name}"
        rows.append([job["library"], job["rel"], job["label"], job["family"], job["toolchain"], job["opt"],
                     fn.arch, name, uid, n, sha] + [round(x, 6) for x in vec])
    return rows, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    jobs = [j for j in map(json.loads, open(HERE / "objects" / "index.jsonl", encoding="utf-8")) if j["ok"]]
    drops: Counter = Counter()
    seen: set[tuple] = set()
    kept = 0
    out = HERE / "functions.csv.gz"
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(args.jobs) as pool:
        w = csv.writer(fh)
        w.writerow(list(META) + list(FEATURES))
        for k, (rows, d) in enumerate(pool.map(_one, jobs, chunksize=8), 1):
            drops.update(d)
            for r in rows:
                key = (r[0], r[4], r[5], r[10])       # library, toolchain, opt, code hash
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)
                w.writerow(r)
                kept += 1
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)} objects, {kept} functions", flush=True)
    report = {"objects": len(jobs), "functions_kept": kept, "dropped": dict(drops), "features": len(FEATURES)}
    (HERE / "extract_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
