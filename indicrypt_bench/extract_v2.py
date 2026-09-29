"""One pass over every compiled object producing all v2 views of every function (Phase A).

Per function, in shards under v2/ (pickle, one list of dicts per shard):
    meta         library, rel, label, family, toolchain, opt, arch, name, uid, code_sha, n_insns
    fvec         the v1 feature vector (74), so v1 and v2 are compared on identical rows
    loops        v1 feature vectors of the (up to) 8 largest loops with >= MIN_INSNS instructions (A1)
    nodes        per-block NODE_FEATURES matrix, float16 (A3/B3)
    cfg, dfg     edge lists (block ids), control flow and block-level def-use (A3/B3)
    tokens       instruction-class tokens, <= 512 (A4)
Drops are counted in extract_v2_report.json, as in extract.py: glue, tiny, duplicate, excluded-name.

    python extract_v2.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from engine.binary_ml.features import MIN_INSNS, decode, vector  # noqa: E402
from engine.binary_ml.functions import functions  # noqa: E402
from engine.binary_ml.graph import build, loop_instructions, node_matrix, tokens  # noqa: E402
from manifest import ALL_LIBRARIES, GLUE_NAME  # noqa: E402

OUT = HERE / "v2"
MAX_LOOPS = 8
_EXCLUDE = {lib.name: re.compile(lib.exclude_names) for lib in ALL_LIBRARIES if lib.exclude_names}


def _one(job: dict) -> tuple[list[dict], Counter]:
    drops: Counter = Counter()
    rows = []
    try:
        funcs = functions(job["obj"])
    except Exception as exc:  # reported per object, never swallowed silently
        drops[f"parse_error:{type(exc).__name__}"] += 1
        return rows, drops
    ex = _EXCLUDE.get(job["library"])
    for fn in funcs:
        name = fn.name.lstrip("_")
        if job["label"] == "crypto" and GLUE_NAME.search(name):
            drops["glue"] += 1
            continue
        if ex is not None and ex.search(name.split(".")[0]):
            drops["excluded_name"] += 1
            continue
        regs: list = []
        ins = decode(fn.code, fn.arch, fn.address, regs=regs)
        if len(ins) < MIN_INSNS:
            drops["tiny"] += 1
            continue
        end = fn.address + len(fn.code)
        g = build(ins, fn.address, end, regs)
        loops = []
        for L in sorted(g.loops, key=lambda loop: -sum(g.blocks[b][1] - g.blocks[b][0] for b in loop))[:MAX_LOOPS]:
            li = loop_instructions(ins, g, L)
            if len(li) >= MIN_INSNS:
                loops.append(vector(li, li[0].addr, li[-1].addr + li[-1].size))
        rows.append({
            "library": job["library"], "rel": job["rel"], "label": job["label"], "family": job["family"],
            "toolchain": job["toolchain"], "opt": job["opt"], "arch": fn.arch, "name": name,
            "uid": f"{job['library']}/{job['rel']}/{name}", "code_sha": hashlib.sha256(fn.code).hexdigest()[:16],
            "n_insns": len(ins),
            "fvec": np.asarray(vector(ins, fn.address, end), dtype=np.float32),
            "loops": np.asarray(loops, dtype=np.float32).reshape(-1, 74),
            "nodes": np.asarray(node_matrix(ins, g), dtype=np.float16),
            "cfg": np.asarray(g.cfg_edges, dtype=np.int16).reshape(-1, 2),
            "dfg": np.asarray(g.dfg_edges, dtype=np.int16).reshape(-1, 2),
            "tokens": np.asarray(tokens(ins), dtype=np.uint8),
        })
    return rows, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--opts", nargs="*", default=None, help="only these optimisation levels")
    ap.add_argument("--skip-libs", nargs="*", default=(), help="leave these libraries out of this pass")
    ap.add_argument("--out", default="v2", help="output directory under indicrypt_bench/")
    args = ap.parse_args()
    global OUT
    OUT = HERE / args.out
    jobs = {}
    for j in map(json.loads, open(HERE / "objects" / "index.jsonl", encoding="utf-8")):
        if j["ok"]:
            jobs[j["obj"]] = j                      # last record per object wins (the index is append-only)
    jobs = [j for j in jobs.values() if j["label"] != "exclude" and Path(j["obj"]).exists()
            and (args.opts is None or j["opt"] in args.opts) and j["library"] not in args.skip_libs]
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("shard_*.pkl"):
        old.unlink()
    drops: Counter = Counter()
    seen: set = set()
    buf: list = []
    shard = kept = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for k, (rows, d) in enumerate(pool.map(_one, jobs, chunksize=8), 1):
            drops.update(d)
            for r in rows:
                key = (r["library"], r["toolchain"], r["opt"], r["code_sha"])
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)
                buf.append(r)
                kept += 1
            if len(buf) >= 5000:
                pickle.dump(buf, open(OUT / f"shard_{shard:03d}.pkl", "wb"), protocol=5)
                shard += 1
                buf = []
            if k % 1000 == 0:
                print(f"  {k}/{len(jobs)} objects, {kept} functions", flush=True)
    if buf:
        pickle.dump(buf, open(OUT / f"shard_{shard:03d}.pkl", "wb"), protocol=5)
    report = {"objects": len(jobs), "functions_kept": kept, "dropped": dict(drops), "shards": shard + 1}
    (OUT / "extract_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
