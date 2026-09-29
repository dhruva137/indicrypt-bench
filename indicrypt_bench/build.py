"""Compile IndiCrypt-Bench v0: every labelled source file x every toolchain x every optimisation level.

Each source file is compiled to its own object file with symbols kept, so every function in the object
can be traced back to the file (and so to the label) it came from. Objects are *not* linked: linking
adds libc and start-up code whose labels we do not control. Stripping is simulated at feature time by
never reading names; names are used only to attach labels and to group duplicates.

Toolchains:
    clang-x64     zig cc (clang) -> x86_64-linux-musl ELF
    clang-a64     zig cc (clang) -> aarch64-linux-musl ELF
    clang-arm32   zig cc (clang) -> arm-linux-musleabihf ELF (A32)
    gcc-x64       MinGW-w64 gcc  -> x86-64 COFF (a second compiler family on the same ISA)

Output: objects/<toolchain>/<opt>/<library>/<flattened path>.o and objects/index.jsonl, one line per
attempted compile (including failures, which are kept visible, not dropped).

    python build.py                 # everything
    python build.py --only clang-x64 --opts O2
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from manifest import ALL_LIBRARIES as LIBRARIES, SKIP, Library

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
EXTRA = HERE / "extra"
OUT = HERE / "objects"

ZIG = [sys.executable, "-m", "ziglang", "cc"]
TOOLCHAINS: dict[str, list[str]] = {
    "clang-x64": ZIG + ["-target", "x86_64-linux-musl"],
    "clang-a64": ZIG + ["-target", "aarch64-linux-musl"],
    "clang-arm32": ZIG + ["-target", "arm-linux-musleabihf", "-marm"],
    "gcc-x64": [shutil.which("gcc") or "gcc"],
}
OPTS = ("O0", "O2", "O3", "Os")  # O3/Os added for v2 (Phase A2): does v1 hold across optimisation levels?
# zig cc turns on UBSan traps in debug builds; they would add non-crypto code to every O0 function.
COMMON = ["-c", "-g0", "-w", "-fno-sanitize=all", "-fno-stack-protector", "-fno-asynchronous-unwind-tables"]


def _matches(rel: str, glob: str) -> bool:
    """Glob match where ``*`` stays inside one directory and ``**/`` spans any number of them."""
    if "**/" in glob:
        head, tail = glob.split("**/", 1)
        return rel.startswith(head) and fnmatch.fnmatch(rel[len(head):].rsplit("/", 1)[-1], tail) \
            and fnmatch.fnmatch(rel, glob.replace("**/", "*"))
    return rel.count("/") == glob.count("/") and fnmatch.fnmatch(rel, glob)


def source_files(lib: Library) -> list[tuple[Path, str, str, str]]:
    """(path, relpath, label, family) for every file the rules select; first matching rule wins."""
    root = SRC / lib.name
    candidates: list[tuple[Path, str]] = []
    for p in root.rglob("*.c"):
        candidates.append((p, p.relative_to(root).as_posix()))
    for extra in lib.extra_sources:
        candidates.append((EXTRA / extra, f"__extra__/{extra}"))
    picked = []
    for path, rel in sorted(candidates, key=lambda c: c[1]):
        if SKIP.search("/" + rel):
            continue
        for rule in lib.rules:
            if _matches(rel, rule.glob):
                picked.append((path, rel, rule.label, rule.family))
                break
    return picked


def compile_one(job: dict) -> dict:
    cmd = list(TOOLCHAINS[job["toolchain"]]) + COMMON + [f"-{job['opt']}"]
    root = SRC / job["library"]
    lib = next(l for l in LIBRARIES if l.name == job["library"])
    cmd += [f"-I{EXTRA / inc}" for inc in lib.extra_include]
    cmd += [f"-I{root / inc}" for inc in lib.include] + [f"-I{root}", f"-I{Path(job['src']).parent}"]
    cmd += [f"-D{d}" for d in lib.defines]
    out = Path(job["obj"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd += [job["src"], "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    job = dict(job)
    job["ok"] = proc.returncode == 0 and out.exists()
    if not job["ok"]:
        job["error"] = (proc.stderr or proc.stdout)[-400:]
    return job


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(TOOLCHAINS))
    ap.add_argument("--opts", nargs="*", default=list(OPTS))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--libs", nargs="*", default=None, help="restrict to these libraries (smoke tests)")
    args = ap.parse_args()

    jobs = []
    for lib in LIBRARIES:
        if args.libs and lib.name not in args.libs:
            continue
        for path, rel, label, family in source_files(lib):
            if label == "exclude":
                continue
            flat = rel.replace("/", "__")[:-2]
            for tc in args.only:
                for opt in args.opts:
                    jobs.append({
                        "library": lib.name, "rel": rel, "label": label, "family": family,
                        "toolchain": tc, "opt": opt, "src": str(path),
                        "obj": str(OUT / tc / opt / lib.name / f"{flat}.o"),
                    })
    before = len(jobs)
    jobs = [j for j in jobs if not Path(j["obj"]).exists()]      # incremental: earlier objects are reused
    print(f"{len(jobs)} compiles ({before - len(jobs)} already built) across {len(args.only)} toolchains x "
          f"{len(args.opts)} opt levels", flush=True)
    OUT.mkdir(exist_ok=True)
    done = ok = 0
    with ThreadPoolExecutor(args.jobs) as pool, open(OUT / "index.jsonl", "a", encoding="utf-8") as idx:
        for res in pool.map(compile_one, jobs):
            done += 1
            ok += res["ok"]
            idx.write(json.dumps(res) + "\n")
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} ({ok} ok)", flush=True)
    print(f"done: {ok}/{done} compiled")


if __name__ == "__main__":
    main()
