"""indicrypt-bench command line: info, score, fetch."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__
from .data import manifest
from .splits import SPLITS, UNASSIGNED, split_of

REPO = "dhruva137/indicrypt-bench"


def cmd_info(args) -> int:
    m = manifest()
    crypto = [lib for lib in m.values() if lib.role == "crypto"]
    print(f"IndiCrypt-Bench {__version__}: {len(m)} libraries ({len(crypto)} cryptographic, "
          f"{len(m) - len(crypto)} non-cryptographic), 4 toolchains, O0/O2/O3/Os, 3 sealed sets")
    for name, libs in SPLITS.items():
        c = sorted(x for x in libs if m[x].role == "crypto")
        n = sorted(x for x in libs if m[x].role != "crypto")
        print(f"  {name:9s} crypto: {', '.join(c) or '-'}")
        print(f"  {'':9s} negatives: {', '.join(n) or '-'}")
    if UNASSIGNED:
        print(f"  pool only (no scored split): {', '.join(sorted(UNASSIGNED))}")
    if args.libraries:
        print(f"\n{'library':12s}{'split':10s}{'role':10s}commit")
        for lib in m.values():
            print(f"{lib.name:12s}{(split_of(lib.name) or '-'):10s}{lib.role:10s}{lib.commit}")
    return 0


def cmd_score(args) -> int:
    import pandas as pd

    from .score import format_report, score
    try:
        df = pd.read_csv(args.predictions)
        res = score(df, split=args.split, alpha=args.alpha, n_splits=args.splits, data_dir=args.data_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2) if args.json else format_report(res))
    return 0


def _release_asset(tag: str | None):
    url = f"https://api.github.com/repos/{REPO}/releases/" + (f"tags/{tag}" if tag else "latest")
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "indicrypt-bench"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for a in rel.get("assets", []):
        if a["name"].startswith("indicrypt-bench-data") and a["name"].endswith(".tar.gz"):
            return a["browser_download_url"], a["name"]
    return None


def cmd_fetch(args) -> int:
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    asset = None if args.rebuild else _release_asset(args.tag)
    if asset:
        url, name = asset
        print(f"downloading {name} from the GitHub release")
        tgz = dest / name
        urllib.request.urlretrieve(url, tgz)
        shutil.unpack_archive(tgz, dest)
        tgz.unlink()
        print(f"data artifacts unpacked into {dest}")
        return 0
    print("No released data artifact is available (or --rebuild was given): rebuilding with the repository scripts.")
    repo = dest / "indicrypt-bench"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", f"https://github.com/{REPO}.git", str(repo)], check=True)
    bench = repo / "indicrypt_bench"
    stages = {
        "sources": [["fetch_sources.py"]],
        "build": [["build.py"]],
        "extract": [["extract_v2.py", "--jobs", str(args.jobs)], ["extract.py"], ["core_labels.py"]],
    }
    order = list(stages)
    print(f"requirements: pip install -r {repo / 'requirements.txt'}  (ziglang, capstone, lief, yara-python, ...)")
    for stage in order[: order.index(args.until) + 1]:
        for argv in stages[stage]:
            print(f"$ python {' '.join(argv)}")
            subprocess.run([sys.executable, *argv], cwd=bench, check=True)
    print(f"done; tables are in {bench}. Pass --data-dir {bench} to `indicrypt-bench score`.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="indicrypt-bench", description="IndiCrypt-Bench: crypto detection in stripped binaries.")
    ap.add_argument("--version", action="version", version=f"indicrypt-bench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="show the benchmark manifest and splits")
    p.add_argument("--libraries", action="store_true", help="list every library with its split and pinned commit")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("score", help="score a detector's per-function predictions")
    p.add_argument("predictions", help="CSV with library, rel, name, arch, score [, label]")
    p.add_argument("--split", default="all", choices=["all", "sealed-all", *SPLITS])
    p.add_argument("--alpha", type=float, default=0.1, help="conformal FDR level (default 0.1)")
    p.add_argument("--splits", type=int, default=200, help="calibration/test resamples (default 200)")
    p.add_argument("--data-dir", default=None, help="directory holding core_labels.csv (default: bundled copy)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("fetch", help="download the released data artifacts, or rebuild them from the pinned sources")
    p.add_argument("--dest", default="indicrypt-data")
    p.add_argument("--tag", default=None, help="release tag (default: latest)")
    p.add_argument("--rebuild", action="store_true", help="skip the release download and rebuild locally")
    p.add_argument("--until", choices=["sources", "build", "extract"], default="extract",
                   help="in rebuild mode, stop after this stage (default: all stages)")
    p.add_argument("--jobs", type=int, default=6)
    p.set_defaults(fn=cmd_fetch)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
