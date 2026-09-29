"""Clone every IndiCrypt-Bench library into src/ at exactly the commit pinned in manifest.py.

Git sources are fetched at the pinned commit and verified. TweetNaCl is a two-file tarball-less release,
so it is downloaded and checked against the SHA-256 recorded in the manifest. The findcrypt3 rules used
by the baseline are fetched the same way.

    python fetch_sources.py
"""
from __future__ import annotations

import hashlib
import subprocess
import urllib.request
from pathlib import Path

from manifest import ALL_LIBRARIES

SRC = Path(__file__).resolve().parent / "src"
FINDCRYPT = ("https://github.com/polymorf/findcrypt-yara.git", "044644b9c52ae3e7b2305bac15b0c12c9e31a282")


def _git(url: str, commit: str, dest: Path) -> None:
    if not dest.exists():
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise SystemExit(f"{dest.name}: expected {commit}, got {head}")


def main() -> None:
    SRC.mkdir(exist_ok=True)
    for lib in ALL_LIBRARIES:
        dest = SRC / lib.name
        if lib.commit.startswith("sha256:"):
            dest.mkdir(exist_ok=True)
            for fname in ("tweetnacl.c", "tweetnacl.h"):
                urllib.request.urlretrieve(lib.repo + fname, dest / fname)
            digest = hashlib.sha256((dest / "tweetnacl.c").read_bytes()).hexdigest()
            if f"sha256:{digest}" != lib.commit:
                raise SystemExit(f"tweetnacl.c hash mismatch: {digest}")
        else:
            _git(lib.repo, lib.commit, dest)
        print(f"{lib.name:12s} {lib.commit}")
    _git(*FINDCRYPT, SRC / "findcrypt")
    print("findcrypt    ", FINDCRYPT[1])


if __name__ == "__main__":
    main()
