"""Function-level labels for crypto files: core computation vs orchestration (labelling protocol v2).

WHY
---
File-level labels (manifest.py) call every function in aes.c cryptographic, including wrappers that only
call other functions. Mnemocrypt labelled at function level (173 crypto functions out of 19,482 in its
OpenSSL build), so wrappers were not crypto there. v1's first run showed the same thing from the other
side: the detector scored core primitives high (Keccak-f 0.999, SHA-2 compression 0.95) and wrappers
low, and the wrappers were counted as misses.

RULE v2 (applied identically to train, dev and sealed libraries, before any model saw these labels)
    ops(f) = arithmetic and bitwise operators in f's preprocessed body, {+ - * / % ^ & | ~ << >>} and
             their compound assignments, NOT counting for-loop headers, array subscripts or ++/--
             + ops of every *static* function defined in the same file that f calls (transitively):
             the compiler inlines those, so their work is f's work in the binary.
    CORE if ops(f) >= CORE_MIN; otherwise ORCHESTRATION, excluded from training and evaluation.
Rule v1 (bitwise and multiplicative only, own body only) was replaced after it labelled ML-KEM's ntt()
(butterflies are + and -, the multiply is in static fqmul) and montgomery_reduce() (3 operators) as
orchestration, and the threshold went from 4 to 3 for the same reason (ntt() scores 3). Both changes
were made on label inspection only, before v1.1 training; the rule is frozen from here.
Preprocessing first (zig cc -E) expands macros, so round macros count. The rule reads *source*, never the
compiled binary, so it cannot leak detector features into labels. Functions in non-crypto files stay
negative whatever they contain: an arithmetic-heavy compressor is exactly the hard negative we want.

Output: core_labels.csv (library, rel, name, ops, core)
"""
from __future__ import annotations

import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tree_sitter_language_pack as tslp

from build import SRC, source_files
from manifest import ALL_LIBRARIES

HERE = Path(__file__).resolve().parent
CORE_MIN = 3   # smallest real kernels (one NTT butterfly, one Montgomery step) have 3-4 operators; frozen
OPS = {"^", "<<", ">>", "&", "|", "*", "%", "/", "+", "-",
       "^=", "<<=", ">>=", "&=", "|=", "*=", "%=", "/=", "+=", "-="}


def _preprocess(lib, path: Path) -> str | None:
    root = SRC / lib.name
    cmd = [sys.executable, "-m", "ziglang", "cc", "-E", "-P", "-w", "-target", "x86_64-linux-musl"]
    cmd += [f"-I{HERE / 'extra' / i}" for i in lib.extra_include]
    cmd += [f"-I{root / i}" for i in lib.include] + [f"-I{root}", f"-I{path.parent}"]
    cmd += [f"-D{d}" for d in lib.defines] + [str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.stdout if proc.returncode == 0 else None


def _count(node) -> tuple[int, set[str]]:
    """(operator count, names of functions called) for one function body."""
    n = 0
    calls: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        t = cur.type
        if t in ("subscript_expression",):          # a[i + 1]: index arithmetic is not computation
            arg = cur.child_by_field_name("argument")
            if arg is not None:
                stack.append(arg)
            continue
        if t == "for_statement":                    # loop header arithmetic is not computation
            body = cur.child_by_field_name("body")
            if body is not None:
                stack.append(body)
            continue
        if t == "update_expression":
            continue
        if t in ("binary_expression", "assignment_expression"):
            op = cur.child_by_field_name("operator")
            if op is not None and op.type in OPS:
                n += 1
        elif t == "unary_expression":
            op = cur.child_by_field_name("operator")
            if op is not None and op.type in ("~", "-"):
                n += 1
        elif t == "call_expression":
            fn = cur.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                calls.add(fn.text.decode())
        stack.extend(cur.children)
    return n, calls


def _name(decl) -> str | None:
    while decl is not None and decl.type not in ("identifier", "field_identifier"):
        nxt = decl.child_by_field_name("declarator")
        if nxt is None:
            for ch in decl.children:
                if ch.type in ("identifier", "function_declarator", "pointer_declarator", "parenthesized_declarator"):
                    nxt = ch
                    break
        decl = nxt
    return decl.text.decode() if decl is not None else None


def label_file(job) -> list[list]:
    lib, path, rel = job
    text = _preprocess(lib, path)
    if text is None:
        return [[lib.name, rel, "", -1, "preprocess_failed"]]
    parser = tslp.get_parser("c")
    tree = parser.parse(text.encode("utf-8", errors="replace"))
    own: dict[str, int] = {}
    calls: dict[str, set[str]] = {}
    static: set[str] = set()
    for node in tree.root_node.children:
        if node.type != "function_definition":
            continue
        name = _name(node.child_by_field_name("declarator"))
        body = node.child_by_field_name("body")
        if not name or body is None:
            continue
        own[name], calls[name] = _count(body)
        if any(ch.type == "storage_class_specifier" and ch.text.decode() == "static" for ch in node.children):
            static.add(name)

    def inclusive(name: str, seen: frozenset) -> int:
        total = own[name]
        for callee in calls[name]:
            if callee in static and callee in own and callee not in seen:
                total += inclusive(callee, seen | {callee})
        return total

    return [[lib.name, rel, name, (ops := inclusive(name, frozenset({name}))), int(ops >= CORE_MIN)]
            for name in own]


def main() -> None:
    jobs = [(lib, path, rel) for lib in ALL_LIBRARIES for path, rel, label, _ in source_files(lib) if label == "crypto"]
    with open(HERE / "core_labels.csv", "w", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(6) as pool:
        w = csv.writer(fh)
        w.writerow(["library", "rel", "name", "ops", "core"])
        n = 0
        for rows in pool.map(label_file, jobs):
            w.writerows(rows)
            n += len(rows)
    print(f"{len(jobs)} crypto files, {n} function definitions labelled -> core_labels.csv")


if __name__ == "__main__":
    main()
