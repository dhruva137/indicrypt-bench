"""Findcrypt3 baseline, attributed to functions the way Mnemocrypt did it.

Findcrypt3 (github.com/polymorf/findcrypt-yara, commit pinned in research/README.md) reports byte
sequences, not functions. Mnemocrypt counted a function as flagged when it cross-references a matched
sequence. In an unlinked ELF object the cross-references are the relocations, so a function is flagged
when:
    (a) a match lies inside the function's own bytes (immediates, inline tables), or
    (b) a relocation inside the function points into a matched range of another section
        (the usual case: code loading an S-box or K-table from .rodata).

ELF objects only (clang-x64, clang-a64, clang-arm32); COFF relocation addends are implicit in the
instruction bytes and are not decoded here, so gcc-x64 is left out of this comparison and says so.

Output: findcrypt_flags.csv  (toolchain, opt, library, rel, name, rules)
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lief
import yara

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.binary_ml.signatures import scan as ntt_scan  # noqa: E402

HERE = Path(__file__).resolve().parent
RULES = HERE / "src" / "findcrypt" / "findcrypt3.rules"
lief.logging.disable()
_rules = None


def _flags(job: dict) -> list[list]:
    global _rules
    if _rules is None:
        _rules = yara.compile(filepath=str(RULES))
    b = lief.ELF.parse(job["obj"])
    if b is None:
        return []
    data = Path(job["obj"]).read_bytes()
    sections = list(b.sections)
    # matched ranges as (section index, start, end, rule)
    ranges = []
    for hit in ntt_scan(data):                          # NTT twiddle tables (Kestrel-style rung)
        for idx, sec in enumerate(sections):
            if sec.offset <= hit.offset < sec.offset + sec.size and sec.size:
                ranges.append((idx, hit.offset - sec.offset, hit.offset - sec.offset + hit.length, f"NTT_{hit.scheme}"))
                break
    for m in _rules.match(data=data):
        for s in m.strings:
            for inst in s.instances:
                off = inst.offset
                for idx, sec in enumerate(sections):
                    if sec.offset <= off < sec.offset + sec.size and sec.size:
                        ranges.append((idx, off - sec.offset, off - sec.offset + inst.matched_length, m.rule))
                        break
    if not ranges:
        return []
    funcs = []
    for sym in b.symbols:
        if sym.type == lief.ELF.Symbol.TYPE.FUNC and sym.size and 0 < sym.shndx < len(sections):
            funcs.append((sym.shndx, sym.value & ~1, (sym.value & ~1) + sym.size, sym.name.lstrip("_")))
    hits: dict[str, set] = {}
    for idx, a, e, rule in ranges:                      # (a) inside the function
        for fidx, fa, fe, name in funcs:
            if fidx == idx and a < fe and e > fa:
                hits.setdefault(name, set()).add(rule)
    name_of_sec = {sec.name: i for i, sec in enumerate(sections)}
    for rel in b.object_relocations:                    # (b) referenced from the function
        if rel.section is None or rel.symbol is None:
            continue
        where = name_of_sec.get(rel.section.name)
        sym = rel.symbol
        tgt_sec = sym.shndx if 0 < sym.shndx < len(sections) else None
        if where is None or tgt_sec is None:
            continue
        target = sym.value + rel.addend
        for idx, a, e, rule in ranges:
            if idx == tgt_sec and a - 8 <= target < e + 8:   # slack for pc-relative addends (-4 on x86-64)
                for fidx, fa, fe, name in funcs:
                    if fidx == where and fa <= rel.address < fe:
                        hits.setdefault(name, set()).add(rule)
    return [[job["toolchain"], job["opt"], job["library"], job["rel"], n, "|".join(sorted(r))]
            for n, r in hits.items()]


def main() -> None:
    jobs = [j for j in map(json.loads, open(HERE / "objects" / "index.jsonl", encoding="utf-8"))
            if j["ok"] and j["toolchain"] != "gcc-x64"]
    n = 0
    with open(HERE / "findcrypt_flags.csv", "w", newline="", encoding="utf-8") as fh, ProcessPoolExecutor(6) as pool:
        w = csv.writer(fh)
        w.writerow(["toolchain", "opt", "library", "rel", "name", "rules"])
        for rows in pool.map(_flags, jobs, chunksize=8):
            w.writerows(rows)
            n += len(rows)
    print(f"{len(jobs)} ELF objects scanned, {n} flagged functions -> findcrypt_flags.csv")


if __name__ == "__main__":
    sys.exit(main())
