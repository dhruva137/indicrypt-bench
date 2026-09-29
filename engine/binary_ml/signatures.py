"""Signature rung for post-quantum lattice schemes: NTT twiddle tables (Kestrel-style, arXiv 2608.25122).

ML-KEM and ML-DSA multiply polynomials with a number-theoretic transform whose twiddle factors (powers of a
fixed root of unity, bit-reversed) are stored as a constant table. The table is derived here from the public
parameters (FIPS 203 / FIPS 204), in every representation implementations use: plain residues, centred
residues, Montgomery form and centred Montgomery form, as 16- or 32-bit little-endian integers.

A window of W consecutive aligned integers is a hit when at least MATCH of them belong to a scheme's
fingerprint set and they cover at least DISTINCT different table entries (multiset matching, so reordered or
partially stored tables still match). The chance that random data does this is negligible (fingerprint sets
cover < 1% of the 16-bit space and 0.0004% of the 32-bit space); the empirical rate on non-crypto code is
measured in research/, not assumed.

Stated limit, as in Kestrel: an implementation that computes its twiddles at start-up stores no table and is
invisible to this rung. That is the case the learned rung exists for.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WINDOW = 32
MATCH = 24
DISTINCT = 16


def _bitrev(i: int, bits: int) -> int:
    return int(f"{i:0{bits}b}"[::-1], 2)


def _centre(v: int, q: int) -> int:
    return v - q if v > q // 2 else v


@lru_cache(maxsize=None)
def fingerprints() -> dict[str, tuple[int, frozenset]]:
    """scheme -> (byte width, set of integer values as stored)."""
    out = {}
    for scheme, q, zeta, bits, width, r in (("ML-KEM", 3329, 17, 7, 2, 1 << 16), ("ML-DSA", 8380417, 1753, 8, 4, 1 << 32)):
        n = 1 << bits
        plain = [pow(zeta, _bitrev(i, bits), q) for i in range(n)]
        mont = [(z * r) % q for z in plain]
        vals = set()
        for form in (plain, mont):
            for v in form:
                for x in (v, _centre(v, q)):
                    vals.add(x & ((1 << (8 * width)) - 1))       # stored as unsigned of that width
        vals.discard(0)
        vals.discard(1)
        out[scheme] = (width, frozenset(vals))
    return out


@dataclass(frozen=True)
class TableHit:
    scheme: str
    offset: int          # byte offset in the scanned blob
    length: int          # bytes covered by the matching window
    matched: int


def scan(blob: bytes) -> list[TableHit]:
    """NTT table hits in a byte blob (a section, a file). Overlapping windows are merged per scheme."""
    hits = []
    for scheme, (width, fp) in fingerprints().items():
        dtype = np.uint16 if width == 2 else np.uint32
        fp_arr = np.fromiter(fp, dtype=np.int64)
        for align in range(width):
            usable = (len(blob) - align) // width
            if usable < WINDOW:
                continue
            vals = np.frombuffer(blob, dtype=dtype, count=usable, offset=align).astype(np.int64)
            member = np.isin(vals, fp_arr)
            counts = np.convolve(member.astype(np.int32), np.ones(WINDOW, dtype=np.int32), mode="valid")
            spans: list[list[int]] = []
            for i in np.nonzero(counts >= MATCH)[0]:
                window = vals[i:i + WINDOW][member[i:i + WINDOW]]
                if len(np.unique(window)) < DISTINCT:
                    continue
                if spans and i <= spans[-1][1]:
                    spans[-1][1] = int(i) + WINDOW
                else:
                    spans.append([int(i), int(i) + WINDOW])
            for a, e in spans:
                idx = np.nonzero(member[a:e])[0]                   # trim to the matching entries themselves
                first, last = a + int(idx[0]), a + int(idx[-1]) + 1
                hits.append(TableHit(scheme, align + first * width, (last - first) * width, int(len(idx))))
    return hits


def references_table(refs: set[int], table_ranges: list[tuple[int, int, int]]) -> bool:
    """True when any referenced address falls in a table, allowing a lead-in of two entries.

    ``table_ranges`` holds (start address, end address, entry width). The lead-in exists because code points
    at a table's first element, and ML-DSA's first twiddle is zeta^0 = 0, which is not in the fingerprint set.
    """
    return any(a - 2 * w <= r < e for r in refs for a, e, w in table_ranges)
