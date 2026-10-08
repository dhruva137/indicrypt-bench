"""The library-disjoint splits. Libraries never cross splits."""
from __future__ import annotations

from .manifest import ALL_LIBRARIES, LIBRARIES, SEALED, SEALED_B, SEALED_C

TRAIN = frozenset({"mbedtls", "libtomcrypt", "bcon", "tiny-aes", "lua", "zstd", "cjson", "zlib"})
DEV = frozenset({"pqclean", "monocypher", "micro-ecc", "stb", "lz4", "xxhash", "kissfft"})
SPLITS: dict[str, frozenset[str]] = {
    "train": TRAIN,
    "dev": DEV,
    "sealed": frozenset(l.name for l in SEALED),
    "sealed-b": frozenset(l.name for l in SEALED_B),
    "sealed-c": frozenset(l.name for l in SEALED_C),
}
# miniz is kept in the pool as an additional compression negative and belongs to no scored split.
UNASSIGNED = frozenset(l.name for l in ALL_LIBRARIES) - frozenset().union(*SPLITS.values())
CRYPTO_LIBS = frozenset(l.name for l in ALL_LIBRARIES if l.role == "crypto")


def split_of(library: str) -> str | None:
    for name, libs in SPLITS.items():
        if library in libs:
            return name
    return None


def libraries(split: str) -> frozenset[str]:
    if split == "all":
        return frozenset(l.name for l in ALL_LIBRARIES)
    if split == "sealed-all":
        return SPLITS["sealed"] | SPLITS["sealed-b"] | SPLITS["sealed-c"]
    return SPLITS[split]


__all__ = ["TRAIN", "DEV", "SPLITS", "UNASSIGNED", "CRYPTO_LIBS", "split_of", "libraries", "LIBRARIES"]
