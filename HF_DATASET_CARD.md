---
# DRAFT dataset card for Hugging Face Datasets. Not published. Review the licence table before upload.
pretty_name: IndiCrypt-Bench
license: apache-2.0
language: []
task_categories:
  - tabular-classification
tags:
  - cryptography
  - binary-analysis
  - reverse-engineering
  - benchmark
  - conformal-prediction
size_categories:
  - 100K<n<1M
---

# IndiCrypt-Bench: labelled functions

Author: **Dhruva P Gowda** (solo). Code and scorer: https://github.com/dhruva137/indicrypt-bench
(`pip install indicrypt-bench`). Part of Paper To Anything: https://papertoanything.com/products/indicrypt-bench/

## What this is

A library-disjoint, cross-architecture benchmark for the question *given a stripped binary, which of its functions
are cryptographic?* The dataset is the table of **128,288 labelled functions** (after dropping tiny, glue and
byte-identical functions; every drop is counted in `extract_v2_all_report.json`) recovered from **16,711 compiled
objects**: 30 open-source libraries at pinned commits (15 cryptographic, 15 non-cryptographic hard negatives), built
with 4 toolchains (clang x86-64, gcc x86-64 via MinGW, clang AArch64, clang ARM32) at O0, O2, O3 and Os.

Each row is one function in one compiled variant. Columns include `library`, `rel` (source file), `name`, `uid`
(`library/rel/name`, shared by every compiled variant of a source function), `toolchain`, `opt`, `arch`
(`x86-64`, `aarch64`, `arm32`), `label`, `family`, `n_insns`, `code_sha`, and the 74-feature vector (mnemonic and
operand counts). Per-function loops, CFG and data-flow edges, and opcode tokens are provided as v2 shards.

The source-derived label table (`core_labels.csv`: library, rel, name, ops, core) ships inside the `indicrypt-bench`
wheel. The compiled-function table and shards are the artifacts intended for this dataset repository.
No source code is redistributed: the dataset contains function names, labels, instruction statistics and hashes.

## Splits

Libraries never cross splits, so a score on a sealed split measures generalisation to code the detector has not seen.

| Split | Role | Cryptographic libraries | Hard negatives |
|---|---|---|---|
| train | fit models | mbedTLS, libtomcrypt, B-Con crypto-algorithms, tiny-AES-c | Lua, zstd, cJSON, zlib |
| dev | model selection only | PQClean, Monocypher, micro-ecc | stb, lz4, xxHash, kissfft |
| sealed | scored once | BearSSL, TweetNaCl, SipHash | brotli, libdeflate, yyjson, lodepng |
| sealed-b | scored once | libsodium, wolfSSL | SQLite, libpng |
| sealed-c | scored once | BLAKE3, Argon2, tiny_sha3 | (negatives from sealed and sealed-b) |

Eight cryptographic libraries are sealed. Each sealed set was registered before training and is scored once.
miniz is kept in the pool as an additional compression negative and belongs to no scored split.

## Labelling rule (frozen, v2)

A function is **crypto** if its source file implements a cryptographic primitive **and** its preprocessed body,
counting the static functions it calls (which the compiler inlines), performs at least 3 arithmetic or bitwise
operations, ignoring loop headers and array indices. Thin wrappers inside crypto files are excluded rather than
labelled negative. Functions in non-crypto files are negative whatever they contain. Labels are read from **source
only**, never from the binary, so they cannot leak detector features. Rationale: `indicrypt_bench/core_labels.py`.

## Intended use and limits

Scoring detectors with ROC-AUC and PR-AUC per ISA, and with the false-discovery proportion among flagged functions at
realistic prevalence (5% and 1% crypto), where a fixed threshold looks strong on balanced sets but is mostly wrong in
the field. Not a measure of whether a binary is secure. Labels reflect a stated rule, not an audit of each function;
inlined crypto inside non-crypto functions is not labelled (loop-level labels are on the roadmap).

## Licences of the 30 source libraries

The dataset is derived from compiled output of these libraries; it contains no source text. Licences below are from
memory of each upstream and **must be re-checked against the pinned commit before publishing**.

| Library | Role | Licence (upstream) |
|---|---|---|
| mbedTLS 3.6.4 | crypto | Apache-2.0 OR GPL-2.0-or-later |
| PQClean | crypto | per scheme: CC0-1.0, Apache-2.0, MIT-0 / public domain mixes |
| libtomcrypt 1.18.2 | crypto | Unlicense / public domain |
| Monocypher 4.0.2 | crypto | BSD-2-Clause OR CC0-1.0 |
| micro-ecc | crypto | BSD-2-Clause |
| tiny-AES-c | crypto | Unlicense |
| B-Con crypto-algorithms | crypto | public domain (author's notice) |
| BearSSL | crypto | MIT |
| TweetNaCl | crypto | public domain |
| SipHash (veorq) | crypto | CC0-1.0 |
| libsodium 1.0.20 | crypto | ISC |
| wolfSSL 5.7.2 | crypto | GPL-2.0-or-later (commercial dual licence) |
| BLAKE3 1.5.4 | crypto | CC0-1.0 OR Apache-2.0 |
| Argon2 (phc-winner) | crypto | CC0-1.0 OR Apache-2.0 |
| tiny_sha3 | crypto | MIT |
| zlib 1.3.1 | negative | zlib licence |
| lz4 1.10.0 | negative | BSD-2-Clause (library) |
| zstd 1.5.6 | negative | BSD-3-Clause OR GPL-2.0 |
| miniz 3.0.2 | negative | MIT |
| stb | negative | MIT OR public domain |
| xxHash 0.8.3 | negative | BSD-2-Clause (library) |
| cJSON 1.7.18 | negative | MIT |
| Lua 5.4.7 | negative | MIT |
| kissfft | negative | BSD-3-Clause |
| brotli 1.1.0 | negative | MIT |
| libdeflate 1.22 | negative | MIT |
| yyjson 0.10.0 | negative | MIT |
| lodepng | negative | zlib licence |
| SQLite 3.46.1 | negative | public domain |
| libpng 1.6.43 | negative | PNG Reference Library License v2 |

The benchmark's own code and the labelled tables: licence to be announced, Copyright 2026 Dhruva P Gowda. Open question for the
owner before upload: whether derived statistics of GPL-licensed builds (wolfSSL, and the GPL option of mbedTLS and zstd)
need a different statement; they contain no code, only names, labels and counts.

## Citation

```
Dhruva P Gowda. IndiCrypt-Bench: a library-disjoint, cross-architecture benchmark for crypto detection in
stripped binaries. 2026. https://github.com/dhruva137/indicrypt-bench
```

```bibtex
@misc{gowda2026indicryptbench,
  author = {Gowda, Dhruva P},
  title  = {IndiCrypt-Bench: a library-disjoint, cross-architecture benchmark for crypto detection in stripped binaries},
  year   = {2026},
  url    = {https://github.com/dhruva137/indicrypt-bench}
}
```
