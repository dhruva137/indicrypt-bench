# IndiCrypt-Bench

A library-disjoint, cross-architecture benchmark for finding cryptography in stripped binaries, with a scorer for ROC-AUC, PR-AUC and the false-discovery proportion under conformal selection.

This is open-source research work, developed and maintained by one person, and it is in development. It supports the V.E.R.A. project (https://github.com/dhruva137/V.E.R.A).

The question it asks: given a stripped binary, which of its functions are cryptographic? Any detector can be scored on the same libraries, the same labels and the same metrics.

## What is in the benchmark

- 30 open-source libraries at pinned commits: 15 cryptographic and 15 non-cryptographic hard negatives.
- 4 toolchains (clang x86-64, gcc x86-64 via MinGW, clang AArch64, clang ARM32) at O0, O2, O3 and Os.
- 16,711 compiled objects and 128,288 labelled functions.
- 3 sealed test sets covering 8 sealed crypto libraries, each registered before training and scored once.
- Labels come from source by a frozen rule, never from the binary.

## Install

```bash
pip install indicrypt-bench
indicrypt-bench info              # manifest, splits, pinned commits (--libraries)
```

The wheel is small: the manifest, the splits, the source-derived label table and the scorer. Compiled objects are not shipped. `indicrypt-bench fetch` downloads released data artifacts or, if none exist, rebuilds them from the pinned sources.

## Quick start: score your detector

Write one row per function with your detector's score (higher means more cryptographic):

```csv
library,rel,name,arch,score
bearssl,src/hash/sha2small.c,br_sha2small_round,x86-64,0.97
brotli,c/enc/hash.c,BrotliHashSomething,aarch64,0.12
```

```bash
indicrypt-bench score predictions.csv --split sealed-all
```

`arch` is `x86-64`, `aarch64` or `arm32`. The report gives ROC-AUC and PR-AUC overall and per instruction set, and the false-discovery proportion among flagged functions at natural, 5% and 1% crypto share, for a fixed 0.9 cut-off and for conformal Benjamini-Hochberg selection (`--alpha`, default 0.1). Python API: `from indicrypt_bench.score import score`.

## Why report false discoveries at realistic prevalence

On the sealed libraries, a fixed 0.9 threshold gives 14.5% false findings at the natural crypto share (34%), 60.8% at 5% and 89.5% at 1%. Conformal selection at alpha = 0.1 gives 6.7%, 8.0% and 3.7%. Source: `results/v1_1_detector.json` in the repository.

## Links

- Homepage: https://papertoanything.com/products/indicrypt-bench/
- Repository: https://github.com/dhruva137/indicrypt-bench
- Documentation: https://github.com/dhruva137/indicrypt-bench#readme

Part of Paper To Anything (https://papertoanything.com), research software developed and maintained by Dhruva P Gowda.

## Licence

To be announced.
