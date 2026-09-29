"""IndiCrypt-Bench v0 manifest: which libraries, at which commit, and how each source file is labelled.

A label is attached to a *source file*, and every function compiled from that file inherits it. This is
the labelling protocol (research/README.md, "Labelling"):

    crypto      the file implements a cryptographic primitive or its arithmetic (block/stream cipher,
                hash, MAC, AEAD, KDF, DRBG, bignum / curve / lattice / code arithmetic, cipher modes)
    noncrypto   the file does something else (compression, checksums, parsers, encoders, interpreters,
                FFTs, TLS/X.509 plumbing), including files that live *inside* crypto libraries
    exclude     glue we cannot label honestly at file level (algorithm dispatch tables, PSA wrappers,
                OS randomness); never used for training, calibration or testing

Inside crypto files, functions whose names mark them as lifecycle or bookkeeping (init, free, wipe,
self-test, ...) are excluded too (``GLUE_NAME``), because they perform no cryptographic computation.
Each library also carries a ``family`` per file so later milestones can move from detection to
primitive classification without relabelling.

Commits are pinned; ``fetch.py`` checks them out exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Lifecycle / bookkeeping functions inside crypto files: excluded, not counted as crypto or non-crypto.
GLUE_NAME = re.compile(
    r"(^|_)(init|free|zeroize|wipe|burn|test|selftest|self_test|clone|keysize|info|string|name|"
    r"version|register|unregister|descriptor|error|get_type|set_type)(_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rule:
    glob: str          # path glob relative to the library root
    label: str         # crypto | noncrypto | exclude
    family: str = ""   # primitive family for crypto files; component kind for non-crypto


@dataclass(frozen=True)
class Library:
    name: str
    repo: str
    ref: str
    commit: str
    rules: tuple[Rule, ...]
    include: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    role: str = "crypto"            # dominant role, used only for reporting
    extra_sources: tuple[str, ...] = field(default=())  # files under research/indicrypt_bench/extra
    sealed: bool = False            # pre-registered final test set: not inspected before the final run
    exclude_names: str = ""         # regex: functions in this library excluded from all splits (mislabel risk)
    extra_include: tuple[str, ...] = ()   # include dirs under research/indicrypt_bench/extra


PQ_SCHEMES = {
    "crypto_kem/ml-kem-512": "pqc-lattice-kem",
    "crypto_kem/ml-kem-768": "pqc-lattice-kem",
    "crypto_kem/ml-kem-1024": "pqc-lattice-kem",
    "crypto_kem/hqc-128": "pqc-code-kem",
    "crypto_sign/ml-dsa-44": "pqc-lattice-sig",
    "crypto_sign/ml-dsa-65": "pqc-lattice-sig",
    "crypto_sign/ml-dsa-87": "pqc-lattice-sig",
    "crypto_sign/falcon-512": "pqc-lattice-sig",
    "crypto_sign/sphincs-sha2-128s-simple": "pqc-hash-sig",
    "crypto_sign/sphincs-shake-128f-simple": "pqc-hash-sig",
}

MBEDTLS_CRYPTO = {
    "aes": "block", "aria": "block", "camellia": "block", "des": "block", "chacha20": "stream",
    "chachapoly": "aead", "poly1305": "mac", "gcm": "aead", "ccm": "aead", "cmac": "mac", "nist_kw": "mode",
    "md5": "hash", "sha1": "hash", "sha256": "hash", "sha512": "hash", "sha3": "hash", "ripemd160": "hash",
    "bignum": "pk-arith", "bignum_core": "pk-arith", "bignum_mod": "pk-arith", "bignum_mod_raw": "pk-arith",
    "ecp": "pk-arith", "ecp_curves": "pk-arith", "ecp_curves_new": "pk-arith", "ecdsa": "pk-sig",
    "ecdh": "pk-kex", "ecjpake": "pk-kex", "rsa": "pk-arith", "rsa_alt_helpers": "pk-arith", "dhm": "pk-kex",
    "hmac_drbg": "drbg", "ctr_drbg": "drbg", "hkdf": "kdf", "pkcs5": "kdf", "lms": "pk-hash-sig",
    "lmots": "pk-hash-sig", "constant_time": "ct-util",
}
MBEDTLS_NONCRYPTO = {
    "asn1parse": "parser", "asn1write": "encoder", "base64": "encoder", "pem": "parser", "oid": "table",
    "x509": "parser", "x509_crt": "parser", "x509_crl": "parser", "x509_csr": "parser", "x509_create": "encoder",
    "x509write": "encoder", "x509write_crt": "encoder", "x509write_csr": "encoder", "pkparse": "parser",
    "pkwrite": "encoder", "pkcs7": "parser", "net_sockets": "io", "debug": "io", "error": "table",
    "platform": "runtime", "platform_util": "runtime", "timing": "runtime", "threading": "runtime",
    "version": "table", "version_features": "table", "memory_buffer_alloc": "allocator",
    "ssl_msg": "protocol", "ssl_tls": "protocol", "ssl_tls12_client": "protocol", "ssl_tls12_server": "protocol",
    "ssl_tls13_client": "protocol", "ssl_tls13_server": "protocol", "ssl_tls13_generic": "protocol",
    "ssl_client": "protocol", "ssl_cookie": "protocol", "ssl_ticket": "protocol", "ssl_cache": "protocol",
    "ssl_ciphersuites": "table", "ssl_debug_helpers_generated": "table", "mps_reader": "parser",
    "mps_trace": "io",
}


def _mbedtls_rules() -> tuple[Rule, ...]:
    rules = [Rule(f"library/{n}.c", "crypto", f) for n, f in MBEDTLS_CRYPTO.items()]
    rules += [Rule(f"library/{n}.c", "noncrypto", f) for n, f in MBEDTLS_NONCRYPTO.items()]
    return tuple(rules)


def _pq_rules() -> tuple[Rule, ...]:
    rules = [Rule(f"{scheme}/clean/*.c", "crypto", fam) for scheme, fam in PQ_SCHEMES.items()]
    rules += [
        Rule("common/fips202.c", "crypto", "hash"),
        Rule("common/sha2.c", "crypto", "hash"),
        Rule("common/aes.c", "crypto", "block"),
        Rule("common/sp800-185.c", "crypto", "hash"),
        Rule("common/randombytes.c", "exclude", "os-rng"),
    ]
    return tuple(rules)


LTC_CRYPTO_DIRS = {
    "src/ciphers/**/*.c": "block", "src/hashes/*.c": "hash", "src/hashes/sha2/*.c": "hash",
    "src/hashes/whirl/*.c": "hash", "src/hashes/chc/*.c": "hash", "src/stream/**/*.c": "stream",
    "src/mac/**/*.c": "mac", "src/modes/**/*.c": "mode", "src/encauth/**/*.c": "aead",
    "src/prngs/chacha20.c": "drbg", "src/prngs/fortuna.c": "drbg", "src/prngs/rc4.c": "drbg",
    "src/prngs/sober128.c": "drbg", "src/prngs/yarrow.c": "drbg", "src/misc/hkdf/*.c": "kdf",
    "src/misc/pkcs5/*.c": "kdf", "src/misc/mem_neq.c": "ct-util",
}
LTC_NONCRYPTO = {
    "src/misc/adler32.c": "checksum", "src/misc/crc32.c": "checksum", "src/misc/base64/*.c": "encoder",
    "src/misc/error_to_string.c": "table", "src/misc/compare_testvector.c": "runtime",
    "src/misc/zeromem.c": "runtime", "src/misc/burn_stack.c": "runtime", "src/pk/asn1/der/**/*.c": "encoder",
}

LIBRARIES: tuple[Library, ...] = (
    Library("mbedtls", "https://github.com/Mbed-TLS/mbedtls.git", "v3.6.4",
            "c765c831e5c2a0971410692f92f7a81d6ec65ec2", _mbedtls_rules(), include=("include", "library")),
    Library("pqclean", "https://github.com/PQClean/PQClean.git", "master",
            "0586a824fc0d49df0b6b6e9179d8d15d06d0974f", _pq_rules(), include=("common",)),
    Library("libtomcrypt", "https://github.com/libtom/libtomcrypt.git", "v1.18.2",
            "7e7eb695d581782f04b24dc444cbfde86af59853",
            tuple(Rule(g, "crypto", f) for g, f in LTC_CRYPTO_DIRS.items())
            + tuple(Rule(g, "noncrypto", f) for g, f in LTC_NONCRYPTO.items()),
            include=("src/headers",), defines=("LTC_NO_TEST", "LTC_NO_FILE")),
    Library("monocypher", "https://github.com/LoupVaillant/Monocypher.git", "4.0.2",
            "0d85f98c9d9b0227e42cf795cb527dff372b40a4",
            (Rule("src/monocypher.c", "crypto", "mixed"), Rule("src/optional/monocypher-ed25519.c", "crypto", "pk-sig")),
            include=("src", "src/optional")),
    Library("micro-ecc", "https://github.com/kmackay/micro-ecc.git", "master",
            "541b3a78026420a3e369c4c9281c396b5e531113", (Rule("uECC.c", "crypto", "pk-arith"),)),
    Library("tiny-aes", "https://github.com/kokke/tiny-AES-c.git", "master",
            "23856752fbd139da0b8ca6e471a13d5bcc99a08d", (Rule("aes.c", "crypto", "block"),)),
    Library("bcon", "https://github.com/B-Con/crypto-algorithms.git", "master",
            "cfbde48414baacf51fc7c74f275190881f037d32",
            (Rule("aes.c", "crypto", "block"), Rule("arcfour.c", "crypto", "stream"),
             Rule("blowfish.c", "crypto", "block"), Rule("des.c", "crypto", "block"), Rule("md2.c", "crypto", "hash"),
             Rule("md5.c", "crypto", "hash"), Rule("sha1.c", "crypto", "hash"), Rule("sha256.c", "crypto", "hash"),
             Rule("base64.c", "noncrypto", "encoder"), Rule("rot-13.c", "exclude", "classical"))),
    # ---- non-crypto libraries: the hard negatives ----
    Library("zlib", "https://github.com/madler/zlib.git", "v1.3.1", "51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf",
            (Rule("*.c", "noncrypto", "compression"),), role="noncrypto"),
    Library("lz4", "https://github.com/lz4/lz4.git", "v1.10.0", "ebb370ca83af193212df4dcbadcc5d87bc0de2f0",
            (Rule("lib/*.c", "noncrypto", "compression"),), role="noncrypto"),
    Library("zstd", "https://github.com/facebook/zstd.git", "v1.5.6", "794ea1b0afca0f020f4e57b6732332231fb23c70",
            (Rule("lib/common/*.c", "noncrypto", "compression"), Rule("lib/compress/*.c", "noncrypto", "compression"),
             Rule("lib/decompress/*.c", "noncrypto", "compression")),
            include=("lib", "lib/common"), defines=("ZSTD_DISABLE_ASM",), role="noncrypto"),
    Library("miniz", "https://github.com/richgel999/miniz.git", "3.0.2", "293d4db1b7d0ffee9756d035b9ac6f7431ef8492",
            (Rule("miniz*.c", "noncrypto", "compression"),), role="noncrypto"),
    Library("stb", "https://github.com/nothings/stb.git", "master", "2c980bb59875b0d32144a71867fbdebb2f77cd20",
            (Rule("__extra__/stb_*.c", "noncrypto", "media"),), role="noncrypto",
            extra_sources=("stb_image_impl.c", "stb_image_write_impl.c", "stb_truetype_impl.c")),
    Library("xxhash", "https://github.com/Cyan4973/xxHash.git", "v0.8.3", "e626a72bc2321cd320e953a0ccf1584cad60f363",
            (Rule("xxhash.c", "noncrypto", "nc-hash"),), role="noncrypto"),
    Library("cjson", "https://github.com/DaveGamble/cJSON.git", "v1.7.18", "acc76239bee01d8e9c858ae2cab296704e52d916",
            (Rule("cJSON.c", "noncrypto", "parser"), Rule("cJSON_Utils.c", "noncrypto", "parser")), role="noncrypto"),
    Library("lua", "https://github.com/lua/lua.git", "v5.4.7", "1ab3208a1fceb12fca8f24ba57d6e13c5bff15e3",
            (Rule("l*.c", "noncrypto", "interpreter"),), role="noncrypto"),
    Library("kissfft", "https://github.com/mborgerding/kissfft.git", "master",
            "e5e3fac46e0d94a8f8170c06706b7a4218828333",
            (Rule("kiss_fft*.c", "noncrypto", "fft-fixed"),), defines=("FIXED_POINT=32",), role="noncrypto"),
)

# ---- SEALED test libraries (pre-registered 2026-09-28 after the first v1 run; see research/README.md).
# Nothing in these is looked at (names, scores) until the final evaluation.
SEALED: tuple[Library, ...] = (
    Library("bearssl", "https://www.bearssl.org/git/BearSSL", "master", "7bea48e5e850ab4cafbe68d3765cdaba13a86d6f",
            tuple(Rule(f"src/{d}/*.c", "crypto", f) for d, f in (
                ("aead", "aead"), ("ec", "pk-arith"), ("hash", "hash"), ("int", "pk-arith"), ("kdf", "kdf"),
                ("mac", "mac"), ("rand", "drbg"), ("rsa", "pk-arith"), ("symcipher", "block")))
            + (Rule("src/codec/*.c", "noncrypto", "encoder"), Rule("src/x509/*.c", "noncrypto", "parser"),
               Rule("src/ssl/*.c", "noncrypto", "protocol")),
            include=("inc", "src"), sealed=True),
    Library("tweetnacl", "https://tweetnacl.cr.yp.to/20140427/", "20140427",
            "sha256:02e65bc3013ff2168983365e55906bc783c4c7e0a60d8100f17bb303a17175c4",
            (Rule("tweetnacl.c", "crypto", "mixed"),), sealed=True),
    Library("siphash", "https://github.com/veorq/SipHash.git", "master", "32d067603b93b47828700880649198e0bfbbcffa",
            (Rule("siphash.c", "crypto", "mac"), Rule("halfsiphash.c", "crypto", "mac")), sealed=True),
    Library("brotli", "https://github.com/google/brotli.git", "v1.1.0", "ed738e842d2fbdf2d6459e39267a633c4a9b2f5d",
            (Rule("c/common/*.c", "noncrypto", "compression"), Rule("c/dec/*.c", "noncrypto", "compression"),
             Rule("c/enc/*.c", "noncrypto", "compression")), include=("c/include",), role="noncrypto", sealed=True),
    Library("libdeflate", "https://github.com/ebiggers/libdeflate.git", "v1.22", "2335c047e91cac6fd04cb0fd2769380395149f15",
            (Rule("lib/*.c", "noncrypto", "compression"),), include=(".",), role="noncrypto", sealed=True),
    Library("yyjson", "https://github.com/ibireme/yyjson.git", "0.10.0", "9ddba001a4ea88e93b46932e5c5b87b222e19a5f",
            (Rule("src/yyjson.c", "noncrypto", "parser"),), role="noncrypto", sealed=True),
    Library("lodepng", "https://github.com/lvandeve/lodepng.git", "master", "ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a",
            (Rule("__extra__/lodepng_impl.c", "noncrypto", "media"),), role="noncrypto", sealed=True,
            extra_sources=("lodepng_impl.c",)),
)

# ---- SEALED-B (pre-registered 2026-09-28, before v2 work): new libraries, never used for training or
# model selection. Tests whether results hold on more code and more compilers/optimisation levels.
_WOLF_NONCRYPTO = {'asn': 'parser', 'coding': 'encoder', 'error': 'table', 'logging': 'io', 'memory': 'allocator', 'wc_port': 'runtime', 'wolfevent': 'runtime', 'pkcs7': 'parser', 'pkcs12': 'parser', 'cpuid': 'runtime', 'compress': 'compression', 'wc_pkcs11': 'io', 'wc_dsp': 'runtime', 'evp': 'exclude', 'hash': 'exclude', 'signature': 'exclude', 'wc_encrypt': 'exclude', 'asm': 'exclude', 'misc': 'exclude', 'wolfmath': 'exclude', 'cryptocb': 'exclude'}
SEALED_B: tuple[Library, ...] = (
    Library("libsodium", "https://github.com/jedisct1/libsodium.git", "1.0.20-RELEASE",
            "9511c982fb1d046470a8b42aa36556cdb7da15de",
            (Rule("src/libsodium/sodium/codecs.c", "noncrypto", "encoder"),
             Rule("src/libsodium/sodium/utils.c", "noncrypto", "runtime"),
             Rule("src/libsodium/sodium/core.c", "noncrypto", "runtime"),
             Rule("src/libsodium/sodium/runtime.c", "noncrypto", "runtime"),
             Rule("src/libsodium/sodium/version.c", "noncrypto", "table"),
             Rule("src/libsodium/randombytes/**/*.c", "exclude", "os-rng"),
             Rule("src/libsodium/**/*.c", "crypto", "mixed")),
            include=("src/libsodium/include", "src/libsodium/include/sodium"), extra_include=("libsodium", "libsodium/sodium"),
            defines=("CONFIGURED=1", "DEV_MODE=0"), sealed=True),
    Library("wolfssl", "https://github.com/wolfSSL/wolfssl.git", "v5.7.2-stable",
            "00e42151ca061463ba6a95adb2290f678cbca472",
            tuple(Rule(f"wolfcrypt/src/{n}.c", lab if lab == "exclude" else "noncrypto", lab)
                  for n, lab in _WOLF_NONCRYPTO.items())
            + (Rule("wolfcrypt/src/sp_arm*.c", "exclude", "arch-specific"), Rule("wolfcrypt/src/sp_cortexm.c", "exclude", "arch-specific"),
               Rule("wolfcrypt/src/sp_x86_64.c", "exclude", "arch-specific"), Rule("wolfcrypt/src/sp_sm2_*.c", "exclude", "arch-specific"),
               Rule("wolfcrypt/src/sp_dsp32.c", "exclude", "arch-specific"), Rule("wolfcrypt/src/*.c", "crypto", "mixed")),
            include=(".",), extra_include=("wolfssl",), defines=("WOLFSSL_USER_SETTINGS",), sealed=True),
    Library("sqlite", "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip", "3.46.1",
            "sha256:77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784",
            (Rule("sqlite3.c", "noncrypto", "database"),), role="noncrypto", sealed=True,
            exclude_names=r"^(chacha_block|sqlite3_randomness)$"),
    Library("libpng", "https://github.com/pnggroup/libpng.git", "v1.6.43", "ed217e3e601d8e462f7fd1e04bed43ac42212429",
            (Rule("png*.c", "noncrypto", "media"),), include=("../zlib",), extra_include=("libpng",),
            role="noncrypto", sealed=True),
)

# ---- SEALED-C (pre-registered 2026-09-28 during the self-audit, before any scoring): more crypto libraries, so
# generalisation is not measured on only five. Negatives come from sealed and sealed_b.
SEALED_C: tuple[Library, ...] = (
    Library("blake3", "https://github.com/BLAKE3-team/BLAKE3.git", "1.5.4", "95e42b84fc4709974c7b23c7ae885989ab36c31e",
            (Rule("c/blake3.c", "crypto", "hash"), Rule("c/blake3_portable.c", "crypto", "hash"),
             Rule("c/blake3_dispatch.c", "crypto", "hash")),
            include=("c",), defines=("BLAKE3_NO_SSE2", "BLAKE3_NO_SSE41", "BLAKE3_NO_AVX2", "BLAKE3_NO_AVX512",
                                     "BLAKE3_USE_NEON=0"), sealed=True),
    Library("argon2", "https://github.com/P-H-C/phc-winner-argon2.git", "20190702", "62358ba2123abd17fccf2a108a301d4b52c01a7c",
            (Rule("src/argon2.c", "crypto", "kdf"), Rule("src/core.c", "crypto", "kdf"), Rule("src/ref.c", "crypto", "kdf"),
             Rule("src/blake2/blake2b.c", "crypto", "hash"), Rule("src/encoding.c", "noncrypto", "encoder"),
             Rule("src/thread.c", "noncrypto", "runtime")),
            include=("include", "src"), sealed=True),
    Library("tiny_sha3", "https://github.com/mjosaarinen/tiny_sha3.git", "master", "dcbb3192047c2a721f5f851db591871d428036a9",
            (Rule("sha3.c", "crypto", "hash"),), sealed=True),
)
ALL_LIBRARIES: tuple[Library, ...] = LIBRARIES + SEALED + SEALED_B + SEALED_C

# Files that must never be compiled even if a glob matches them.
SKIP = re.compile(r"(_test\.c$|/test|/tests?/|onelua\.c$|ltests\.c$|lua\.c$|luac\.c$|/demos?/|sha3_test\.c$|pngtest\.c$|/example\.c$|shell\.c$)")
