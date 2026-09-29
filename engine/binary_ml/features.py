"""Architecture-neutral function features for signature-free crypto detection in binaries.

WHY THIS EXISTS
---------------
The binary collector (collectors/binary_scanner.py) finds cryptography by linked libraries, symbols,
banners and constant tables. It states its own gap: no disassembly, so custom cryptography, stripped
static builds, and implementations that compute their tables at start-up are missed. This module is
the first rung of the detector that covers that gap: every function is disassembled (capstone) and
described by a fixed-length vector that a small model scores on a CPU.

TWO FEATURE FAMILIES (kept separate so the ablation in research/ can isolate each)
    mnemonic   what Mnemocrypt (NDSS BAR 2025) uses: structure plus statistics of instruction
               *classes*. Our re-implementation, in the same spirit, not their released model.
    operand    what Mnemocrypt leaves out ("it might be possible to obtain better results by also
               utilizing ... the instruction operands", their Section VII): immediates and their bit
               entropy, known-constant hits, shift and rotate amounts, rotates folded into ARM
               barrel-shifter operands, indexed table loads, byte extraction, straight-line ALU runs.

ARCHITECTURE NEUTRALITY
-----------------------
Mnemonics are mapped to one set of operation classes for x86-64, AArch64 and ARM32, so a model
trained on one ISA can be *measured* on another. On ARM a rotate often hides inside another
instruction's operand (``eor w0, w1, w2, ror #13``); the operand family counts it, the mnemonic
family cannot. research/ tests whether that matters.

Nothing here executes code or reads key material: bytes in, numbers out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import capstone
from capstone import arm, arm64, x86

ARCHES = ("x86-64", "aarch64", "arm32")
MAX_INSNS = 20_000          # a function larger than this is truncated, and flagged by a feature
MIN_INSNS = 10              # below this a function carries too little signal to score

CLASSES = (
    "mov", "load", "store", "stack", "add", "mul", "div", "and", "or", "xor", "not", "shift", "rot",
    "bitm", "cmp", "cbr", "jmp", "call", "ret", "simd", "float", "cext", "nop", "other",
)
_ALU = {"add", "mul", "div", "and", "or", "xor", "not", "shift", "rot", "bitm"}
_LOGIC = {"and", "or", "xor", "not", "shift", "rot"}

# Constants whose presence as an *immediate* is telling. Crypto: hash IVs and round constants,
# ChaCha/Salsa sigma, TEA delta, lattice moduli and Montgomery factors. Non-crypto: CRC and
# fast-hash primes, so the model can learn that "odd 32-bit constant" alone is not cryptography.
CRYPTO_CONSTS = frozenset({
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0,               # MD5 / SHA-1 IV
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6,                           # SHA-1 K
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE, 0xF57C0FAF, 0x4787C62A,   # MD5 T
    0x61707865, 0x3320646E, 0x79622D32, 0x6B206574,                           # "expand 32-byte k"
    0x9E3779B9, 0xC6EF3720,                                                   # TEA delta, 32*delta
    0xB7E15163,                                                               # RC5/RC6 P32
    3329, 62209, 0xF301, 8380417, 58728449, 4193792, 0x7FE001,                # ML-KEM / ML-DSA
    12289,                                                                    # Falcon q
    0x0000000000008082, 0x800000000000808A, 0x8000000080008000,               # Keccak RC
    0xF3BCC908, 0x84CAA73B, 0xFE94F82B, 0x5F1D36F1, 0xADE682D1, 0x2B3E6C1F,   # SHA-512 / BLAKE2b IV low
})
NONCRYPTO_CONSTS = frozenset({
    0xEDB88320, 0x04C11DB7, 0x82F63B78, 0x1EDC6F41,                           # CRC-32 / CRC-32C
    0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1,               # xxHash32 primes
    0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,               # xxHash64 primes
    0x01000193, 0x811C9DC5, 0x100000001B3, 0xCBF29CE484222325,                # FNV
    0xCC9E2D51, 0x1B873593, 0x85EBCA6B, 0xC2B2AE35,                           # MurmurHash3
    65521,                                                                    # Adler-32 base
})


@dataclass(frozen=True)
class Insn:
    cls: str
    imms: tuple[int, ...]
    shift_imm: int | None        # amount for shift/rotate by immediate, including ARM operand shifts
    folded: str | None           # "shift" / "rot" when an ARM operand carries a shift or rotate
    indexed_load: bool           # memory read with an index register (table lookup shape)
    byte_extract: bool           # movzx byte / uxtb / ubfx #8-aligned / and #0xff
    addr: int
    size: int
    target: int | None           # direct branch or call target


# ---------------------------------------------------------------- mnemonic -> class, per ISA

_X86_PREFIX = (
    ("aes", "cext"), ("sha1", "cext"), ("sha256", "cext"), ("pclmul", "cext"), ("vpclmul", "cext"),
    ("vaes", "cext"), ("gf2p8", "cext"),
    ("cmov", "mov"), ("set", "mov"), ("movzx", "mov"), ("movsx", "mov"), ("movabs", "mov"),
    ("push", "stack"), ("pop", "stack"),
    ("imul", "mul"), ("mul", "mul"), ("idiv", "div"), ("div", "div"),
    ("xor", "xor"), ("pxor", "xor"), ("vpxor", "xor"), ("and", "and"), ("pand", "and"), ("vpand", "and"),
    ("andn", "and"), ("or", "or"), ("por", "or"), ("vpor", "or"), ("not", "not"), ("neg", "add"),
    ("rorx", "rot"), ("rol", "rot"), ("ror", "rot"), ("rcl", "rot"), ("rcr", "rot"), ("vprol", "rot"), ("vpror", "rot"),
    ("shld", "shift"), ("shrd", "shift"), ("shl", "shift"), ("shr", "shift"), ("sal", "shift"), ("sar", "shift"),
    ("shlx", "shift"), ("shrx", "shift"), ("sarx", "shift"), ("psll", "shift"), ("psrl", "shift"), ("psra", "shift"),
    ("vpsll", "shift"), ("vpsrl", "shift"), ("vpsra", "shift"),
    ("bswap", "bitm"), ("popcnt", "bitm"), ("lzcnt", "bitm"), ("tzcnt", "bitm"), ("bsf", "bitm"), ("bsr", "bitm"),
    ("bt", "bitm"), ("pdep", "bitm"), ("pext", "bitm"), ("bextr", "bitm"), ("blsr", "bitm"),
    ("add", "add"), ("adc", "add"), ("adox", "add"), ("adcx", "add"), ("sub", "add"), ("sbb", "add"),
    ("inc", "add"), ("dec", "add"), ("lea", "add"), ("paddd", "add"), ("paddq", "add"), ("psub", "add"),
    ("cmp", "cmp"), ("test", "cmp"),
    ("call", "call"), ("ret", "ret"), ("jmp", "jmp"), ("j", "cbr"), ("loop", "cbr"),
    ("nop", "nop"), ("endbr", "nop"), ("int3", "nop"), ("ud2", "nop"),
    ("f", "float"), ("cvt", "float"), ("ucomis", "float"), ("comis", "float"),
    ("vp", "simd"), ("p", "simd"), ("v", "simd"),
    ("mov", "mov"), ("xchg", "mov"), ("lods", "mov"), ("stos", "mov"), ("cdq", "mov"), ("cqo", "mov"), ("cdqe", "mov"),
)
_X86_FLOAT_SUFFIX = ("ss", "sd", "ps", "pd")


def _cls_x86(m: str) -> str:
    if m.endswith(_X86_FLOAT_SUFFIX) and not m.startswith(("p", "vp", "movs", "cmps", "lods", "stos", "scas")):
        if m not in ("movsd",):  # string op sharing a mnemonic with the scalar double move
            return "float"
    for prefix, cls in _X86_PREFIX:
        if m.startswith(prefix):
            return cls
    return "other"


_A64 = {
    "mov": "mov", "movz": "mov", "movk": "mov", "movn": "mov", "adr": "mov", "adrp": "mov", "csel": "mov",
    "csinc": "mov", "csinv": "mov", "csneg": "mov", "cset": "mov", "csetm": "mov", "fmov": "float",
    "sxtw": "mov", "sxtb": "mov", "sxth": "mov", "uxtw": "mov",
    "uxtb": "bitm", "uxth": "bitm",
    "add": "add", "adds": "add", "adc": "add", "adcs": "add", "sub": "add", "subs": "add", "sbc": "add",
    "sbcs": "add", "neg": "add", "negs": "add", "cmn": "cmp", "cmp": "cmp", "tst": "cmp", "ccmp": "cmp", "ccmn": "cmp",
    "mul": "mul", "madd": "mul", "msub": "mul", "mneg": "mul", "smull": "mul", "umull": "mul", "smulh": "mul",
    "umulh": "mul", "smaddl": "mul", "umaddl": "mul", "smsubl": "mul", "umsubl": "mul",
    "sdiv": "div", "udiv": "div",
    "and": "and", "ands": "and", "bic": "and", "bics": "and", "orr": "or", "orn": "or", "eor": "xor",
    "eon": "xor", "mvn": "not",
    "lsl": "shift", "lsr": "shift", "asr": "shift", "lslv": "shift", "lsrv": "shift", "asrv": "shift",
    "ror": "rot", "rorv": "rot", "extr": "rot",
    "ubfx": "bitm", "sbfx": "bitm", "ubfiz": "bitm", "sbfiz": "bitm", "bfi": "bitm", "bfxil": "bitm",
    "ubfm": "bitm", "sbfm": "bitm", "bfm": "bitm", "rev": "bitm", "rev16": "bitm", "rev32": "bitm",
    "rbit": "bitm", "clz": "bitm", "cls": "bitm",
    "b": "jmp", "br": "jmp", "bl": "call", "blr": "call", "ret": "ret",
    "cbz": "cbr", "cbnz": "cbr", "tbz": "cbr", "tbnz": "cbr",
    "nop": "nop", "hint": "nop", "brk": "nop", "udf": "nop",
    "stp": "stack", "ldp": "stack",
    "aese": "cext", "aesd": "cext", "aesmc": "cext", "aesimc": "cext", "sha1c": "cext", "sha1h": "cext",
    "sha1m": "cext", "sha1p": "cext", "sha1su0": "cext", "sha1su1": "cext", "sha256h": "cext",
    "sha256h2": "cext", "sha256su0": "cext", "sha256su1": "cext", "pmull": "cext", "pmull2": "cext",
    "sha512h": "cext", "sha512h2": "cext", "eor3": "cext", "rax1": "cext", "xar": "cext", "bcax": "cext",
}


def _cls_a64(m: str, n_regs_simd: bool) -> str:
    base = m.split(".")[0]
    if base.startswith("b.") or m.startswith("b."):
        return "cbr"
    if base in _A64:
        cls = _A64[base]
        if n_regs_simd and cls in _ALU:
            return "simd"
        return cls
    if base.startswith(("ldr", "ldur", "ldar", "ldx", "ldax", "ld1", "ld2", "ld3", "ld4")):
        return "load"
    if base.startswith(("str", "stur", "stlr", "stx", "stlx", "st1", "st2", "st3", "st4")):
        return "store"
    if base.startswith("f") or base.startswith(("scvtf", "ucvtf")):
        return "float"
    return "simd" if n_regs_simd else "other"


_A32 = {
    "mov": "mov", "movw": "mov", "movt": "mov", "mvn": "not", "adr": "mov",
    "add": "add", "adds": "add", "adc": "add", "adcs": "add", "sub": "add", "subs": "add", "sbc": "add",
    "sbcs": "add", "rsb": "add", "rsbs": "add", "rsc": "add",
    "mul": "mul", "muls": "mul", "mla": "mul", "mls": "mul", "umull": "mul", "smull": "mul", "umlal": "mul",
    "smlal": "mul", "umaal": "mul", "sdiv": "div", "udiv": "div",
    "and": "and", "ands": "and", "bic": "and", "bics": "and", "orr": "or", "orrs": "or", "orn": "or",
    "eor": "xor", "eors": "xor",
    "lsl": "shift", "lsls": "shift", "lsr": "shift", "lsrs": "shift", "asr": "shift", "asrs": "shift",
    "ror": "rot", "rors": "rot", "rrx": "rot",
    "uxtb": "bitm", "uxth": "bitm", "sxtb": "mov", "sxth": "mov", "ubfx": "bitm", "sbfx": "bitm", "bfi": "bitm",
    "bfc": "bitm", "rev": "bitm", "rev16": "bitm", "rbit": "bitm", "clz": "bitm", "uxtab": "bitm", "uxtah": "bitm",
    "cmp": "cmp", "cmn": "cmp", "tst": "cmp", "teq": "cmp",
    "b": "jmp", "bx": "jmp", "bl": "call", "blx": "call",
    "push": "stack", "pop": "stack", "nop": "nop", "udf": "nop", "bkpt": "nop",
    "aese": "cext", "aesd": "cext", "aesmc": "cext", "aesimc": "cext", "sha256h": "cext", "vmull": "cext",
}


def _cls_a32(m: str, cond: bool) -> str:
    base = m.split(".")[0]
    if base in ("b",) and cond:
        return "cbr"
    if base in _A32:
        return _A32[base]
    if base.startswith(("ldr", "ldm", "pop")):
        return "load"
    if base.startswith(("str", "stm", "push")):
        return "store"
    if base.startswith("v"):
        return "float" if any(t in m for t in (".f32", ".f64")) else "simd"
    return "other"


@lru_cache(maxsize=None)
def _disassembler(arch: str) -> capstone.Cs:
    if arch == "x86-64":
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    elif arch == "aarch64":
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    elif arch == "arm32":
        md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
    else:
        raise ValueError(f"unsupported architecture {arch!r}; supported: {ARCHES}")
    md.detail = True
    md.skipdata = True       # data in a code range becomes a .byte pseudo-insn, not a stop
    return md


def _mask(v: int) -> int:
    return v & 0xFFFFFFFFFFFFFFFF


def decode(code: bytes, arch: str, base: int = 0, regs: list | None = None) -> list[Insn]:
    """Disassemble one function into architecture-neutral instructions.

    When ``regs`` is a list, one (registers read, registers written) pair of frozensets is appended to it per
    instruction, for the block data-flow graph (graph.py). The v1 runtime never asks for them.
    """
    md = _disassembler(arch)
    out: list[Insn] = []
    for ins in md.disasm(code, base):
        if len(out) >= MAX_INSNS:
            break
        m = ins.mnemonic.lower()
        if regs is not None:
            try:
                rd, wr = ins.regs_access()
                regs.append((frozenset(rd), frozenset(wr)))
            except capstone.CsError:
                regs.append((frozenset(), frozenset()))
        if m == ".byte":
            out.append(Insn("other", (), None, None, False, False, ins.address, ins.size, None))
            continue
        imms: list[int] = []
        shift_imm = folded = target = None
        indexed = byte_x = False
        try:
            ops = ins.operands
        except capstone.CsError:
            ops = []
        if arch == "x86-64":
            cls = _cls_x86(m)
            reads_mem = False
            for op in ops:
                if op.type == x86.X86_OP_IMM:
                    imms.append(_mask(op.imm))
                elif op.type == x86.X86_OP_MEM:
                    if op.mem.index != 0 and op.access & capstone.CS_AC_READ:
                        indexed = True
                    if op.access & capstone.CS_AC_READ:
                        reads_mem = True
            if cls == "mov" and reads_mem:
                cls = "load"
            elif cls == "mov" and ops and ops[0].type == x86.X86_OP_MEM:
                cls = "store"
            if m.startswith("movzx") and len(ops) == 2 and ops[1].size == 1:
                byte_x = True
            if cls in ("shift", "rot") and imms:
                shift_imm = imms[-1] & 0x3F
            if cls in ("call", "jmp", "cbr") and ops and ops[0].type == x86.X86_OP_IMM:
                target = ops[0].imm
                imms = []
        elif arch == "aarch64":
            simd = any(op.type == arm64.ARM64_OP_REG and ins.reg_name(op.reg)
                       and ins.reg_name(op.reg)[0] in "vqds" and not ins.reg_name(op.reg).startswith("sp")
                       for op in ops)
            cls = _cls_a64(m, simd)
            for op in ops:
                if op.type == arm64.ARM64_OP_IMM:
                    imms.append(_mask(op.imm))
                elif op.type == arm64.ARM64_OP_MEM and op.mem.index != 0:
                    indexed = True
                if op.shift.type != arm64.ARM64_SFT_INVALID and op.shift.value:
                    folded = "rot" if op.shift.type == arm64.ARM64_SFT_ROR else "shift"
                    shift_imm = op.shift.value & 0x3F
            if cls in ("shift", "rot") and shift_imm is None and imms:
                shift_imm = imms[-1] & 0x3F
            if m in ("uxtb",) or (m in ("ubfx", "and") and imms and imms[-1] in (0xFF, 8)):
                byte_x = True
            if cls in ("call", "jmp", "cbr") and imms:
                target = imms[-1]
                imms = []
        else:  # arm32
            cond = getattr(ins, "cc", arm.ARM_CC_AL) not in (arm.ARM_CC_AL, arm.ARM_CC_INVALID)
            cls = _cls_a32(m, cond)
            for op in ops:
                if op.type == arm.ARM_OP_IMM:
                    imms.append(_mask(op.imm))
                elif op.type == arm.ARM_OP_MEM and op.mem.index != 0:
                    indexed = True
                if op.shift.type not in (0, arm.ARM_SFT_INVALID) and op.shift.value:
                    rot = op.shift.type in (arm.ARM_SFT_ROR, arm.ARM_SFT_ROR_REG)
                    folded = "rot" if rot else "shift"
                    shift_imm = op.shift.value & 0x3F
            if cls in ("shift", "rot") and shift_imm is None and imms:
                shift_imm = imms[-1] & 0x3F
            if m == "uxtb" or (m == "and" and imms and imms[-1] == 0xFF):
                byte_x = True
            if cls in ("call", "jmp", "cbr") and imms:
                target = imms[-1]
                imms = []
        out.append(Insn(cls, tuple(imms), shift_imm, folded, indexed, byte_x, ins.address, ins.size, target))
    return out


# ---------------------------------------------------------------- the feature vector

_BIGRAMS = (("xor", "rot"), ("rot", "xor"), ("add", "rot"), ("rot", "add"), ("xor", "shift"),
            ("shift", "xor"), ("and", "xor"), ("xor", "and"), ("mul", "add"), ("add", "mul"),
            ("load", "xor"), ("xor", "load"), ("shift", "and"), ("and", "shift"), ("xor", "xor"),
            ("add", "xor"), ("or", "shift"), ("shift", "or"))
_CLASS_FEATS = ("add", "mul", "div", "and", "or", "xor", "not", "shift", "rot", "bitm", "cmp", "cbr",
                "jmp", "call", "load", "store", "stack", "simd", "float", "cext", "other")

MNEMONIC_FEATURES: tuple[str, ...] = (
    ("log_insns", "log_blocks", "backedges", "cyclomatic", "max_block", "mean_block", "caballero",
     "caballero_adjusted", "caballero_asym", "has_float", "has_cext")
    + tuple(f"d_{c}" for c in _CLASS_FEATS)
    + tuple(f"bg_{a}_{b}" for a, b in _BIGRAMS)
    + ("blk_std_logic", "blk_max_logic", "blk_std_move")
)
OPERAND_FEATURES: tuple[str, ...] = (
    "imm_rate", "imm_big_rate", "imm_entropy_mean", "imm_distinct_big", "const_crypto", "const_noncrypto",
    "shift_imm_rate", "shift_amt_distinct", "shift_amt_entropy", "rot_amt_nonbyte", "folded_shift",
    "folded_rot", "rot_total", "indexed_load_rate", "byte_extract_rate", "mask_ff_rate",
    "alu_run_max", "alu_run_mean", "xor_rot_chain", "reg_alu_rate", "truncated",
)
FEATURES: tuple[str, ...] = MNEMONIC_FEATURES + OPERAND_FEATURES


def _entropy(counts: dict) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _bit_entropy(v: int) -> float:
    """How random a constant's bits look: 1.0 for half ones, 0 for all-zero or all-one."""
    width = 64 if v >> 32 else 32
    p = bin(v).count("1") / width
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def vector(insns: list[Insn], start: int, end: int) -> list[float]:
    """Fixed-length feature vector (FEATURES order) for one function spanning [start, end)."""
    n = len(insns)
    if n == 0:
        return [0.0] * len(FEATURES)
    counts = {c: 0 for c in CLASSES}
    for i in insns:
        counts[i.cls] += 1
    nonmov = max(1, n - counts["mov"] - counts["load"] - counts["store"] - counts["stack"])

    # Basic blocks: leaders are the entry, direct-branch targets inside the function, and fall-throughs.
    leaders = {insns[0].addr}
    backedges = 0
    for k, i in enumerate(insns):
        # A target equal to the branch itself is an unresolved relocation in an object file, not a loop.
        if i.cls in ("cbr", "jmp") and i.target is not None and start <= i.target < end and i.target != i.addr:
            leaders.add(i.target)
            if i.target <= i.addr:
                backedges += 1
        if i.cls in ("cbr", "jmp", "ret") and k + 1 < n:
            leaders.add(insns[k + 1].addr)
    blocks: list[list[Insn]] = []
    cur: list[Insn] = []
    for i in insns:
        if i.addr in leaders and cur:
            blocks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    sizes = [len(b) for b in blocks]
    edges = sum(1 for i in insns if i.cls == "cbr") * 2 + sum(1 for i in insns if i.cls == "jmp")
    logic_per_block = [sum(1 for i in b if i.cls in _LOGIC) for b in blocks]
    move_per_block = [sum(1 for i in b if i.cls in ("mov", "load", "store")) for b in blocks]

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    arith_logic = sum(counts[c] for c in _ALU)
    adjusted = sum(counts[c] for c in ("xor", "shift", "rot", "and", "or", "add")) / nonmov
    asym = sum(counts[c] for c in ("mul", "add", "div")) / nonmov
    bigram = {b: 0 for b in _BIGRAMS}
    for a, b in zip(insns, insns[1:]):
        key = (a.cls, b.cls)
        if key in bigram:
            bigram[key] += 1

    mnemonic = [
        math.log1p(n), math.log1p(len(blocks)), float(backedges), float(edges - len(blocks) + 2),
        float(max(sizes)), sum(sizes) / len(sizes), arith_logic / n, adjusted, asym,
        float(counts["float"] > 0), float(counts["cext"] > 0),
    ]
    mnemonic += [counts[c] / nonmov for c in _CLASS_FEATS]
    mnemonic += [bigram[b] / max(1, n - 1) for b in _BIGRAMS]
    mnemonic += [_std(logic_per_block), float(max(logic_per_block)), _std(move_per_block)]

    # Operand family.
    all_imms = [v for i in insns for v in i.imms]
    big = [v for v in all_imms if 0xFFFF < v < 0xFFFFFFFFFFFF0000]   # excludes small and small-negative
    with_imm = sum(1 for i in insns if i.imms)
    shifts = [i for i in insns if i.cls in ("shift", "rot") or i.folded]
    amts: dict[int, int] = {}
    for i in shifts:
        if i.shift_imm is not None:
            amts[i.shift_imm] = amts.get(i.shift_imm, 0) + 1
    rot_like = [i for i in insns if i.cls == "rot" or i.folded == "rot"]
    rot_nonbyte = sum(1 for i in rot_like if i.shift_imm is not None and i.shift_imm % 8)
    runs, run = [], 0
    for i in insns:
        if i.cls in _ALU:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    xor_rot = sum(1 for a, b, c in zip(insns, insns[1:], insns[2:])
                  if a.cls == "xor" and ("rot" in (b.cls, c.cls) or "rot" in (b.folded, c.folded)))
    alu = [i for i in insns if i.cls in _ALU]
    reg_alu = sum(1 for i in alu if not i.imms)
    masks = sum(1 for v in all_imms if v in (0xFF, 0xFFFF, 0xFFFFFFFF))
    operand = [
        with_imm / n, len(big) / n,
        (sum(_bit_entropy(v) for v in big) / len(big)) if big else 0.0,
        math.log1p(len(set(big))),
        float(sum(1 for v in all_imms if v in CRYPTO_CONSTS)),
        float(sum(1 for v in all_imms if v in NONCRYPTO_CONSTS)),
        sum(1 for i in shifts if i.shift_imm is not None) / n,
        float(len(amts)), _entropy(amts), rot_nonbyte / n,
        sum(1 for i in insns if i.folded == "shift") / n,
        sum(1 for i in insns if i.folded == "rot") / n,
        len(rot_like) / n,
        sum(1 for i in insns if i.indexed_load) / n,
        sum(1 for i in insns if i.byte_extract) / n,
        masks / n,
        float(max(runs) if runs else 0), (sum(runs) / len(runs)) if runs else 0.0,
        xor_rot / n, (reg_alu / len(alu)) if alu else 0.0,
        float(n >= MAX_INSNS),
    ]
    return mnemonic + operand


def function_vector(code: bytes, arch: str, base: int = 0) -> tuple[list[float], int]:
    """(features, instruction count) for the bytes of one function."""
    insns = decode(code, arch, base)
    return vector(insns, base, base + len(code)), len(insns)
