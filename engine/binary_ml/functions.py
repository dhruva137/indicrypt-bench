"""Find function boundaries in object files and linked binaries, and hand back their bytes.

Three sources of boundaries, recorded on every function so a result says how it was found:

    symbols      ELF STT_FUNC symbols with sizes, or COFF function symbols (size = distance to the
                 next symbol in the same section). Used for relocatable objects and unstripped binaries.
    call-targets For stripped binaries: a linear sweep of each executable section; every direct call
                 target, plus the section start, begins a function that runs to the next start.
                 Cheap and deterministic, but it misses functions reached only indirectly and can
                 merge tail-called neighbours. research/ measures it against symbol boundaries.

Names are returned for labelling and grouping in research only; no feature ever reads a name, so a
stripped binary and an unstripped one are scored identically.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import lief

from engine.binary_ml.features import decode

lief.logging.disable()

_ELF_ARCH = {
    lief.ELF.ARCH.X86_64: "x86-64",
    lief.ELF.ARCH.AARCH64: "aarch64",
    lief.ELF.ARCH.ARM: "arm32",
}
_COFF_ARCH = {"AMD64": "x86-64", "ARM64": "aarch64", "ARMNT": "arm32"}
_PE_ARCH = {"AMD64": "x86-64", "ARM64": "aarch64", "ARMNT": "arm32"}


@dataclass(frozen=True)
class Function:
    name: str           # "" when boundaries were recovered without symbols
    arch: str
    address: int
    code: bytes
    boundary: str       # symbols | call-targets


def _elf_functions(b: lief.ELF.Binary) -> list[Function]:
    arch = _ELF_ARCH.get(b.header.machine_type)
    if arch is None:
        return []
    relocatable = b.header.file_type == lief.ELF.Header.FILE_TYPE.REL
    sections = list(b.sections)
    out = []
    seen = set()
    for sym in b.symbols:
        if sym.type != lief.ELF.Symbol.TYPE.FUNC or sym.size == 0 or sym.shndx in (0, 0xFFF1, 0xFFF2):
            continue
        value = sym.value & ~1 if arch == "arm32" else sym.value
        if relocatable:
            if sym.shndx >= len(sections):
                continue
            sec = sections[sym.shndx]
            off = value
        else:
            sec = b.section_from_virtual_address(value)
            if sec is None:
                continue
            off = value - sec.virtual_address
        key = (sec.name, off)
        if key in seen:
            continue
        seen.add(key)
        content = bytes(sec.content)
        out.append(Function(sym.name, arch, value, content[off:off + sym.size], "symbols"))
    return out


def _coff_functions(path: str) -> list[Function]:
    c = lief.COFF.parse(path)
    if c is None:
        return []
    machine = str(c.header.machine).split(".")[-1]
    arch = _COFF_ARCH.get(machine)
    if arch is None:
        return []
    sections = list(c.sections)
    by_sec: dict[int, list[tuple[int, str]]] = {}
    for s in c.symbols:
        idx = getattr(s, "section_idx", 0)
        if idx <= 0 or str(getattr(s, "complex_type", "")).split(".")[-1] != "FUNCTION":
            continue
        by_sec.setdefault(idx, []).append((s.value, s.name))
    out = []
    for idx, syms in by_sec.items():
        sec = sections[idx - 1]
        content = bytes(sec.content)
        syms.sort()
        for k, (start, name) in enumerate(syms):
            end = syms[k + 1][0] if k + 1 < len(syms) else len(content)
            if end > start:
                out.append(Function(name, arch, start, content[start:end], "symbols"))
    return out


def _recover_by_calls(code: bytes, base: int, arch: str) -> list[Function]:
    starts = {base}
    for ins in decode(code, arch, base):
        if ins.cls == "call" and ins.target is not None and base <= ins.target < base + len(code):
            starts.add(ins.target)
    ordered = sorted(starts)
    out = []
    for k, s in enumerate(ordered):
        e = ordered[k + 1] if k + 1 < len(ordered) else base + len(code)
        out.append(Function("", arch, s, code[s - base:e - base], "call-targets"))
    return out


def _linked(b, kind: str) -> list[Function]:
    """Functions of a parsed ELF or PE: symbols first, call-target recovery when there are none."""
    if kind == "elf":
        funcs = _elf_functions(b)
        if funcs or b.header.file_type == lief.ELF.Header.FILE_TYPE.REL:
            return funcs
        arch = _ELF_ARCH.get(b.header.machine_type)
        if arch is None:
            return []
        out = []
        for sec in b.sections:
            if sec.has(lief.ELF.Section.FLAGS.EXECINSTR) and sec.size:
                out += _recover_by_calls(bytes(sec.content), sec.virtual_address, arch)
        return out
    arch = _PE_ARCH.get(str(b.header.machine).split(".")[-1])
    if arch is None:
        return []
    out = []
    for sec in b.sections:
        if sec.has_characteristic(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE) and sec.size:
            out += _recover_by_calls(bytes(sec.content), b.optional_header.imagebase + sec.virtual_address, arch)
    return out


def functions_in_bytes(data: bytes, fmt: str) -> list[Function]:
    """Functions of an ELF or PE image held in memory (the collector's path); [] if it cannot be parsed."""
    stream = io.BytesIO(data)
    b = lief.ELF.parse(stream) if fmt == "elf" else lief.PE.parse(stream) if fmt == "pe" else None
    return _linked(b, fmt) if b is not None else []


def functions(path: str | Path) -> list[Function]:
    """Every function in an ELF/COFF object or an ELF/PE binary, with how its boundary was found."""
    path = str(path)
    if lief.is_elf(path):
        return _linked(lief.ELF.parse(path), "elf")
    if lief.is_pe(path):
        return _linked(lief.PE.parse(path), "pe")
    if path.endswith(".o"):
        return _coff_functions(path)
    return []
