"""Data cross-references from code in linked binaries: which addresses does a function point at?

Signature rungs (constant tables, NTT twiddles) find *data*; fusion needs the *function* that uses it. In an
object file relocations say that; in a stripped, linked binary they are gone, so references are recovered
from the instructions themselves, per ISA:
    x86-64   RIP-relative memory operands and lea: target = next instruction address + displacement;
             absolute displacements (table + index*scale) in non-PIE code
    AArch64  adrp (page) followed by add/ldr with an immediate on the same register
    ARM32    ldr rX, [pc, #imm] loads a word from the literal pool; the word itself is the address
Direct absolute immediates that land in a data section are also counted. Approximate by design: indirect
and computed addresses are missed, which can only lower recall, never invent a reference.
"""
from __future__ import annotations

import struct

import capstone
from capstone import arm, arm64, x86

from engine.binary_ml.features import _disassembler


def data_refs(code: bytes, arch: str, base: int, image: bytes | None = None, image_base: int = 0) -> set[int]:
    """Addresses the function at ``base`` refers to. ``image`` (+ its load address) resolves ARM32 literals."""
    md = _disassembler(arch)
    refs: set[int] = set()
    pages: dict[int, int] = {}
    for ins in md.disasm(code, base):
        try:
            ops = ins.operands
        except capstone.CsError:
            continue
        if arch == "x86-64":
            for op in ops:
                if op.type == x86.X86_OP_MEM and op.mem.base == x86.X86_REG_RIP:
                    refs.add(ins.address + ins.size + op.mem.disp)
                elif op.type == x86.X86_OP_MEM and op.mem.disp > 0x10000:
                    refs.add(op.mem.disp)          # absolute table + index*scale (non-PIE static code)
                elif op.type == x86.X86_OP_IMM and op.imm > 0x10000:
                    refs.add(op.imm)
        elif arch == "aarch64":
            m = ins.mnemonic
            if m == "adrp" and len(ops) == 2:
                pages[ops[0].reg] = ops[1].imm
                refs.add(ops[1].imm)
            elif m == "adr" and len(ops) == 2:
                refs.add(ops[1].imm)
            elif m == "add" and len(ops) == 3 and ops[1].type == arm64.ARM64_OP_REG and ops[1].reg in pages \
                    and ops[2].type == arm64.ARM64_OP_IMM:
                refs.add(pages[ops[1].reg] + ops[2].imm)
            elif m.startswith("ldr") and ops and ops[-1].type == arm64.ARM64_OP_MEM and ops[-1].mem.base in pages:
                refs.add(pages[ops[-1].mem.base] + ops[-1].mem.disp)
        else:
            if ins.mnemonic.startswith("ldr") and ops and ops[-1].type == arm.ARM_OP_MEM and ops[-1].mem.base == arm.ARM_REG_PC:
                lit = ((ins.address + 8) & ~3) + ops[-1].mem.disp
                refs.add(lit)
                if image is not None and 0 <= lit - image_base <= len(image) - 4:
                    refs.add(struct.unpack_from("<I", image, lit - image_base)[0])
    return refs
