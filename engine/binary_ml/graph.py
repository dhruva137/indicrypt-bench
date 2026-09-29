"""Structural views of one function for the v2 detectors: CFG, loops, block data flow, token sequence.

v1 (features.py) summarises a whole function as one vector. Two v1 findings motivate these views:
  - a cipher inlined into main was missed because its loop sits inside orchestration code, so loops
    (strongly connected components of the CFG) are scored on their own (research Phase A1);
  - the graph and sequence models of Phase A3/A4/B3 need per-block features, edges and tokens.
Everything is derived from features.decode(), so the ISA mapping and operand extraction are shared with v1.

    blocks      basic blocks as index ranges into the instruction list
    cfg_edges   (src_block, dst_block) for fall-through and direct branch edges inside the function
    loops       SCCs with more than one block, or a block that branches to itself (Tarjan, iterative)
    dfg_edges   block-level def-use edges from reaching definitions over the CFG (registers only)
    node_matrix one row of NODE_FEATURES per block
    tokens      instruction class x operand flags, at most MAX_TOKENS, for the sequence model
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from engine.binary_ml.features import CLASSES, CRYPTO_CONSTS, NONCRYPTO_CONSTS, Insn

MAX_TOKENS = 512
MAX_NODES = 512
_CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
_LOGIC = {"and", "or", "xor", "not", "shift", "rot"}
NODE_FEATURES: tuple[str, ...] = tuple(f"n_{c}" for c in CLASSES) + (
    "log_len", "imm_rate", "big_imm_rate", "const_crypto", "const_noncrypto", "shift_imm_rate",
    "folded_shift", "folded_rot", "rot_rate", "indexed_load", "byte_extract", "xor_rot", "logic_rate",
    "log_in_deg", "log_out_deg", "in_loop", "loop_header",
)
TOKEN_VOCAB = len(CLASSES) * 4 + 2        # + PAD (0) and CLS (1)


@dataclass
class FunctionGraph:
    blocks: list[tuple[int, int]]          # [start, end) instruction indices
    cfg_edges: list[tuple[int, int]]
    loops: list[list[int]]                  # block ids per loop
    dfg_edges: list[tuple[int, int]]


def build(insns: list[Insn], start: int, end: int, regs: list[tuple[frozenset, frozenset]] | None = None) -> FunctionGraph:
    """Basic blocks, CFG, loops and (when register sets are given) block data flow for one function."""
    n = len(insns)
    if n == 0:
        return FunctionGraph([], [], [], [])
    addr_to_idx = {ins.addr: i for i, ins in enumerate(insns)}
    leaders = {0}
    for i, ins in enumerate(insns):
        if ins.cls in ("cbr", "jmp", "ret"):
            if i + 1 < n:
                leaders.add(i + 1)
            if ins.cls != "ret" and ins.target is not None and ins.target != ins.addr and ins.target in addr_to_idx:
                leaders.add(addr_to_idx[ins.target])
    starts = sorted(leaders)
    blocks = [(s, starts[k + 1] if k + 1 < len(starts) else n) for k, s in enumerate(starts)]
    if len(blocks) > MAX_NODES:
        blocks = blocks[:MAX_NODES]
    block_of = {}
    for b, (s, e) in enumerate(blocks):
        for i in range(s, e):
            block_of[i] = b
    edges = set()
    for b, (s, e) in enumerate(blocks):
        last = insns[e - 1]
        if last.cls in ("cbr", "jmp") and last.target is not None and last.target in addr_to_idx:
            t = block_of.get(addr_to_idx[last.target])
            if t is not None:
                edges.add((b, t))
        if last.cls not in ("jmp", "ret") and b + 1 < len(blocks):
            edges.add((b, b + 1))
    cfg_edges = sorted(edges)
    loops = _sccs(len(blocks), cfg_edges)
    dfg = _reaching_defs(blocks, cfg_edges, regs) if regs is not None else []
    return FunctionGraph(blocks, cfg_edges, loops, dfg)


def _sccs(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Tarjan's algorithm, iterative; returns SCCs that contain a cycle."""
    adj = [[] for _ in range(n)]
    self_loop = set()
    for a, b in edges:
        adj[a].append(b)
        if a == b:
            self_loop.add(a)
    index, low, on, stack, out = [None] * n, [0] * n, [False] * n, [], []
    counter = 0
    for root in range(n):
        if index[root] is not None:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            if i < len(adj[v]):
                work.append((v, i + 1))
                w = adj[v][i]
                if index[w] is None:
                    work.append((w, 0))
                elif on[w]:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1 or v in self_loop:
                    out.append(sorted(comp))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return out


def _reaching_defs(blocks, edges, regs) -> list[tuple[int, int]]:
    """Block-level def-use edges: block A -> block B when a register written in A reaches a read in B."""
    nb = len(blocks)
    gen: list[dict] = []
    use: list[set] = []
    for s, e in blocks:
        last_def: dict = {}
        used: set = set()
        for i in range(s, e):
            read, write = regs[i]
            used |= {r for r in read if r not in last_def}
            for r in write:
                last_def[r] = True
        gen.append(last_def)
        use.append(used)
    preds = [[] for _ in range(nb)]
    for a, b in edges:
        preds[b].append(a)
    # IN[b] = union over preds of OUT[p]; OUT[b] = GEN[b] U (IN[b] minus regs killed in b). Facts: (reg, def_block).
    out = [frozenset((r, b) for r in gen[b]) for b in range(nb)]
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        for b in range(nb):
            inb = set()
            for p in preds[b]:
                inb |= out[p]
            new = frozenset({(r, d) for (r, d) in inb if r not in gen[b]} | {(r, b) for r in gen[b]})
            if new != out[b]:
                out[b] = new
                changed = True
    dfg = set()
    for b in range(nb):
        inb = set()
        for p in preds[b]:
            inb |= out[p]
        for r, d in inb:
            if r in use[b] and d != b:
                dfg.add((d, b))
    return sorted(dfg)


def node_matrix(insns: list[Insn], g: FunctionGraph) -> list[list[float]]:
    """One NODE_FEATURES row per block."""
    indeg = [0] * len(g.blocks)
    outdeg = [0] * len(g.blocks)
    for a, b in g.cfg_edges:
        outdeg[a] += 1
        indeg[b] += 1
    in_loop = set(b for loop in g.loops for b in loop)
    headers = {b for a, b in g.cfg_edges if b <= a}
    rows = []
    for b, (s, e) in enumerate(g.blocks):
        seg = insns[s:e]
        n = max(1, len(seg))
        counts = [0] * len(CLASSES)
        for i in seg:
            counts[_CLASS_ID[i.cls]] += 1
        imms = [v for i in seg for v in i.imms]
        big = [v for v in imms if 0xFFFF < v < 0xFFFFFFFFFFFF0000]
        xr = sum(1 for x, y in zip(seg, seg[1:]) if x.cls == "xor" and (y.cls == "rot" or y.folded == "rot"))
        rows.append([c / n for c in counts] + [
            math.log1p(len(seg)), sum(1 for i in seg if i.imms) / n, len(big) / n,
            float(sum(1 for v in imms if v in CRYPTO_CONSTS)), float(sum(1 for v in imms if v in NONCRYPTO_CONSTS)),
            sum(1 for i in seg if i.shift_imm is not None) / n,
            sum(1 for i in seg if i.folded == "shift") / n, sum(1 for i in seg if i.folded == "rot") / n,
            sum(1 for i in seg if i.cls == "rot" or i.folded == "rot") / n,
            sum(1 for i in seg if i.indexed_load) / n, sum(1 for i in seg if i.byte_extract) / n,
            xr / n, sum(1 for i in seg if i.cls in _LOGIC) / n,
            math.log1p(indeg[b]), math.log1p(outdeg[b]), float(b in in_loop), float(b in headers),
        ])
    return rows


def tokens(insns: list[Insn]) -> list[int]:
    """CLS, then one token per instruction: class x {plain, big immediate, rotate, both}."""
    out = [1]
    for i in insns[: MAX_TOKENS - 1]:
        big = any(0xFFFF < v < 0xFFFFFFFFFFFF0000 for v in i.imms)
        rot = i.cls == "rot" or i.folded is not None
        out.append(2 + _CLASS_ID[i.cls] * 4 + (1 if big else 0) + (2 if rot else 0))
    return out


def loop_instructions(insns: list[Insn], g: FunctionGraph, loop: list[int]) -> list[Insn]:
    return [insns[i] for b in loop for i in range(*g.blocks[b])]
