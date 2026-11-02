# ryzen-sm86-hybrid — research ledger

> Do the "unconventional Ryzen CPU + NVIDIA SM86 (Ampere) CUDA" pairing on this
> box: real workflow PNG-level proof, not SVG. Accept that a hypothesis may be
> wrong. Iterations stamped by real datetime, not vague dates. Dark mode.

## Hardware (verified 2026-11-01T17:10-07:00)

| Device | Detail |
|---|---|
| GPU | NVIDIA RTX 3090, compute_cap **8.6** (Ampere GA102, SM86 class), 24 GiB, driver 615.78.08 |
| CPU | AMD **Ryzen 7 8700F**, 8 cores (Zen 4), x86_64 |

The Ryzen+NVIDIA pairing is **not** rare in the abstract — the genuinely
unusual part is the CPU-GPU memory-coherence angle below.

## Paper findings (web_search + arxiv)

1. **`2610.13125`** — Characterizing CPU-GPU Memory Sharing over Unified Memory
   Architecture: UMA sharing is a correctness+perf building block for
   heterogeneous compute.
2. **Unified Splitting (Kastner/UCSD)** — Inference split across CPU+GPU with
   UM; GPU processes batch N+1 while CPU completes batch N (pipelined overlap).
3. **HECC / CUDA Programming Guide (Unified Memory)** — the GPU has *exclusive
   access* to managed pages during kernel execution; the CPU must use
   `cuda::barrier` / `cuda::semaphore` (or wait-based schemes) to coordinate —
   it cannot touch managed data while the GPU is busy.

### The genuinely intriguing approach

Pipelined CPU-GPU overlap on UM with explicit barriers: while the GPU executes
kernel batch N+1, the CPU prepares batch N into managed memory, and
`cuda::barrier` orders publication. That is the hypothesis the PNG proof
targets — real overlap and correctness coexistence, not a toy memory copy.

## MCP/tooling dependency (fixed in same effort)

The paper-research lane depended on the `arxiv-advanced` MCP server, which was
crashed by a cross-Python-ABI PYTHONPATH bleed (root cause + fix: see
`../arxiv-mcp-vendored/README.md` and tau `f0bfd034`). The server is vendored
and fixed; search now returns real papers.

## Status / proof

- [ ] PNG-level real-workflow proof (dark mode) — pending (proof phase).
- [ ] 6 subagent iterations, datetime-stamped — pending.