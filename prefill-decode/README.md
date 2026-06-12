# Prefill vs Decode & Batch Scaling: PLaMo 2 (hybrid) vs PLaMo 3 (attention) on Apple Silicon

M4 Pro 48 GB, llama.cpp b9596 Metal, Q8 quants (plamo-2-1b with the ssm_out fix),
`llama-bench` + `llama-batched-bench` (npp=512, ntg=128).

## 1. Phase asymmetry (plamo-2-1b, B=1)

| Phase | Rate | Bound by |
|---|---|---|
| Prefill pp512 | 2,381 t/s | compute (flat at pp2048: 2,343) |
| Decode tg128 | 126 t/s | memory bandwidth (~19× slower per token) |

Each decoded token re-reads all weights (~1.34 GB) → ~127 t/s ≈ 170 GB/s effective,
consistent with bandwidth-bound decode on a ~273 GB/s part.

## 2. Batch scaling of decode (aggregate t/s)

| B | plamo-2-1b (hybrid, 1.3B) | scaling | plamo-3-2b (attention, 2.6B) | scaling |
|---|---|---|---|---|
| 1 | 128 | 1.0× | 79 | 1.0× |
| 2 | 223 | 1.74× | 148 | 1.88× |
| 4 | 263 | 2.05× | 178 | 2.27× |
| 8 | **CRASH** | — | 182 | 2.31× |
| 16 | **CRASH** | — | 391 | 4.97× |

- **Attention-only PLaMo 3 scales to ~5× at B=16** (prefill stays flat ~1,415 t/s —
  already compute-saturated, exactly the classic asymmetry). The superlinear jump
  from B=8→16 (182→391) suggests a Metal kernel-path switch at that batch width —
  unverified, footnote-grade.
- **Hybrid PLaMo 2 bends earlier** (2.05× at B=4 vs 2.27×) — consistent with
  per-sequence SSM state updates being non-amortizable compute — **and then crashes
  outright at B≥6** (see bug below). On this hardware/build, continuous batching
  beyond 5 concurrent users is simply unavailable for the hybrid.

## 3. Crash: plamo2 Mamba layer with ≥6 parallel sequences (llama.cpp bug #2)

```
GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
  ggml_view_1d <- llama_model_plamo2::graph::build_plamo2_mamba_layer (llm_graph_input_rs)
```

Repro (deterministic; works at -npl 5, crashes at -npl 6):

```bash
llama-batched-bench -m plamo-2-1b-Q8.gguf -c 16384 -ngl 99 -npp 512 -ntg 128 -npl 6
```

The plamo2 graph builder creates an out-of-bounds 1-D view into the recurrent-state
buffer once parallel sequences exceed 5. plamo-3 (attention-only) runs B=16 cleanly
on the same build, so this is specific to the hybrid path.

## 4. Takeaways (deployment-relevant)

1. Decode, not prefill, is the laptop bottleneck (19×) — speeding up single-user
   inference means attacking decode (quantization, speculative decoding, batching),
   never prefill.
2. "More replicas" on one box is the wrong lever: N processes duplicate weights in
   RAM and contend for the same bandwidth. One server with `--parallel N`
   (continuous batching) shares one weight copy — PLaMo 3 gets 5× aggregate
   throughput that way, free.
3. Architecture choice changes serving economics: the hybrid's batch ceiling
   (≤5 sequences, earlier bend) vs attention's clean scaling is a concrete,
   measured cost of Samba-style designs under multi-user serving in llama.cpp today
   — complementing the spec-decode study's finding (SSM checkpoint overhead) from a
   second angle.

## Open follow-up

- Replica-vs-`--parallel` head-to-head (2× single-slot servers vs 1 server `--parallel 2`).
- File the B≥6 crash upstream (added to llamacpp-issue.md as bug #2).
- PLaMo 3 B=8→16 superlinearity — profile which Metal kernels switch.
