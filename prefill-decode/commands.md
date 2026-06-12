# Commands used for the prefill/decode & batch-scaling experiment

llama.cpp b9596 release binaries (macOS arm64, Metal). Models: Q8 GGUFs
(plamo-2-1b quantized with `--tensor-type ssm_out=bf16`, see study 1).

## 1. Phase asymmetry (prefill vs decode, single sequence)

```bash
llama-bench -m plamo-2-1b-Q8fixed.gguf -p 512,2048 -n 128,512 -r 3
# pp512/pp2048 = prefill tokens/s; tg128/tg512 = decode tokens/s
```

## 2. Batch scaling (aggregate decode throughput vs parallel sequences)

```bash
# hybrid (PLaMo 2) — crashes at -npl >= 6, works at 5 (llama.cpp bug, reported)
llama-batched-bench -m plamo-2-1b-Q8fixed.gguf -c 16384 -ngl 99 -npp 512 -ntg 128 -npl 1,2,4,8,16

# attention-only (PLaMo 3) — scales cleanly to 16
llama-batched-bench -m plamo-3-2b-Q8_0.gguf   -c 16384 -ngl 99 -npp 512 -ntg 128 -npl 1,2,4,8,16
```

Columns: `S_PP t/s` = prefill rate, `S_TG t/s` = aggregate decode rate across all
B sequences, `B` = parallel sequences.

## 3. Crash boundary isolation

```bash
for N in 5 6 7; do
  llama-batched-bench -m plamo-2-1b-Q8fixed.gguf -c 16384 -ngl 99 -npp 512 -ntg 128 -npl $N
done
# 5 = ok; 6,7 = GGML_ASSERT in build_plamo2_mamba_layer (ggml_view_1d out of bounds)
```

Interactive report: `docs/prefill-decode-study.html` (or the GitHub Pages site).
