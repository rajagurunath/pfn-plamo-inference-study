# PLaMo Inference & Fine-Tuning Studies

Three hands-on studies of [Preferred Networks' PLaMo models](https://huggingface.co/pfnet),
all measured on a single MacBook (M4 Pro, 48 GB, Apple Silicon / Metal).

**📊 Read the reports:** [interactive site](https://rajagurunath.github.io/pfn-plamo-inference-study/) ·
**🤗 Model:** [plamo-3-nict-2b-tool-calling-lora](https://huggingface.co/Gurunath/plamo-3-nict-2b-tool-calling-lora)

## The three studies, in one minute

### 1. Speculative decoding for PLaMo 2 (Mamba-hybrid) — `bench/`, `report/`

Can a 1B PLaMo draft for an 8B PLaMo in llama.cpp? **It works correctly but never pays off**:
outputs are byte-identical to baseline (sha256-verified), yet even 90% draft acceptance is
slower than no speculation — the per-round cost of drafting plus SSM state checkpointing
exceeds the savings on bandwidth-bound hardware. Only free n-gram drafting on translation
wins (1.21×). Along the way we **root-caused a quantization bug**: Q8_0 on the `ssm_out`
tensor makes PLaMo 2 emit invisible reserved tokens forever on any multi-line prompt —
likely affecting every community PLaMo 2 GGUF. Fix: `llama-quantize --tensor-type ssm_out=bf16`. Reported upstream: [llama.cpp#24501](https://github.com/ggml-org/llama.cpp/issues/24501).

### 2. Tool-calling LoRA for PLaMo 3 — `finetune/`, `report-finetune/`

PLaMo 3's open base models secretly ship chat-format rails (an official chat template +
undocumented control tokens like `<|plamo:key|>`, `<|plamo:constrain|>`). A **39-minute LoRA
on a laptop** takes function calling from 0% → **100% argument-exact** on unseen queries —
the first open-weights PLaMo with tool calling. The honest part: we hit (and document) two
evaluation traps — a contaminated eval (97.5% from duplicate leakage) and a label flaw that
made *more* training data *worse* (90% → 55%) until the data was fixed. The lesson is about
data, not models.

### 3. Prefill vs decode & batch scaling — `prefill-decode/`

Prefill is compute-bound, decode is bandwidth-bound — **19× apart per token** on this
hardware. Continuous batching recovers ~**5×** aggregate decode throughput for attention-only
PLaMo 3, while the Mamba-hybrid PLaMo 2 scales worse and **crashes beyond 5 concurrent
sequences** (a second llama.cpp bug, with a deterministic one-command repro).
Commands: [`prefill-decode/commands.md`](prefill-decode/commands.md).

**The through-line:** Samba-style hybrid architectures pay three measured deployment taxes
(speculative-decoding rollback overhead, `ssm_out` quantization fragility, batching ceiling)
that PLaMo 3's return to full attention avoids.

## Repo layout

| Path | What |
|---|---|
| `bench/` | Speculative-decoding benchmark harness (`llama-server`-driven, lossless check) |
| `results/` | Raw per-request benchmark JSONL + server logs |
| `report/` | Study 1: LaTeX source, PDF, interactive HTML, figures |
| `finetune/` | Study 2: data prep, LoRA training (MLflow-tracked), eval, per-question results |
| `report-finetune/` | Study 2 report: LaTeX/PDF (draft), interactive HTML, figures |
| `prefill-decode/` | Study 3: write-up + exact benchmark commands |
| `docs/` | GitHub Pages site (landing page + all reports) |
| `commands-log.md` | Full lab notebook — every command, failure, and fix, chronologically |

## Reproduce the headline results

```bash
# Study 1 smoke test: the ssm_out bug (5 minutes, ungated Apache-2.0 model)
python convert_hf_to_gguf.py pfnet/plamo-2-1b --outfile 1b-BF16.gguf --outtype bf16
llama-quantize 1b-BF16.gguf 1b-Q8.gguf Q8_0
printf '# Fibonacci\ndef fib(n, memo=None):\n' > p.txt
llama-completion -m 1b-Q8.gguf  -ngl 99 -n 32 --temp 0 -f p.txt   # empty (bug)
llama-quantize --tensor-type ssm_out=bf16 1b-BF16.gguf 1b-Q8fix.gguf Q8_0
llama-completion -m 1b-Q8fix.gguf -ngl 99 -n 32 --temp 0 -f p.txt # correct code

# Study 2: see finetune/README.md   |   Study 3: see prefill-decode/commands.md
```

## Licenses

Code here: MIT. PLaMo weights: Apache 2.0 (plamo-2-1b) / [PLaMo Community License](https://huggingface.co/pfnet/plamo-3-nict-2b-base/blob/main/LICENSE/en)
(plamo-2-8b, plamo-3) — the LoRA is a PLaMo derivative ("Built with PLaMo").
Training data: glaive-function-calling-v2 (Apache 2.0).
