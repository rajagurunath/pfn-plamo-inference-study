# Commands Log — PLaMo 2 Speculative Decoding Study

Lab notebook of every command tried so far, with outcomes. Two environments:
**Colab** (T4 GPU runtime, free tier) and **Local** (MacBook M4 Pro, 48 GB, macOS 15.7).

---

## 1. Colab environment (T4, CUDA 13.0 driver, 12 GB RAM, ~113 GB disk)

### 1.1 Environment check ✅

```bash
!nvidia-smi | head -12        # Tesla T4, 15.4 GB VRAM, driver 580.82.07 / CUDA 13.0
!df -h / | tail -1            # 66 GB free at start
!free -g | head -2            # 12 GB RAM
```

### 1.2 Install llama.cpp CUDA build from conda-forge ✅ (install) / ❌ (unusable on T4 — see 1.6)

No prebuilt Linux CUDA binaries exist in llama.cpp GitHub releases (Windows only).
conda-forge ships CUDA builds including `llama-speculative`, `llama-lookup`, etc.

```bash
cd /content
export MAMBA_ROOT_PREFIX=/content/mamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
./bin/micromamba create -y -p /content/llamacpp -c conda-forge "llama.cpp=9574=cuda130*"

# llama.cpp source at matching tag, for the conversion script
git clone --quiet --depth 1 --branch b9574 https://github.com/ggml-org/llama.cpp /content/llama.cpp-src
pip install --quiet /content/llama.cpp-src/gguf-py sentencepiece
```

Result: installed fine; binaries present (`llama-speculative llama-speculative-simple
llama-lookup llama-quantize llama-bench ...`).

### 1.3 Download models ✅ (after auth fixes)

```python
import os
from huggingface_hub import snapshot_download

pats = ["*.safetensors*", "*.json", "*.jsonl", "*.txt", "*.model", "*.py", "tokenizer*"]
for repo in ["pfnet/plamo-2-1b", "pfnet/plamo-2-8b"]:
    snapshot_download(repo, allow_patterns=pats,
                      local_dir=f"/content/models/{repo.split('/')[-1]}")
```

Auth journey (plamo-2-8b is gated, `gated: auto`):
1. ❌ No token → `GatedRepoError 401`. Fix: accept PLaMo Community License on the
   model page (logged-in browser).
2. ❌ Colab Secrets + `userdata.get("HF_TOKEN")` → `SecretNotFoundError`, then after
   adding the secret: `TimeoutException: Secrets can only be fetched when running
   from the Colab UI` (cells driven via colab-mcp cannot read Secrets).
3. ✅ Workaround: temporary cell `os.environ["HF_TOKEN"] = "hf_..."`, run once,
   **delete the cell immediately** (never leave a token in a shareable notebook).

Sizes: plamo-2-1b = 4.9 GB; plamo-2-8b = **34 GB (float32 shards, not bf16!)**.

### 1.4 Convert to GGUF Q8_0 ✅

```bash
mkdir -p /content/gguf
for m in plamo-2-1b plamo-2-8b; do
  python /content/llama.cpp-src/convert_hf_to_gguf.py /content/models/$m \
    --outfile /content/gguf/$m-Q8_0.gguf --outtype q8_0
done
```

Result: `plamo-2-1b-Q8_0.gguf` = 1.3 GB, `plamo-2-8b-Q8_0.gguf` = 9.1 GB.
The PLaMo 2 (plamo2 / Samba hybrid) conversion path works out of the box.

### 1.5 CUDA smoke test ❌ — killed the runtime

```bash
/content/llamacpp/bin/llama-cli -m /content/gguf/plamo-2-1b-Q8_0.gguf \
  -ngl 99 -n 48 --temp 0 -no-cnv --no-display-prompt -p "日本で一番高い山は"
```

Result: hung ~10 min, then **"Your session crashed after using all available RAM"**.

### 1.6 Root cause (Colab)

The conda-forge `cuda130` package contains no precompiled sm_75 (T4) kernels —
only PTX. The CUDA driver tried to JIT-compile llama.cpp's entire kernel set at
startup, which exhausted the 12 GB system RAM. Also "Disk is almost full"
(98/113 GB) from the 34 GB float32 download + GGUFs.

Fixes for the next Colab session (Phase 5 of plan-detailed.md):
- `File → Save a copy in Drive` first (we worked in an unsaveable "scratchpad" —
  lost cells twice).
- Try `"llama.cpp=9574=cuda129*"` (may include sm_75 SASS), else source-build
  with `-DCMAKE_CUDA_ARCHITECTURES=75`.
- Delete safetensors right after conversion (`rm -rf /content/models/plamo-2-8b`).
- Better: upload GGUFs to a personal HF repo once; later sessions download those.

---

## 2. Local environment (M4 Pro, 48 GB RAM, macOS 15.7)

### 2.1 transformers attempt (try.py) ❌

```python
pipeline = transformers.pipeline("text-generation", model="pfnet/plamo-2-1b",
                                 trust_remote_code=True)
# AttributeError: 'list' object has no attribute 'keys'
# (modeling_utils.get_expanded_tied_weights_keys)
```

Cause: PLaMo 2's custom `modeling_plamo.py` pins `transformers ≤ 4.57`; the venv
has transformers 5.11. Fix if ever needed: `uv add 'transformers<4.58'`.
Not needed for the llama.cpp path. (Silver lining: the 4.8 GB weights landed in
the local HF cache and were reused below.)

### 2.2 Install llama.cpp prebuilt macOS arm64 binaries ✅

```bash
curl -sL https://github.com/ggml-org/llama.cpp/releases/download/b9596/llama-b9596-bin-macos-arm64.tar.gz | tar -xz
./llama-b9596/llama-cli --version   # version: 9596 (18ef86ece), AppleClang, arm64
```

Note: the macOS release package has **no `llama-speculative`** binary; it does
have `llama-server` (which supports `--model-draft` + per-request draft stats in
JSON timings) — the benchmark harness will drive `llama-server` instead.

### 2.3 Conversion tooling ✅ (two gotchas)

```bash
git clone --quiet --depth 1 --branch b9596 https://github.com/ggml-org/llama.cpp llama.cpp-src
# Gotcha 1: plain `uv pip install` targeted the anaconda env, not the project venv:
uv pip install --python .venv/bin/python ./llama.cpp-src/gguf-py sentencepiece
```

```python
# Gotcha 2: the transformers pipeline had NOT downloaded tokenizer.jsonl /
# tokenizer_config.json (PLaMo custom tokenizer files). Complete the snapshot:
from huggingface_hub import snapshot_download
snapshot_download('pfnet/plamo-2-1b')   # reuses cached 4.8 GB, adds small files
```

### 2.4 Convert to GGUF Q8_0 ✅ — but caused a kernel panic on a full disk ⚠️

```bash
mkdir -p gguf
.venv/bin/python llama.cpp-src/convert_hf_to_gguf.py \
  ~/.cache/huggingface/hub/models--pfnet--plamo-2-1b/snapshots/92c75fd6... \
  --outfile gguf/plamo-2-1b-Q8_0.gguf --outtype q8_0
```

Result: `gguf/plamo-2-1b-Q8_0.gguf` = 1.3 GB, file verified intact.
⚠️ The first conversion ran while the disk was ~100% full (2.5 GB free) → macOS
**kernel panic**: `watchdog timeout: no checkins from watchdogd in 92 seconds`.
Rule adopted: **keep ≥ 20 GB free before any heavy I/O.**

### 2.5 Metal smoke test — wrong binary ❌, then ✅

```bash
# ❌ b9596 llama-cli is chat-only now; it silently ignored -no-cnv and entered an
# interactive REPL loop against closed stdin, spewing "> " (1.9 GB log before kill):
./llama-b9596/llama-cli -m gguf/plamo-2-1b-Q8_0.gguf -ngl 99 -n 48 --temp 0 -no-cnv ...

# ✅ plain completion moved to its own binary in this build:
./llama-b9596/llama-completion -m gguf/plamo-2-1b-Q8_0.gguf \
  -ngl 99 -c 2048 -n 48 --temp 0 --no-display-prompt \
  -p "日本で一番高い山は" </dev/null
```

Output: `富士山で標高は3,776mです。…` (correct, coherent Japanese).

**Timings (M4 Pro, Metal, Q8_0, 1B):**
- decode: **133.96 tok/s** (7.46 ms/token)
- prefill: 64.10 tok/s
- load: 879 ms

→ Phase 1 exit criterion (>100 tok/s decode) met.

---

## 3. The Q8_0 `ssm_out` bug hunt (the big finding)

Symptom: in the first benchmark run, all prompts containing a bare `\n` (code,
translation) returned **empty content** while generating 256 tokens at full
speed — both 1B and 8B targets.

Investigation chain (each step = one experiment):

1. `return_tokens: true` → the model emits token **31 = `<|plamo:reserved:0x1F|>`**
   repeatedly; reserved tokens render as empty text and are not EOG, so
   generation runs to the limit.
2. Tokenizer round-trip (`/tokenize` + `/detokenize`) — **perfect**, including
   code with indentation. Not a tokenizer bug.
3. Full prompt token-id comparison HF vs llama.cpp — **identical** (modulo BOS;
   prepending BOS manually changed nothing).
4. Ground truth via HF transformers on CPU (required pinning
   `transformers<4.58` AND writing pure-PyTorch shims for `causal_conv1d` and
   `mamba_ssm` reference functions — pfnet's modeling code's CPU fallback still
   calls those packages): **HF generates correct code** for the failing prompt.
   → llama.cpp-side problem.
5. bf16 GGUF (unquantized) in llama.cpp: **correct output** → quantization, not
   inference.
6. `llama-quantize` Q8_0 from bf16: still broken → not a converter bug.
7. Per-tensor isolation with `--tensor-type X=bf16`:
   - token embeddings bf16: broken; output tensor bf16: broken
   - all `ssm_*` mats bf16: **fixed**
   - one at a time: `ssm_in` ✗, `ssm_x` ✗, `ssm_dt` ✗, **`ssm_out` ✓ alone fixes it**

**Conclusion:** quantizing PLaMo 2's Mamba output projection (`ssm_out`) to
Q8_0 causes degenerate generation on newline-containing prompts. Mitigation:

```bash
llama-quantize --tensor-type ssm_out=bf16 model-BF16.gguf model-Q8fixed.gguf Q8_0
```

Likely affects all community PLaMo 2 GGUFs. Upstream issue/PR candidate:
protect `ssm_out` in llama.cpp's plamo2 quantization recipe.

## 4. Harness bugs found & fixed along the way

- `llama-server` b9596 needs `--spec-type draft-simple` — `-md` alone loads the
  draft model but **speculation stays off** (log: "no implementations specified
  for speculative decoding"). The first spec sweep measured nothing; archived to
  `results/archive-broken-q8/`. With spec engaged, `timings.draft_n` /
  `draft_n_accepted` appear in responses.
- `bench.py` content hash used Python `hash()` (salted per process →
  meaningless across runs); now sha256 + full content stored.
- n-gram comparison: `--spec-type ngram-simple` (no draft model needed).

## 5. Tool-calling LoRA sprint (plamo-3-nict-2b, local MPS)

Discovery: PLaMo 3 base repos ship an official `chat_template.jinja`
(`role<|plamo:msg|>content<|plamo:tag|>`) and the vocab contains undocumented
chat/structured-output/FIM control tokens (`tag,msg,key,val,choice,constrain,
fim_*`) — none of this exists in PLaMo 2. PLaMo 3's modeling code is pure
PyTorch (no mamba_ssm/causal_conv1d) → runs on Apple MPS directly.

```bash
# env: /tmp/plamo-hf venv (transformers<4.58) + peft datasets
python finetune/prep_data.py      # 400 train / 52 eval from glaive-function-calling-v2
python finetune/eval_tools.py --condition base          # before, zero-shot
python finetune/eval_tools.py --condition base-fewshot  # before, 2-shot
python finetune/train_lora.py     # LoRA r16 all-linear, asst-only masking, 5 min on MPS
python finetune/eval_tools.py --condition lora          # after
```

Gotchas hit: (a) background shell inherited a stray `cd finetune` → first eval
run failed on doubled paths; (b) **eval contamination** — glaive's heavy row
duplication put 41/52 eval queries verbatim in training; first LoRA eval read
97.5% across the board. Rebuilt eval with query-level dedup → honest result:
**0% → 90% args-exact tool calling** (table in finetune/README.md).

## 6. Current status & next step

- Phase 1 ✅; harness ✅ (with spec engaged + sha256 lossless check);
  fixed 1B quant ✅ (`gguf/plamo-2-1b-Q8fixed.gguf`).
- The 8B Q8_0 GGUF is **broken** (built before the ssm_out discovery) and the
  fp32 source was deleted for space → must re-download 34 GB and rebuild as
  bf16 → Q8fixed. Blocked on ~26 GB more free disk (candidates: old HF model
  caches — whisper-large-v3 7.8G, Qwen3-TTS 4.2G, moondream2 3.6G, gemma 3.4G,
  chatterbox 3.0G, TinyLlama 2.1G, Qwen3-ASR 1.8G).
- Then: re-run baseline + spec sweep (d=2/4/8/16) + ngram-simple on fixed
  quants, lossless check, analysis, report.
