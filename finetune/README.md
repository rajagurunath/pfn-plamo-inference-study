# Tool-Calling LoRA for PLaMo 3 (proof of concept)

**Goal:** the first open-weights PLaMo with function calling. PFN ships tool calling
only in the API-only PLaMo Prime line; all open PLaMo models are base models with no
chat or tool capability. PLaMo 3's vocabulary, however, contains an undocumented
control-token set clearly designed for chat and structured output
(`<|plamo:tag|>`, `<|plamo:msg|>`, `key`, `val`, `choice`, `constrain`, FIM tokens),
and the base repos ship an official `chat_template.jinja`. This experiment tests how
far a small LoRA on those rails gets, in a single local session on a MacBook (M4 Pro,
48 GB, MPS).

**Published:** sprint adapter + per-question eval records at
[Gurunath/plamo-3-nict-2b-tool-calling-lora](https://huggingface.co/Gurunath/plamo-3-nict-2b-tool-calling-lora).
**In progress:** full-data run (11,182 deduplicated examples, ~45% no-call mix,
checkpointed every 200 steps to `adapter-ckpt/`, loss curves tracked in MLflow —
`mlflow ui --backend-store-uri sqlite:///mlflow.db`, experiment
`plamo3-tool-calling`). Its row will be added to the table below on the same
frozen eval set.

## Setup

- Base model: `pfnet/plamo-3-nict-2b-base` (2.6B params, attention-only, PLaMo Community License)
- Data: single-turn examples derived from `glaiveai/glaive-function-calling-v2`
  (Apache 2.0), query-level deduplicated: system message lists tool schemas;
  assistant answers with a JSON call `{"name":..., "arguments":...}` — or a
  plain answer for no-tool cases (to measure false tool-calls).
  Sprint run: 400 train (12.5% no-call). Full run: 11,182 train (45% no-call).
  Frozen eval: 52 cases, zero query overlap with either training set.
- Format: the **official PLaMo 3 chat template** (`role<|plamo:msg|>content<|plamo:tag|>`).
- Training: LoRA r=16, α=32, all-linear, assistant-only loss masking, 1 epoch,
  bf16, on Apple MPS (`train_lora.py`), MLflow-tracked, periodic checkpoints.
- Eval (`eval_tools.py`), greedy: parse rate, function-name accuracy, argument
  exact-match (40 call cases), false-call rate (12 no-call cases); every
  question/gold/prediction saved per condition in `results/`.

## Results

40 call cases + 12 no-call cases, all with **zero query overlap** with training data:

| Condition | Parse rate | Func-name acc | Args exact | False-call rate |
|---|---|---|---|---|
| base, zero-shot | 35.0% | 32.5% | **0.0%** | 0.0%¹ |
| base, 2-shot | 67.5% | 30.0% | **0.0%** | 41.7% |
| **+ LoRA, 400 ex. (sprint)** | **92.5%** | **92.5%** | **90.0%** | 41.7% |
| + LoRA, 11,182 ex. (full) | *training…* | *training…* | *training…* | *training…* |

¹ trivially low — the zero-shot base model rarely emits a call at all.

**Headline:** a 5-minute LoRA (400 examples, M4 Pro MPS) takes argument-exact tool
calling from 0% to 90% on unseen queries. Prompting alone cannot do this: few-shot
raises JSON-shaped output but not correctness, while making the model fire tools on
41.7% of questions that need none.

**Eval-contamination incident (kept for honesty):** the first eval measured 97.5%
across the board — too good. Investigation showed 41/52 eval queries appeared
verbatim in training, because glaive-function-calling-v2 contains massive row
duplication (after query-level dedup, the entire dataset has only ~71 unique
unseen call queries beyond our training set). The eval was rebuilt with zero
query overlap (results above); the contaminated run is archived in
`results/leaked-eval/`. Function names still overlap between train and eval
(same tools, different queries/arguments) — the eval therefore measures
in-distribution generalization, not novel-tool generalization (that would need
BFCL-style held-out schemas).

**Known weakness:** over-eager calling (41.7% false-call rate), unchanged from
few-shot. Expected with only 50 no-call training examples (12.5% of data);
raising that fraction is the obvious next fix.

*(Qualitative note from the pre-training probe: the base model already follows the
chat template — it answers helpfully in role format — but ignores tool-call
instructions, e.g. it explains how to use a weather API rather than emitting a call.)*

## Honest limitations

Demo-grade by design: 400 training examples, single-turn only, English-only data,
one seed, exact-match argument scoring (no partial credit), no BFCL evaluation.
The point is direction and feasibility on PFN's own template rails, not a
production model.

## Reproduce

```bash
python prep_data.py        # build data/{train,eval}.jsonl from glaive-v2
python train_lora.py       # ~40 min on M4 Pro MPS -> adapter/
python eval_tools.py --condition base
python eval_tools.py --condition base-fewshot
python eval_tools.py --condition lora
```
