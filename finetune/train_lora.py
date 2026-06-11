#!/usr/bin/env python3
"""LoRA fine-tune of pfnet/plamo-3-nict-2b-base for tool calling, on Apple MPS.

Assistant-only loss masking; official PLaMo 3 chat template via apply_chat_template.
Demo-grade: small data, 1 epoch. Saves adapter to finetune/adapter/.
"""

import json
import time
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "pfnet/plamo-3-nict-2b-base"
MAX_LEN = 768
EPOCHS = 1
LR = 1e-4
GRAD_ACCUM = 8
LORA_R, LORA_ALPHA = 16, 32
CKPT_EVERY = 200  # optimizer steps (~1,600 examples, ~30 min on MPS)

mlflow.set_tracking_uri(f"sqlite:///{HERE.parent / 'mlflow.db'}")
mlflow.set_experiment("plamo3-tool-calling")

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16)
model.to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(HERE / "data" / "train.jsonl")]

def encode(rec):
    msgs = rec["messages"]
    prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)
    full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False)
    full_ids = full_ids + [tok.eos_token_id]
    if len(full_ids) > MAX_LEN:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels

encoded = [e for e in (encode(r) for r in rows) if e]
print(f"training on {len(encoded)}/{len(rows)} examples (rest over {MAX_LEN} tokens)")

run = mlflow.start_run(run_name=f"lora-r{LORA_R}-n{len(encoded)}")
mlflow.log_params({"base_model": MODEL, "n_examples": len(encoded), "epochs": EPOCHS,
                   "lr": LR, "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN,
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "device": device, "dataset": "glaive-function-calling-v2 (dedup, single-turn)"})

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1,
                                          total_iters=max(1, len(encoded) * EPOCHS // GRAD_ACCUM))
model.train()
t0 = time.time()
step = 0
for ep in range(EPOCHS):
    for i, (ids, labels) in enumerate(encoded):
        input_ids = torch.tensor([ids], device=device)
        lab = torch.tensor([labels], device=device)
        out = model(input_ids=input_ids, labels=lab)
        (out.loss / GRAD_ACCUM).backward()
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            mlflow.log_metrics({"loss": out.loss.item(),
                                "lr": sched.get_last_lr()[0],
                                "examples_seen": i + 1 + ep * len(encoded),
                                "minutes_elapsed": (time.time() - t0) / 60}, step=step)
            if step % 5 == 0:
                el = time.time() - t0
                print(f"ep{ep} ex{i+1}/{len(encoded)} step{step} "
                      f"loss={out.loss.item():.3f} {el/60:.1f}min", flush=True)
            if step % CKPT_EVERY == 0:
                model.save_pretrained(HERE / "adapter-ckpt")
                mlflow.log_metric("checkpoint_step", step, step=step)
                print(f"checkpoint saved at step {step} -> adapter-ckpt/", flush=True)

model.save_pretrained(HERE / "adapter")
tok.save_pretrained(HERE / "adapter")
mlflow.log_metric("total_minutes", (time.time() - t0) / 60)
mlflow.log_artifact(str(HERE / "adapter" / "adapter_config.json"))
mlflow.end_run()
print(f"adapter saved to {HERE/'adapter'}; total {(time.time()-t0)/60:.1f} min")
