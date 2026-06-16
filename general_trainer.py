"""Fine-tune Gemma on the Giga group's chat to imitate its collective voice.

Pipeline position::

    parser.py  ->  data/conversations.jsonl  ->  [this]  ->  output/giga

``data/conversations.jsonl`` from ``parser.py`` is used by default. Raw
``dataset.txt`` is still accepted if passed with ``--input dataset.txt``.

Why chat-format SFT with loss masking?
  We want the model to know *who said what* (so it learns the group's turn-taking
  and dynamics) but to *never emit a speaker name* in its own replies. So every
  training example is:

      system   : the Chat persona prompt (see giga_common.SYSTEM_PROMPT)
      user     : the recent transcript, rendered "author: content" per line
      assistant: the bare next message (no name)

  The loss is masked over everything except the assistant message, so the model is
  only ever trained to produce name-free replies. Speaker labels appear solely in
  the (masked) context.

A held-out validation split is evaluated during training for monitoring. By
default the *final* model is kept (not the lowest-eval-loss one): for a radical
personality transfer you actually want to ride the run to the end and let it
over-fit onto Giga. Pass ``--keep-best`` to restore the best checkpoint instead.

LoRA (``--lora-r``) is the memory-safe default. For the most radical override use
``--no-lora`` (full fine-tune); on 16 GB it fits via gradient checkpointing + bf16
+ a memory-light optimizer (adafactor, or 8-bit Adam if bitsandbytes is present).

Run::

    python general_trainer.py --max-steps 50                 # throughput probe
    python general_trainer.py --no-lora --epochs 3           # radical full fine-tune
    python general_trainer.py --lora-r 64 --epochs 4         # radical LoRA fallback
"""
import argparse
import importlib.util
import inspect
import json
import os
import random

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from giga_common import SYSTEM_PROMPT, render_transcript

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing_path(*paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    return None


# Prefer a locally-downloaded copy; fall back to the Hugging Face Hub id.
_LOCAL_MODEL = _first_existing_path(
    "models/gemma-4-e4b-it",
    "models/gemma-4-E4B-it",
    os.path.join(SCRIPT_DIR, "models/gemma-4-e4b-it"),
    os.path.join(SCRIPT_DIR, "models/gemma-4-E4B-it"),
)
MODEL_NAME = _LOCAL_MODEL or "google/gemma-4-e4b-it"


def resolve_existing_file(path):
    """Allow defaults to work from either repo root or this file's directory."""
    if os.path.exists(path):
        return path
    script_relative = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(script_relative):
        return script_relative
    return path


def looks_like_jsonl(path):
    if path.endswith((".jsonl", ".json")):
        return True
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.lstrip()
            if stripped:
                return stripped.startswith("{")
    return False


def load_jsonl_sessions(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["turns"] for line in fh if line.strip()]


def load_raw_export_sessions(path, args):
    from parser import build_sessions as build_raw_sessions
    from parser import parse_messages

    messages = list(parse_messages(path, set(args.drop_authors), args.min_chars))
    sessions = build_raw_sessions(messages, args.gap_minutes, args.merge_minutes,
                                  args.max_turns)
    raw = getattr(parse_messages, "raw_count", 0)
    print(f"raw message headers : {raw}")
    print(f"kept messages       : {len(messages)}")
    print(f"sessions built      : {len(sessions)}")
    return sessions


def load_sessions(path, args):
    path = resolve_existing_file(path)
    if looks_like_jsonl(path):
        print(f"Loading pre-parsed JSONL sessions from {path}")
        return load_jsonl_sessions(path)
    print(f"Parsing raw Discord export from {path}")
    return load_raw_export_sessions(path, args)


def build_examples(sessions, context_turns):
    """Yield (context_turns_list, target_content) sliding over each session."""
    for turns in sessions:
        for i in range(1, len(turns)):
            context = turns[max(0, i - context_turns):i]
            target = turns[i]["content"]
            if target:
                yield context, target


def tokenizer_accepts_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hello"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception as exc:
        print(f"[info] chat template does not accept a system role; "
              f"folding system prompt into the user message ({exc})")
        return False


def make_prompt_messages(context, use_system_role):
    transcript = render_transcript(context)
    if use_system_role:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
    return [{
        "role": "user",
        "content": f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript}",
    }]


def make_tokenize_fn(tokenizer, max_len):
    """Build the function that turns one example into masked input_ids/labels.

    We render the prompt (persona + context, with the assistant generation cue)
    and construct a full chat sequence with the target assistant reply. Only the
    reply tokens carry loss.
    """
    eos = tokenizer.eos_token or ""
    use_system_role = tokenizer_accepts_system_role(tokenizer)

    def tokenize(context, target):
        prompt_msgs = make_prompt_messages(context, use_system_role)
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            full_ids = tokenizer(prompt_text + target + eos,
                                 add_special_tokens=False)["input_ids"]

        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(full_ids))):
            labels[j] = -100
        if len(full_ids) > max_len:        # keep the tail so the reply survives
            full_ids = full_ids[-max_len:]
            labels = labels[-max_len:]
        if all(t == -100 for t in labels):
            return None  # nothing left to learn from
        return {"input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels}
    return tokenize


def build_dataset(sessions, tokenizer, context_turns, max_len, max_examples, label="train"):
    tokenize = make_tokenize_fn(tokenizer, max_len)
    rows = []
    for n, (context, target) in enumerate(build_examples(sessions, context_turns), 1):
        row = tokenize(context, target)
        if row is not None:
            rows.append(row)
        if n % 20000 == 0:
            print(f"  tokenizing {label}... {len(rows)} examples", flush=True)
        if max_examples and len(rows) >= max_examples:
            break
    return Dataset.from_list(rows)


def find_lora_target_modules(model):
    """Return exact text-tower Linear module names for PEFT LoRA injection."""
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"}
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if ".language_model." not in name:
            continue
        if name.rsplit(".", 1)[-1] in wanted:
            targets.append(name)

    if targets:
        print(f"LoRA target modules : {len(targets)} language_model Linear layers")
        return targets

    # Fallback for plain text-only architectures with unwrapped projections.
    fallback = sorted({
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in wanted
    })
    if not fallback:
        raise RuntimeError("Could not find any supported Linear modules for LoRA")
    print(f"LoRA target modules : {fallback}")
    return fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/conversations.jsonl",
                    help="pre-parsed conversations JSONL, or raw Discord export")
    ap.add_argument("--output-dir", default="output/giga")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="base model id/path")
    ap.add_argument("--context-turns", type=int, default=8,
                    help="how many prior turns to feed as context")
    ap.add_argument("--gap-minutes", type=int, default=15,
                    help="raw dataset.txt only: gap above which a new session starts")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="raw dataset.txt only: hard cap on turns per session")
    ap.add_argument("--merge-minutes", type=int, default=2,
                    help="raw dataset.txt only: merge same-author bursts in this window")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="raw dataset.txt only: drop shorter cleaned messages")
    ap.add_argument("--drop-authors", nargs="*", default=[],
                    help="raw dataset.txt only: usernames to exclude entirely")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="defaults to 2e-4 for LoRA, 2e-5 for full fine-tune")
    ap.add_argument("--lora-r", type=int, default=64,
                    help="LoRA rank: higher = more capacity/fidelity (and params)")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (defaults to 2 x rank)")
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="fraction of sessions held out for validation (0 disables)")
    ap.add_argument("--max-val-examples", type=int, default=2000,
                    help="cap validation examples so eval stays fast")
    ap.add_argument("--eval-steps", type=int, default=1000,
                    help="evaluate + checkpoint every N steps")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="cap training examples (0 = all); handy for smoke tests")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap optimizer steps (-1 = full run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lora", action="store_true",
                    help="full fine-tune instead of LoRA (the most radical option)")
    ap.add_argument("--optim", default="auto",
                    help="optimizer; 'auto' picks adamw_torch (LoRA) or a "
                         "memory-light one (full fine-tune)")
    ap.add_argument("--grad-checkpointing", choices=["auto", "on", "off"],
                    default="auto",
                    help="trade compute for memory; auto = on for full fine-tune")
    ap.add_argument("--keep-best", action="store_true",
                    help="restore the lowest-eval-loss checkpoint instead of the "
                         "final (less radical, avoids over-fitting)")
    args = ap.parse_args()

    use_lora = not args.no_lora
    lr = args.lr if args.lr is not None else (2e-4 if use_lora else 2e-5)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    # bf16 is more stable than fp16 for a full fine-tune and is supported on Ada
    # (RTX 40-series). Fall back to fp16, then fp32 on CPU.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    grad_ckpt = {"auto": not use_lora, "on": True, "off": False}[args.grad_checkpointing]

    if args.optim != "auto":
        optim = args.optim
    elif use_lora:
        optim = "adamw_torch"
    elif importlib.util.find_spec("bitsandbytes") is not None:
        optim = "paged_adamw_8bit"      # best quality if bitsandbytes is installed
    else:
        optim = "adafactor"             # memory-light, no extra deps (Windows-safe)

    model_name = args.model
    print(f"base model         : {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We truncate manually to --max-len, so silence the "longer than 2048" notice.
    tokenizer.model_max_length = int(1e9)

    # Split sessions (not examples) so validation context never leaks into training.
    sessions = load_sessions(args.input, args)
    random.Random(args.seed).shuffle(sessions)
    n_val = int(len(sessions) * args.val_frac)
    val_sessions, train_sessions = sessions[:n_val], sessions[n_val:]

    print("Building dataset...")
    train_ds = build_dataset(train_sessions, tokenizer, args.context_turns,
                             args.max_len, args.max_examples)
    val_ds = None
    if n_val > 0:
        val_ds = build_dataset(val_sessions, tokenizer, args.context_turns,
                               args.max_len, args.max_val_examples, label="val")
    print(f"training examples  : {len(train_ds)}")
    print(f"validation examples: {len(val_ds) if val_ds is not None else 0}")

    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_lora_target_modules(model),
        )
        model = get_peft_model(model, lora)
        if grad_ckpt:
            model.enable_input_require_grads()  # required for PEFT + checkpointing
        model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True,
                                      label_pad_token_id=-100)

    do_eval = val_ds is not None and len(val_ds) > 0
    ta_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0 if use_lora else 0.01,
        optim=optim,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=do_eval and args.keep_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )
    # Drop kwargs the installed transformers version doesn't accept (arg names
    # like overwrite_output_dir get removed/renamed across releases).
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    unsupported = [k for k in ta_kwargs if k not in valid]
    if unsupported:
        print(f"[info] transformers ignoring unsupported args: {unsupported}")
    training_args = TrainingArguments(**{k: v for k, v in ta_kwargs.items()
                                         if k in valid})

    trainer = Trainer(model=model, args=training_args, data_collator=collator,
                      train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    trainer.save_model(args.output_dir)        # LoRA adapter, or full model
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
