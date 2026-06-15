"""Quick offline test of the trained Giga model -- no Discord needed.

Loads output/giga and lets you type chat context and see the bot's reply, using
the exact same prompt format and generation as discordGiga.py. Great for sanity-
checking the personality (e.g. straight on the training box before you destroy it).

Run::

    python try_giga.py
    context> jayteeh: anyone up for ranked?
    giga> ...
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from giga_common import SYSTEM_PROMPT, strip_name_prefix

# Path to test: a CLI arg, else the final model. Can be a mid-run checkpoint,
# e.g.  python try_giga.py output/giga/checkpoint-2000
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/giga"
BASE_MODEL = "models/TinyLlama-1.1B-Chat-v1.0"  # for tokenizer fallback / LoRA base
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Intermediate checkpoints don't include the tokenizer -- fall back to the base
# model's (it's identical).
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
except OSError:
    print(f"[info] no tokenizer in {MODEL_DIR}; using base model's tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL), MODEL_DIR)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()
print(f"loaded {MODEL_DIR} on {DEVICE}")


def reply(context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_text},
    ]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=True, top_p=0.95,
                             temperature=0.8, repetition_penalty=1.3,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return strip_name_prefix(text).strip()


if __name__ == "__main__":
    print("Type chat context (e.g.  jayteeh: anyone up?). Blank line quits.")
    while True:
        try:
            line = input("\ncontext> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        print("giga>", reply(line))
