"""Giga group-voice Discord bot.

Loads the model fine-tuned by ``general_trainer.py`` (``output/giga`` -- a LoRA
adapter by default, or a full model) and replies in the collective voice of the
Giga friend group. It keeps the last few channel messages as context, formats
them exactly like training (``author: content`` lines under the shared system
prompt), and generates the next message. Speaker names are only ever fed *in*;
a safety net strips any name the model might still try to emit.

Set the bot token via the DISCORD_TOKEN environment variable, or paste it into
``client.run(...)`` below.

Run::

    python discordGiga.py
"""
import os
import re
import json
import random
import asyncio
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque

import discord
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import voice_giga
from giga_common import SYSTEM_PROMPT, render_transcript, strip_name_prefix

# ====================== MODEL CONFIGURATION ======================
# Switch this section when you train a new model.
# For Qwen2.5-7B fine-tune:
ADAPTER_DIR = "output/giga7b_light/checkpoint-3850"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# How hard the LoRA pulls: 0.0 = pure base Qwen, 1.0 = full fine-tune.
# Lower = more coherent, higher = more Giga. Tune this to taste (~0.85-1.0 for max edge).
LORA_WEIGHT = 1.0 # 0.42

# For the old TinyLlama model, use:
# ADAPTER_DIR = "output/giga"
# BASE_MODEL = "models/TinyLlama-1.1B-Chat-v1.0"   # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use the *larger* value here so the deque always keeps enough history.
# We slice per-trigger inside generate_reply.
CONTEXT_TURNS = 12        # max history kept (we slice to 4 on pings, 12 on replies)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset logging: full chat-style examples plus the active interaction
# (the message that pinged/replied to the bot + the AI reply).
DATASET_FILE = "dataset.txt"
INTERACTION_DATASET = "dataset_interactions.txt"
# Spoken interactions (voice transcript + reply) go here, in the same single-turn
# format as INTERACTION_DATASET so they fold into the same training pipeline.
INTERACTION_DATASET_VOICE = "dataset_interactions_voice.txt"

# When True, every time the bot generates a reply it will print the exact context
# it received (with clear visual markers) so you can debug what the model sees.
DEBUG_SHOW_CONTEXT = True
MEMBER_TOOL_MAX_MEMBERS = 200
GIF_TOOL_MAX_RESULTS = 8
GIF_SEARCH_TIMEOUT = 8
KLIPY_API_BASE = "https://api.klipy.com/api/v1"
KLIPY_LOCALE = "en-US"
NATURAL_GIF_ENABLED = True
NATURAL_GIF_COOLDOWN_SECONDS = 600
NATURAL_GIF_CHANCE = 0.20
GIF_DIRECTIVE_PATTERN = r"(?im)^\s*(?:GIF[\s_-]*SEARCH|SEND[\s_-]*GIF|GIF)\s*:\s*(.+?)\s*$"
# Model sometimes hallucinates a fake GIF "search result" line instead of using
# the GIF_SEARCH directive. It shows up in many shapes, e.g.:
#   GIF SearchResult #469: :cry_cat_face: cry_cat_face.jpg
#   GIF Search Result: cat reaction
#   GIF SearchResult(Giphy): <https://media.tenor.com/x/cat-gif.gif>
# Match the whole line no matter how it ends; group 1 is the raw payload, from
# which _gif_query_from_hallucination recovers a usable search query.
GIF_HALLUCINATION_PATTERN = r"(?im)^\s*GIF\s*Search\s*Results?\b[^\n:]*:\s*(.*?)\s*$"
# ================================================================

# 4-bit (nf4) quantization -> ~4-5 GB VRAM, leaves the rest of the GPU free.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---- Load model (LoRA adapter on top of base, or a full fine-tune) ----------
print(f"Loading tokenizer from {ADAPTER_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model (base={BASE_MODEL})...")
if os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    # Scale the adapter's influence so it nudges rather than overpowers the base.
    if LORA_WEIGHT != 1.0:
        n = 0
        for module in model.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                for name in list(scaling):
                    scaling[name] *= LORA_WEIGHT
                    n += 1
        print(f"Scaled {n} LoRA layers to weight {LORA_WEIGHT}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
model.eval()
print("Model loaded successfully!")

# Recent messages per channel, used to build conversational context.
history = defaultdict(lambda: deque(maxlen=CONTEXT_TURNS))
natural_gif_last_sent_at = {}

# Names that refer to the bot itself (used to separate "self" from "other").
BOT_NAME_VARIANTS = {"giga-ai", "giga ai", "giga_ai", "gigaa i", "gigaai"}


def _format_reply_reference(replied_to):
    """Return the Discord message that the target is replying to."""
    return f"{replied_to['author']}: {replied_to['content']}"


def _clean_context_turns(turns):
    """
    Separate "self" (Giga-AI) from "other" in the context window.
    Bot messages are either dropped or clearly labeled so the model
    never mistakes its own past output for a human message.
    """
    cleaned = []
    for t in turns:
        author_lower = t.get("author", "").lower().replace("-", " ").replace("_", " ")
        if author_lower in BOT_NAME_VARIANTS:
            # Label clearly instead of dropping — the model can still learn its own style.
            cleaned.append({
                "author": "Giga-AI (me)",
                "content": t["content"]
            })
        else:
            cleaned.append(t)
    return cleaned


def _debug_print_context(context_turns, target, replied_to, interlocutor, tool_context=None):
    """Pretty-print the exact context the model will receive."""
    print("\n" + "=" * 80)
    print("📡  CONTEXT SENT TO MODEL")
    print("=" * 80)
    print(f"👤 Conversation partner : {interlocutor}")
    if replied_to:
        print(f"↪️  Replying to          : {replied_to['author']}: {replied_to['content'][:80]}"
              + ("..." if len(replied_to['content']) > 80 else ""))
    if tool_context:
        print("🛠️  Tool context         :")
        for line in tool_context.splitlines():
            print(f"   {line}")
    print("-" * 80)
    if context_turns:
        for i, t in enumerate(context_turns, 1):
            prefix = "🤖" if "Giga-AI (me)" in t["author"] else "👤"
            print(f"{prefix} [{i:02d}] {t['author']}: {t['content']}")
    else:
        print("(no prior context)")
    print("-" * 80)
    print(f"🎯 TARGET MESSAGE       : {target['author']}: {target['content']}")
    print("=" * 80 + "\n")


def generate_reply(history, target, replied_to=None, context_turns=None, tool_context=None):
    """Reply aimed at `target` -- the message that pinged or replied to the bot.

    `context_turns` controls how many prior messages to include.
    If the user is replying to another message, that message is shown separately
    from the transcript so the model can use it without copying it as history.
    """
    ctx = context_turns if context_turns is not None else 4
    raw_turns = list(history)[-ctx:]   # keep only the most recent N
    reply_reference_visible = (
        replied_to is not None
        and replied_to.get("id") is not None
        and any(t.get("id") == replied_to["id"] for t in raw_turns)
    )

    # Separate self vs other so the model never confuses its own lines with human input.
    context_turns = _clean_context_turns(raw_turns)

    context_block = render_transcript(context_turns) if context_turns else "(no prior context)"
    reply_reference = (
        _format_reply_reference(replied_to)
        if replied_to and not reply_reference_visible
        else None
    )
    reply_reference_block = (
        f"The target message is replying to:\n{reply_reference}\n\n"
        if reply_reference else ""
    )
    target_line = f"{target['author']}: {target['content']}"

    # "Who am I talking to?" signal – gives the model explicit Theory-of-Mind information.
    interlocutor = target["author"]

    # Debug: show exactly what the model will see (only when flag is enabled).
    if DEBUG_SHOW_CONTEXT:
        _debug_print_context(context_turns, target, replied_to, interlocutor, tool_context)

    # Put the *target* front-and-center so the model replies to it, not the history.
    tool_context_block = (
        f"Live Discord tool context:\n{tool_context}\n\n"
        if tool_context else ""
    )
    user_content = (
        f"Current conversation partner: {interlocutor}\n"
        f"You are Giga-AI. You are talking to {interlocutor} right now.\n\n"
        f"{tool_context_block}"
        f"Group chat transcript (your own past lines are labeled 'Giga-AI (me)'):\n{context_block}\n\n"
        f"{reply_reference_block}"
        f"THE MESSAGE YOU MUST REPLY TO:\n{target_line}\n\n"
        "Reply ONLY to the final human message above. "
        "If live Discord tool context is shown, use it as current server facts or available actions. "
        "If a GIF tool is offered, you CAN send a GIF; the bot code handles the actual sending. "
        "To use it, write your normal reply, then add one command line by itself: `GIF_SEARCH: cat reaction`. "
        "That command line is not spoken to the chat and will be removed before sending. "
        "Never say you do not know how to send a GIF. If you do not want a GIF, reply normally without a GIF_SEARCH line. "
        "If a reply reference is shown, use it only to understand the target; do not repeat it. "
        "ONE short savage reply."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,                       # passes attention_mask too
            max_new_tokens=120,             # keep replies chat-length
            pad_token_id=tokenizer.eos_token_id,
            top_k=50,
            top_p=0.92,
            temperature=0.6,               # higher = more unhinged/edgy
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,         # blocks repeating loops
            do_sample=True,
        )
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True)
    reply = strip_name_prefix(reply).strip()
    reply = re.sub(r"@\S+", "", reply).strip()   # drop @mention / ping spam
    return reply


def log_interaction(context_turns, target, ai_reply, replied_to=None):
    """Append a training example.

    When context is provided (list of turns), it logs the full context + target + reply
    to DATASET_FILE (original training format).
    It also always logs a clean single-turn entry (target + reply only) to
    INTERACTION_DATASET for direct-reply training.

    Bot messages are logged with the "Giga-AI (me)" label so the training data
    never contains ambiguous self-references.
    """
    clean_reply = strip_name_prefix(ai_reply).strip()
    clean_reply = re.sub(r"@\S+", "", clean_reply).strip()

    # Apply the same self/other separation that the prompt uses.
    logged_turns = _clean_context_turns(context_turns) if context_turns else []
    # 1) Full context log (if we have turns)
    try:
        if logged_turns or replied_to:
            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                for t in logged_turns:
                    f.write(f"{t['author']}: {t['content']}\n")
                if replied_to:
                    f.write(f"(replying to {_format_reply_reference(replied_to)})\n")
                f.write(f"{target['author']}: {target['content']}\n")
                f.write(f"{clean_reply}\n\n")
    except Exception as e:
        print(f"[dataset] Failed to log full context: {e}")

    # 2) Clean single-turn interaction (target + AI reply)
    try:
        with open(INTERACTION_DATASET, "a", encoding="utf-8") as f:
            if replied_to:
                f.write(f"(replying to {_format_reply_reference(replied_to)})\n")
            f.write(f"{target['author']}: {target['content']}\n")
            f.write(f"{clean_reply}\n\n")
    except Exception as e:
        print(f"[dataset] Failed to log interaction: {e}")


# ---- Discord wiring ----------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True   # needed to join/track voice channels
client = discord.Client(intents=intents)

# ---- Voice chat (join VC, transcribe speech, reply with TTS) -----------------
# Spoken conversations keep their own short context per guild so the voice
# persona matches the text one without the two histories stepping on each other.
voice_history = defaultdict(lambda: deque(maxlen=CONTEXT_TURNS))


def log_voice_interaction(speaker_name, heard_text, ai_reply):
    """Append a spoken interaction (transcript + reply) to the voice dataset.

    Same single-turn shape as ``log_interaction`` writes to INTERACTION_DATASET:

        Speaker: what they said
        the bot's reply
        <blank line>
    """
    clean_reply = strip_name_prefix(ai_reply or "").strip()
    clean_reply = re.sub(r"@\S+", "", clean_reply).strip()
    try:
        with open(INTERACTION_DATASET_VOICE, "a", encoding="utf-8") as f:
            f.write(f"{speaker_name}: {heard_text}\n")
            f.write(f"{clean_reply}\n\n")
    except Exception as e:
        print(f"[dataset] Failed to log voice interaction: {e}")


def _voice_heard(text, speaker_name, guild_id):
    """Remember every successful voice transcription for future voice context."""
    text = (text or "").strip()
    if not text:
        return
    voice_history[guild_id].append({
        "id": None,
        "author": speaker_name,
        "content": text,
    })


def _voice_reply(text, speaker_name, guild_id):
    """Produce Giga-AI's spoken reply to a transcribed utterance.

    Reuses the exact text-chat generation path so the voice persona is identical.
    Called from voice_giga (already off the event loop), so it can block on the GPU.
    """
    hist = voice_history[guild_id]
    target = {"author": speaker_name, "content": text}
    context = list(hist)
    heard_already = bool(
        context
        and context[-1].get("author") == speaker_name
        and context[-1].get("content") == text
    )
    if heard_already:
        context = context[:-1]

    reply = generate_reply(context, target, context_turns=CONTEXT_TURNS)
    if not heard_already:
        hist.append({"id": None, "author": speaker_name, "content": text})
    if reply:
        hist.append({"id": None, "author": "Giga-AI", "content": reply})
        log_voice_interaction(speaker_name, text, reply)
    return reply


# Speech-to-text vocabulary bias. Whisper has never "heard" the word Giga, so it
# writes it as "Gege/gaga/jigga"; listing it (plus the distinctive member names)
# as an initial prompt pulls transcription toward the right spellings. This fixes
# the wake word AND cleans up the text the model then has to reply to. Override or
# extend it for your own server via the GIGA_VOICE_PROMPT environment variable.
GIGA_VOICE_PROMPT = (
    "Casual voice chat in the Giga Discord group with Giga-AI, also called Giga. "
    "Usual speakers: Lundii, Pete (Trolltusk), Wilson (BitPumpkin) , Jake (Jayteeh), Dakota (Rpgman1234), "
    "Wyatt, Xinny. DarnBannedAgain (Zack), Sybr, and sometimes others. "
)


voice_chat = voice_giga.VoiceChat(
    reply_fn=_voice_reply,
    transcript_fn=_voice_heard,
    whisper_model_size=os.environ.get("WHISPER_MODEL", "small"),
    # Override faster-whisper precision. Use int8 / int8_float16 if a bigger model
    # (e.g. 'small') runs the GPU out of VRAM next to the LLM (it also auto-falls
    # back to int8 on a load failure).
    whisper_compute=os.environ.get("WHISPER_COMPUTE", "int8"),
    # Bias speech-to-text toward in-group vocabulary it would otherwise mangle —
    # above all the wake word "Giga". Override/extend via GIGA_VOICE_PROMPT.
    initial_prompt=os.environ.get("GIGA_VOICE_PROMPT", GIGA_VOICE_PROMPT),
    # Warm/caring adult female default. edge-tts does not support arbitrary
    # mstts:express-as SSML, but these prosody knobs generate the supported
    # Edge <voice><prosody> XML internally.
    tts_voice=os.environ.get("TTS_VOICE", "en-US-JennyNeural"),
    tts_rate=os.environ.get("TTS_RATE", "-8%"),
    tts_pitch=os.environ.get("TTS_PITCH", "-8Hz"),
    tts_volume=os.environ.get("TTS_VOLUME", "+0%"),
    device=DEVICE,
    # Sentence-boundary guessing for voice. A speaker is considered "done"
    # after GIGA_VOICE_SILENCE_SECONDS, then held for GIGA_VOICE_MERGE_PAUSE
    # more seconds so natural mid-thought pauses can be merged before STT/reply.
    silence_seconds=float(os.environ.get("GIGA_VOICE_SILENCE_SECONDS", "0.8")),
    merge_pause_seconds=float(os.environ.get("GIGA_VOICE_MERGE_PAUSE", "2.4")),
    # Drop any utterance that waited longer than this many seconds to be handled,
    # so the bot answers recent speech instead of a growing backlog.
    max_latency_seconds=float(os.environ.get("GIGA_VOICE_MAX_LATENCY", "6.0")),
    # Voice analogue of @-mentioning: only reply when "giga" is spoken, unless
    # GIGA_VOICE_RESPOND_ALL=1 is set (then it answers everything it hears).
    respond_to_all=os.environ.get("GIGA_VOICE_RESPOND_ALL", "0").lower() in ("1", "true", "yes"),
    # Set GIGA_VOICE_DEBUG_DIR=some/folder to save each captured utterance as a
    # WAV (lets you listen to exactly what the bot heard when debugging audio).
    debug_audio_dir=os.environ.get("GIGA_VOICE_DEBUG_DIR") or None,
    # Don't mirror the speech-to-text of what people say into the text channel.
    # Set GIGA_VOICE_POST_TRANSCRIPTS=1 to bring those "🎤 speaker: ..." lines back.
    post_transcripts=os.environ.get("GIGA_VOICE_POST_TRANSCRIPTS", "0").lower() in ("1", "true", "yes"),
    # The bot's spoken reply is still posted as text; set GIGA_VOICE_POST_REPLIES=0 to hide it too.
    post_replies=os.environ.get("GIGA_VOICE_POST_REPLIES", "0").lower() in ("1", "true", "yes"),
)


def _format_author_name(author):
    """Show Discord display name plus stable username for human-readable context."""
    username = getattr(author, "name", None) or str(author)
    display_name = (
        getattr(author, "display_name", None)
        or getattr(author, "global_name", None)
        or username
    )
    return f"{display_name} ({username})"


def _turn_from_message(message):
    """Convert a Discord message into the internal context-turn shape."""
    content = message.clean_content.strip()
    if not content:
        return None

    author = "Giga-AI" if message.author == client.user else _format_author_name(message.author)
    return {
        "id": message.id,
        "author": author,
        "content": content,
    }


def _needs_member_tool(text):
    """Return True when the target asks about server members/people."""
    normalized = text.lower()
    patterns = [
        r"\bmembers?\b",
        r"\beveryone\b",
        r"\bpeople\s+(in|on)\s+(the\s+)?server\b",
        r"\bwho('?s| is)\s+(in|on)\s+(the\s+)?server\b",
        r"\bwho('?s| is)\s+here\b",
        r"\b(best|worst|favorite|favourite)\s+(member|person|user|one)\b",
        r"\b(best|worst|favorite|favourite)\b.*\bserver\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _clean_gif_query(query):
    """Normalize a user/model GIF search query."""
    query = re.sub(r"@\S+", "", query or "")
    query = re.sub(r"\s+", " ", query).strip(" .?!,:;\"'")
    return query[:80] if query else None


def _extract_direct_gif_query(text):
    """Detect obvious user requests to send/search a GIF."""
    if not re.search(r"\bgifs?\b", text, re.IGNORECASE):
        return None
    if not re.search(r"\b(send|post|drop|give|find|search|show|fetch)\b", text, re.IGNORECASE):
        return None

    patterns = [
        r"\bgifs?\s+(?:of|for|about|with)\s+(.+)$",
        r"\b(?:send|post|drop|give|find|search|show|fetch)\b.*?\bgifs?\b(?:\s+(?:of|for|about|with))?\s*(.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = _clean_gif_query(match.group(1))
            return query or "funny reaction"
    return "funny reaction"


def _should_offer_natural_gif(channel_id, target_text):
    """Return True when the model may naturally choose to send a GIF."""
    if not NATURAL_GIF_ENABLED:
        return False

    normalized = (target_text or "").lower().strip()
    if not normalized or len(normalized) > 180:
        return False
    serious_patterns = [
        r"\b(help|explain|debug|fix|error|bug|code|implement|install|setup)\b",
        r"\b(why|how|what is|who is|where|when)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in serious_patterns):
        return False

    now = time.monotonic()
    last_sent = natural_gif_last_sent_at.get(channel_id, 0.0)
    if now - last_sent < NATURAL_GIF_COOLDOWN_SECONDS:
        return False

    return random.random() < NATURAL_GIF_CHANCE


def _gif_query_from_hallucination(payload):
    """Recover a real search query from a hallucinated GIF-result payload.

    ``payload`` is whatever followed the ``...SearchResult...:`` header, e.g.
    ``:cry_cat_face: cry_cat_face.jpg``, ``cat reaction``, or a media URL.
    """
    payload = payload or ""
    # 1) Prefer an :emoji: shortcode (e.g. :cry_cat_face: -> "cry cat face").
    emoji = re.search(r":([a-z0-9_]+):", payload, re.IGNORECASE)
    if emoji:
        return _clean_gif_query(emoji.group(1).replace("_", " ")) or "reaction"
    # 2) If it's a link, use the last path segment (e.g. .../cat-gif.gif -> "cat").
    url = re.search(r"https?://\S+", payload)
    if url:
        slug = url.group(0).rstrip(">").rsplit("/", 1)[-1]
        slug = re.sub(r"\.[a-z0-9]+$", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\bgif\b", "", slug, flags=re.IGNORECASE)
        return _clean_gif_query(slug) or "reaction"
    # 3) Otherwise drop filename extensions and use the remaining words.
    payload = re.sub(r"\.(gif|jpe?g|png|webp|mp4|webm|gifv)\b", " ", payload, flags=re.IGNORECASE)
    payload = re.sub(r"[-_]+", " ", payload)
    return _clean_gif_query(payload) or "reaction"


def _extract_gif_directive(text):
    """Read a hidden model directive like `GIF_SEARCH: cat reaction`."""
    match = re.search(GIF_DIRECTIVE_PATTERN, text or "")
    if match:
        return _clean_gif_query(match.group(1))
    # Model hallucinated a fake GIF "search result" line — recover a query from it.
    hall = re.search(GIF_HALLUCINATION_PATTERN, text or "")
    if hall:
        return _gif_query_from_hallucination(hall.group(1))
    return None


def _strip_gif_directive(text):
    """Remove GIF tool-call lines before sending/logging the reply."""
    text = re.sub(GIF_DIRECTIVE_PATTERN, "", text or "").strip()
    text = re.sub(GIF_HALLUCINATION_PATTERN, "", text).strip()
    return text


def _is_gif_inability_reply(text):
    """Detect model confusion about GIF capability, not a real refusal."""
    normalized = (text or "").lower()
    return bool(
        re.search(r"\b(can'?t|cannot|don'?t know how to|unable to)\b", normalized)
        and re.search(r"\bgifs?\b", normalized)
    )


def _nested_get(data, path):
    """Return a nested value from dictionaries using a dotted path."""
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _extract_klipy_media_url(item):
    """Extract the best sendable media URL from a Klipy content item."""
    paths = [
        "gif",
        "webp",
        "url",
        "thumbnail_url_webp",
        "thumbnail_url",
        "file.md.gif.url",
        "file.md.webp.url",
        "file.sm.gif.url",
        "file.sm.webp.url",
        "file.hd.gif.url",
        "file.hd.webp.url",
        "md.gif.url",
        "md.webp.url",
        "sm.gif.url",
        "sm.webp.url",
        "hd.gif.url",
        "hd.webp.url",
        "file.gif",
        "file.webp",
        "file.url",
        "file.thumbnail_url_webp",
        "file.thumbnail_url",
    ]
    for path in paths:
        value = _nested_get(item, path)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

    slug = item.get("slug")
    item_type = item.get("type", "gif")
    if isinstance(slug, str) and slug:
        return f"https://klipy.com/{item_type}s/{slug}"
    return None


def _search_klipy_gif_sync(query):
    """Blocking Klipy GIF search. Returns a GIF/media URL or None."""
    api_key = os.environ.get("KLIPY_API_KEY")
    client_key = os.environ.get("KLIPY_CLIENT_KEY")
    platform_key = os.environ.get("KLIPY_PLATFORM_KEY")
    auth_key = api_key or client_key
    if not auth_key and not platform_key:
        print("[tool:gifs] KLIPY_API_KEY is not set; paste Klipy's 'Your Key' there")
        return None

    api_base = os.environ.get("KLIPY_API_BASE", KLIPY_API_BASE).rstrip("/")
    params = urllib.parse.urlencode({
        "q": query,
        "locale": os.environ.get("KLIPY_LOCALE", KLIPY_LOCALE),
        "page": 1,
        "per_page": GIF_TOOL_MAX_RESULTS,
    })

    path_keys = []
    # Klipy's dashboard search uses the client key/secret as the URL segment
    # before /gifs/search. The API Keys page labels that visible value "Your Key".
    for value in (client_key, api_key, platform_key):
        if value and value not in path_keys:
            path_keys.append(value)
    candidate_paths = [
        f"/{urllib.parse.quote(path_key, safe='')}/gifs/search?{params}"
        for path_key in path_keys
    ]
    candidate_paths.append(f"/gifs/search?{params}")

    data = None
    last_error = None
    for path in candidate_paths:
        request = urllib.request.Request(
            f"{api_base}{path}",
            headers={"User-Agent": "Giga-AI Discord Bot"},
        )
        if auth_key:
            # Klipy partner keys are managed separately from dashboard bearer tokens.
            # Send the key in common header forms so this works with either auth mode.
            request.add_header("Authorization", f"Bearer {auth_key}")
            request.add_header("x-api-key", auth_key)
        try:
            with urllib.request.urlopen(request, timeout=GIF_SEARCH_TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e
            continue

    if data is None:
        print(f"[tool:gifs] GIF search failed for {query!r}: {last_error}")
        return None

    results = data.get("data", {}).get("data", data.get("data", data.get("results", [])))
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        print(f"[tool:gifs] Unexpected Klipy response shape for {query!r}")
        return None

    urls = [
        url for url in (_extract_klipy_media_url(item) for item in results)
        if url
    ]

    return random.choice(urls) if urls else None


async def _search_klipy_gif(query):
    """Async wrapper for the Klipy GIF search tool."""
    return await asyncio.to_thread(_search_klipy_gif_sync, query)


async def _fetch_member_tool_context(guild):
    """Fetch server members as display-name-plus-username prompt context."""
    if guild is None:
        return None

    members_by_id = {
        member.id: member
        for member in getattr(guild, "members", [])
        if not member.bot
    }

    try:
        async for member in guild.fetch_members(limit=None):
            if not member.bot:
                members_by_id[member.id] = member
    except (discord.Forbidden, discord.HTTPException) as e:
        print(f"[tool:members] Failed to fetch full member list: {e}")

    members = sorted(
        members_by_id.values(),
        key=lambda member: (_format_author_name(member).lower(), member.id),
    )
    if not members:
        return None

    shown = members[:MEMBER_TOOL_MAX_MEMBERS]
    lines = [f"- {_format_author_name(member)}" for member in shown]
    if len(members) > MEMBER_TOOL_MAX_MEMBERS:
        lines.append(f"- ...and {len(members) - MEMBER_TOOL_MAX_MEMBERS} more members")

    return "Server member list, excluding bots:\n" + "\n".join(lines)


async def _fetch_recent_context(message, limit):
    """Fetch the latest real channel history before `message`, oldest-to-newest."""
    if limit <= 0 or not hasattr(message.channel, "history"):
        return []

    turns = []
    fetch_limit = max(limit * 3, limit)
    try:
        async for prior in message.channel.history(
            limit=fetch_limit,
            before=message,
            oldest_first=False,
        ):
            turn = _turn_from_message(prior)
            if turn:
                turns.append(turn)
                if len(turns) >= limit:
                    break
    except (discord.Forbidden, discord.HTTPException) as e:
        print(f"[context] Failed to fetch channel history: {e}")
        return []

    return list(reversed(turns))


def _referenced_message(message):
    """Return the message this one is a reply to (Discord reply), or None."""
    ref = message.reference.resolved if message.reference else None
    return ref if isinstance(ref, discord.Message) else None


def _voice_command(message):
    """Classify a message as a voice 'join'/'leave' request, or None.

    Triggers on bare commands (``!join``/``!leave``/``giga join``/``giga leave``)
    or on an @-mention that mentions joining/leaving a vc/voice/call.
    """
    content = message.content.lower().strip()
    mentioned = client.user.mentioned_in(message)
    wants_vc = any(w in content for w in ("vc", "voice", "call"))

    if content in {"!join", "!vc", "giga join"} or (
        mentioned and "join" in content and wants_vc
    ):
        return "join"
    if content in {"!leave", "!dc", "!disconnect", "giga leave"} or (
        mentioned and ("leave" in content or "disconnect" in content) and wants_vc
    ):
        return "leave"
    return None


async def _handle_voice_command(message, command):
    """Join the speaker's voice channel or leave the current one."""
    if command == "leave":
        await voice_chat.leave(message.guild)
        await message.reply("dipping from vc", mention_author=False)
        return

    voice_state = getattr(message.author, "voice", None)
    channel = voice_state.channel if voice_state else None
    if channel is None:
        await message.reply("you're not even in a vc, genius", mention_author=False)
        return
    try:
        await voice_chat.join(channel, text_channel=message.channel)
        await message.reply(f"pulling up to **{channel.name}** 🎤", mention_author=False)
    except voice_giga.VoiceDepsMissing as exc:
        await message.reply(f"can't do voice — {exc}", mention_author=False)
    except Exception as exc:  # connection/permission failures shouldn't crash the bot
        print(f"[voice] join failed: {exc}")
        await message.reply(f"couldn't join the vc: {exc}", mention_author=False)


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # Record every human message so context builds up over the conversation.
    turn = _turn_from_message(message)
    if turn:
        history[message.channel.id].append(turn)

    # Voice commands take priority over normal chat replies.
    if message.guild is not None:
        command = _voice_command(message)
        if command:
            await _handle_voice_command(message, command)
            return

    if message.mention_everyone:
        return

    # Trigger on a ping OR a Discord reply to the bot.
    ref = _referenced_message(message)
    is_reply_to_bot = ref is not None and ref.author == client.user
    if not (client.user.mentioned_in(message) or is_reply_to_bot):
        return

    # The triggering message is what we answer (strip the bot's own @mention).
    me = message.guild.me.display_name if message.guild else client.user.display_name
    target = {
        "author": _format_author_name(message.author),
        "content": message.clean_content.replace(f"@{me}", "").strip(),
    }

    # Discord replies are useful context even when they point at the bot's own
    # previous message. Label bot references as self so followups like "why?"
    # can anchor to that line without treating it as a fresh human prompt.
    replied_to = None
    if ref is not None:
        replied_to_author = "Giga-AI (me)" if is_reply_to_bot else _format_author_name(ref.author)
        replied_to = {
            "id": ref.id,
            "author": replied_to_author,
            "content": ref.clean_content,
        }

    # History = everything except the triggering message itself. Keep the real
    # timeline intact for generation so the bot sees its own prior messages.
    background = list(history[message.channel.id])[:-1]
    background_for_log = background
    if ref is not None:
        background_for_log = [t for t in background if t.get("id") != ref.id]

    # Different context depth: tight (4) on @pings, fuller (12) on Discord replies.
    ctx_turns = 12 if is_reply_to_bot else 4
    fetched_background = await _fetch_recent_context(message, ctx_turns)
    if fetched_background:
        background = fetched_background
        background_for_log = background
        if ref is not None:
            background_for_log = [t for t in background if t.get("id") != ref.id]

    tool_context_parts = []
    if _needs_member_tool(target["content"]):
        member_context = await _fetch_member_tool_context(message.guild)
        if member_context:
            tool_context_parts.append(member_context)

    direct_gif_query = _extract_direct_gif_query(target["content"])
    natural_gif_offered = False
    if direct_gif_query:
        tool_context_parts.append(
            "GIF tool is available because the user requested a GIF. "
            "You do not need to know how Discord/Klipy works; the bot code will search and attach it. "
            f"To send one, write a normal short reply and add this command on its own line: `GIF_SEARCH: {direct_gif_query}`. "
            "To refuse, reply normally without mentioning tools or saying you cannot send GIFs."
        )
    elif _should_offer_natural_gif(message.channel.id, target["content"]):
        natural_gif_offered = True
        tool_context_parts.append(
            "Optional GIF tool available for a natural reaction. "
            "Use it rarely, only if a reaction GIF strongly fits casual banter or a joke. "
            "The bot code will search and attach it for you. "
            "To send one, write a normal short reply and add a command line like `GIF_SEARCH: laughing reaction`. "
            "Skip it by replying normally without GIF_SEARCH. "
            "Do not use it for serious questions, factual answers, or when it would feel random."
        )
    tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else None

    async with message.channel.typing():
        raw_reply = generate_reply(
            background,
            target,
            replied_to,
            context_turns=ctx_turns,
            tool_context=tool_context,
        )
    gif_query = _extract_gif_directive(raw_reply)
    reply = _strip_gif_directive(raw_reply)

    gif_url = None
    gif_tool_allowed = bool(direct_gif_query or natural_gif_offered)
    if gif_query and not gif_tool_allowed:
        print(f"[tool:gifs] Ignored GIF_SEARCH because the GIF tool was not offered: {gif_query!r}")
        gif_query = None

    if gif_query:
        gif_url = await _search_klipy_gif(gif_query)

    if gif_url and natural_gif_offered and not direct_gif_query:
        natural_gif_last_sent_at[message.channel.id] = time.monotonic()

    if gif_query and not gif_url and direct_gif_query:
        reply = "gif search is cooked rn"
    elif direct_gif_query and not gif_query and _is_gif_inability_reply(reply):
        reply = "nah, not wasting a gif on that"

    if reply or gif_url:
        # Log the interaction (context + target + AI reply).
        reply_for_log = reply if reply else f"[sent GIF: {gif_query}]"
        log_interaction(
            background_for_log[-ctx_turns:] if ctx_turns else background_for_log,
            target,
            reply_for_log,
            replied_to,
        )

        # Reply directly to the message that pinged/replied to the bot, then
        # remember our own line for future context.
        send_content = reply
        if gif_url:
            send_content = f"{send_content}\n{gif_url}" if send_content else gif_url
        sent = await message.reply(send_content, mention_author=False)
        history_content = reply
        if gif_url:
            history_content = f"{reply}\n[sent GIF: {gif_query}]".strip()
        history[message.channel.id].append({
            "id": sent.id,
            "author": "Giga-AI",
            "content": history_content,
        })


client.run(os.environ.get("DISCORD_TOKEN", ""))
