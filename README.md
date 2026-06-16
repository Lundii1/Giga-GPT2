# Giga-GPT2

A Discord friend-group AI built around **Gemma 4 E4B-it**. It fine-tunes the
model on exported server conversations, then runs it as a live Discord bot with
text chat, image analysis, optional reasoning mode, GIF reactions, and voice chat.

---

## Overview

**Giga-GPT2** trains a LoRA adapter on a Discord server export so the bot learns
the server's shared tone, slang, in-jokes, and conversational rhythm as one
blended "Giga" voice.

The current bot uses **Gemma 4 E4B-it** as the base model, with the adapter loaded
from `output/giga/checkpoint-6000` by default. The trainer builds chat-style
examples from `data/conversations.jsonl`, masks the context tokens, and trains
only the assistant reply. Speaker names are included as context so the model can
track who is talking, but generated name prefixes are stripped before Discord
receives the reply.

The bot now supports more than plain text:

- **Image analysis:** when the bot is pinged with an attached image, it switches to
  image-only mode, drops previous chat context, and asks Gemma to answer from the
  image pixels instead of the filename or surrounding conversation.
- **Reasoning mode:** objective tasks like math, coding/debugging, logic puzzles,
  seating/constraint problems, planning, and comparisons automatically trigger a
  slower Gemma thinking pass. Subjective banter like "who is the best member" stays
  in normal persona mode.
- **Voice chat:** the bot can join a voice channel, transcribe speech, generate a
  persona reply, and speak back with TTS.
- **GIF tool:** the model can optionally request a GIF reaction with a hidden
  `GIF_SEARCH:` directive; the bot performs the search and attaches the result.

---

## Pipeline

```text
dataset.txt -> parser.py -> data/conversations.jsonl -> general_trainer.py -> output/giga -> discordGiga.py
 raw export    clean/sessionize       training data          Gemma LoRA         adapter      Discord bot
```

| File | Role |
|------|------|
| `parser.py` | Cleans the raw Discord export and turns it into JSONL conversation sessions. |
| `general_trainer.py` | Builds masked chat-format examples and fine-tunes Gemma 4 E4B-it with LoRA. |
| `discordGiga.py` | Loads Gemma 4 E4B-it plus the LoRA adapter and runs the Discord bot. |
| `giga_common.py` | Shared system prompt, username aliases, transcript formatting, and cleanup helpers. |
| `voice_giga.py` | Optional voice-channel support: receive audio, transcribe speech, and play TTS replies. |
| `requirements-voice.txt` | Extra packages for voice receive, STT, and TTS. |

---

## Installation

```bash
git clone https://github.com/Lundii1/Giga-GPT2.git
cd Giga-GPT2
pip install torch transformers accelerate datasets peft discord.py pyarrow pillow
```

For voice support, install the voice extras too:

```bash
pip install -U --pre -r requirements-voice.txt
```

You also need **FFmpeg** on your PATH for voice playback.

---

## Usage

### 1. Parse Discord Logs

```bash
python parser.py --input dataset.txt --output data/conversations.jsonl
```

`dataset.txt` is the raw export. `data/conversations.jsonl` is the preferred
training input and is what `general_trainer.py` uses by default.

### 2. Fine-Tune Gemma

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python general_trainer.py \
  --epochs 3 \
  --batch-size 8 \
  --grad-accum 4 \
  --lora-r 64 \
  --max-len 1024 \
  --eval-steps 2000 \
  --val-frac 0.02
```

The default base model is the local `models/gemma-4-e4b-it` directory when
present, otherwise `google/gemma-4-e4b-it`. The default dataset is
`data/conversations.jsonl`; raw text input still works with `--input dataset.txt`.

### 3. Run the Discord Bot

```bash
export DISCORD_TOKEN="your-bot-token"
python discordGiga.py
```

Mention the bot or reply to one of its messages to trigger a response.

---

## Discord Bot Features

### Text Persona

The bot keeps a short channel history window, labels its own old messages as
`Chat (me)`, and replies in the tuned Giga persona. On normal pings it keeps the
context tight; on replies it can use more context so follow-up messages make
sense.

### Image Analysis

Attach an image while pinging the bot and it will use Gemma's multimodal processor
through `AutoModelForMultimodalLM` / `AutoProcessor`.

When images are present:

- prior chat context is removed;
- member, GIF, and reasoning tools are disabled;
- the prompt tells the model to prioritize visual accuracy and OCR;
- the debug log prints `Images attached: N` when image bytes were loaded.

### Reasoning Mode

Reasoning is routed by the bot, not by Discord. The bot detects objective prompts
and enables Gemma's thinking mode with `enable_thinking=True`.

It triggers for things like:

- arithmetic and math expressions;
- coding, debugging, algorithms, and errors;
- logic puzzles and constraint problems;
- prompts asking to determine all possible arrangements;
- planning, tradeoff analysis, and precise comparisons.

It intentionally does **not** trigger for greetings, roasts, simple opinions,
image questions, or subjective server-member ranking questions.

Useful console logs:

```text
[tool:reasoning] offered=True objective=True subjective=False
[tool:reasoning] Standalone task detected; dropping prior Discord context
[tool:reasoning] Auto reasoning triggered from user request; enable_thinking=True
```

Reasoning token budgets are configurable:

| Variable | Default | Effect |
|----------|---------|--------|
| `GIGA_REASONING_MAX_NEW_TOKENS` | `2200` | Main reasoning generation budget. |
| `GIGA_OOM_RETRY_MAX_NEW_TOKENS` | `700` | Shorter retry budget after CUDA OOM. |
| `GIGA_IMAGE_MAX_NEW_TOKENS` | `512` | Image-analysis generation budget. |
| `GIGA_MAX_NEW_TOKENS` | `120` | Normal text-chat generation budget. |

### GIF Reactions

If the GIF tool is offered, the model can add a hidden command line like:

```text
GIF_SEARCH: laughing reaction
```

The bot strips that line, searches Klipy, and sends the GIF URL. Set
`KLIPY_API_KEY`, `KLIPY_CLIENT_KEY`, or `KLIPY_PLATFORM_KEY` to enable GIF search.

---

## Voice Chat

Chat can join a voice channel, listen, and talk back. Voice currently uses a
dedicated STT/TTS pipeline rather than Gemma audio:

```text
speech -> faster-whisper STT -> Gemma persona reply -> edge-tts TTS -> voice channel
```

Install the voice extras:

```bash
pip install -U --pre -r requirements-voice.txt
```

Use it in Discord:

1. Join a voice channel.
2. Type `!join` or mention the bot with a join request.
3. Say something with **"giga"** in it; that wake word acts like a voice mention.
4. Type `!leave` or mention the bot with a leave request to disconnect.

Voice tuning variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `WHISPER_MODEL` | `small` | faster-whisper model size. Bigger is usually more accurate and slower. |
| `WHISPER_COMPUTE` | `int8` | faster-whisper precision. Use int8 variants if VRAM is tight next to Gemma. |
| `GIGA_VOICE_RESPOND_ALL` | `0` | Set to `1` to reply to every utterance without the wake word. |
| `GIGA_VOICE_SILENCE_SECONDS` | `0.8` | Silence before an utterance is considered finished. |
| `GIGA_VOICE_MERGE_PAUSE` | `2.4` | Extra pause window for merging natural mid-thought breaks. |
| `GIGA_VOICE_MAX_LATENCY` | `6.0` | Drops stale utterances that waited too long. |
| `TTS_VOICE` | `en-US-JennyNeural` | edge-tts voice. |
| `TTS_RATE` | `-8%` | TTS speaking rate. |
| `TTS_PITCH` | `-8Hz` | TTS pitch. |
| `TTS_VOLUME` | `+0%` | TTS volume. |
| `GIGA_VOICE_POST_TRANSCRIPTS` | `0` | Set to `1` to post heard speech as text. |
| `GIGA_VOICE_POST_REPLIES` | `0` | Set to `1` to post spoken replies as text. |

> **Discord voice note:** normal Discord voice receive may be affected by DAVE
> end-to-end encryption depending on the library and channel type. Stage channels
> are the safest workaround when receive audio is garbled.

---

## Configuration

### Data

- `dataset.txt` is the raw Discord export and should stay local.
- `data/conversations.jsonl` is the parsed session dataset from `parser.py`.
- `dataset_interactions.txt` and `dataset_interactions_voice.txt` collect new bot
  interactions for later training.

### Training

Common `general_trainer.py` flags:

```text
--input --base-model --output-dir --context-turns --max-len --epochs --lr
--batch-size --grad-accum --lora-r --max-examples --max-steps --val-frac
```

The shared persona prompt and author formatting live in `giga_common.py` so the
trainer and Discord bot stay aligned.

### Discord Setup

1. Create a bot in the Discord Developer Portal.
2. Enable the **Message Content** intent.
3. Invite it with permissions to read/send messages and, for voice, connect/speak.
4. Set `DISCORD_TOKEN` before running `discordGiga.py`.

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- Transformers with Gemma 4 / multimodal support
- Accelerate, PEFT, Datasets, PyArrow
- discord.py
- Pillow for image loading
- A GPU is strongly recommended
- Optional voice: FFmpeg, faster-whisper, discord voice receive extras, edge-tts

> Fine-tuning on raw group chat will make the bot mirror that server's style,
> including rough language, bad habits, and weird in-jokes. Curate the dataset if
> you want a cleaner model.
