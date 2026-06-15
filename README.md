# Giga-GPT2

A **PyTorch- and TinyLlama-powered** system for training an AI that imitates the
**collective chatting personality of a Discord friend group** and deploying it as a
live **Discord bot**.

---

## Overview

**Giga-GPT2** fine-tunes [**TinyLlama-1.1B-Chat**](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
on a raw Discord server export so the model learns to talk like the group — its
slang, in-jokes, and conversational rhythm — as one blended "Giga" voice.

The model is given the recent chat as context (so it knows *who said what*) but is
trained to reply with only the next message. **Speaker names are used purely as
context and never appear in what the model generates** — it imitates the group
without ever signing a friend's name to a message. This is achieved with
chat-format supervised fine-tuning and **loss masking** (only the reply tokens are
trained on; the name-labelled context is masked out).

---

## Pipeline

```
dataset.txt ──► parser.py ──► data/conversations.jsonl ──► general_trainer.py ──► output/giga ──► discordGiga.py
   raw export      clean + sessionize       training data         fine-tune (LoRA)      model        Discord bot
```

| File | Role |
|------|------|
| `parser.py` | Cleans the raw export (strips attachments/embeds/reactions, URLs, system messages, mentions), merges message bursts, and splits the log into conversation sessions. |
| `general_trainer.py` | Builds chat-format examples with loss masking and fine-tunes TinyLlama (LoRA by default). |
| `discordGiga.py` | Loads the fine-tuned model and runs the group-voice Discord bot. |
| `giga_common.py` | Shared system prompt + formatting helpers (keeps trainer and bot in sync). |
| `voice_giga.py` | Optional voice-channel support: joins a VC, transcribes each speaker (STT), and replies out loud (TTS). |

---

## Installation

```bash
git clone https://github.com/Lundii1/Giga-GPT2.git
cd Giga-GPT2
pip install torch transformers accelerate datasets peft discord.py==2.2.2 pyarrow
```

---

## Usage

**1. Parse the raw Discord export** into training data:
```bash
python parser.py --input dataset.txt --output data/conversations.jsonl
```

**2. Fine-tune the model** (LoRA, ~1 epoch). Use `--max-steps 5` first for a quick smoke test:
```bash
python general_trainer.py
```
Add `--no-lora` for a full fine-tune if you have a 16 GB+ GPU.

**3. Run the Discord bot** (set your token first):
```bash
# PowerShell:  $env:DISCORD_TOKEN = "your-bot-token"
python discordGiga.py
```

---

## Voice chat (optional)

Chat can also join a **voice channel**, listen, and talk back. It receives each
speaker's audio separately, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(local, GPU-capable), feeds the text through the *same* persona model used for text
chat, and speaks the reply with [edge-tts](https://github.com/rany2/edge-tts).

```
🎤 speech ──► faster-whisper (STT) ──► generate_reply (persona) ──► edge-tts (TTS) ──► 🔊 voice channel
       per-speaker attribution                                          FFmpeg playback
```

**Install the voice extras** (the text bot works without them). Note the
`--pre` flag — `discord-ext-voice-recv` only ships pre-releases, and an older
build can't decrypt Discord's current voice encryption (`OpusError: corrupted
stream`):
```bash
pip install -U --pre -r requirements-voice.txt
```
You also need **FFmpeg on your PATH**, and **discord.py ≥ 2.7**.

> ### ⚠️ Discord DAVE E2EE breaks voice receive on normal channels
> On **2026-03-02** Discord began enforcing **DAVE end-to-end encryption** on all
> **non-stage** voice calls. This affects *receiving* audio across the whole
> ecosystem (discord.js too), not just this bot:
> - discord.py **< 2.7** can't even connect — the voice socket closes with **code 4017**.
> - discord.py **≥ 2.7** connects, but `voice-recv` doesn't yet implement DAVE
>   decryption, so audio from a normal voice channel arrives as **garbled noise**
>   ([voice-recv #53](https://github.com/imayhaveborkedit/discord-ext-voice-recv/issues/53)).
> - **Stage channels are exempt** from E2EE enforcement, so `voice-recv` can still
>   decode them — running the bot in a **Stage channel** is the current workaround.
>
> Install [`davey`](https://github.com/Snazzah/davey) (`pip install "davey>=0.1.5"`) so
> discord.py 2.7+ can participate in DAVE calls and **speak** (TTS) under E2EE — but note
> this does **not** fix *receiving*: `voice-recv` decrypts packets itself and has no DAVE
> support ([PR not yet merged](https://github.com/imayhaveborkedit/discord-ext-voice-recv)).
> Until it does, use discord.py ≥ 2.7 **+** `davey`, and a **Stage channel** for receive.

**Use it in Discord:**
1. Join a voice channel yourself.
2. Type `!join` (or `@Chat join the vc`). The bot pulls up and starts listening.
3. Say something with **"giga"** in it — that's the voice version of @-mentioning it.
   It transcribes you, replies in character, and speaks the reply out loud.
4. Type `!leave` (or `@Chat leave vc`) to make it disconnect.

**Tuning (environment variables):**

| Variable | Default | Effect |
|----------|---------|--------|
| `WHISPER_MODEL` | `small` | faster-whisper size (`tiny`/`base`/`small`/`medium`/`large-v3`). Bigger = more accurate, slower. |
| `TTS_VOICE` | `en-US-JennyNeural` | edge-tts voice. Jenny is the default warm/caring adult female voice. |
| `TTS_RATE` | `-8%` | edge-tts prosody rate. Negative values speak slower. |
| `TTS_PITCH` | `-10Hz` | edge-tts prosody pitch. Negative values lower the voice. |
| `TTS_VOLUME` | `+0%` | edge-tts prosody volume. |
| `GIGA_VOICE_RESPOND_ALL` | `0` | Set to `1` to reply to **everything** it hears (no "giga" wake word needed). |

> **Note:** STT and TTS run alongside the LLM. A `base` Whisper model + the 4-bit
> 7B fine-tune fit comfortably on a single mid-range GPU. If `edge-tts` isn't
> available it falls back to offline `pyttsx3`.

---

## Configuration

### Data
- `dataset.txt` is the raw Discord export (timestamped `[YYYY-MM-DD HH:MM] username` messages).
  It stays local — it is git-ignored, since it contains private messages.
- `parser.py` flags: `--gap-minutes`, `--max-turns`, `--merge-minutes`, `--min-chars`,
  `--drop-authors "Deleted User"`.

### Training
- `general_trainer.py` flags: `--context-turns`, `--max-len`, `--epochs`, `--lr`,
  `--max-examples`, `--max-steps`, `--no-lora`. The persona system prompt lives in
  `giga_common.py`.

### Discord Setup
1. Create a bot via the [Discord Developer Portal](https://discord.com/developers/applications)
   and enable the **Message Content** intent.
2. Provide its token through the `DISCORD_TOKEN` environment variable.
3. Invite it to your server with the OAuth2 URL. Mention the bot to get a reply.

---

## Requirements

- **Python 3.10+**
- **PyTorch 2.x**, **Transformers**, **Accelerate**, **PEFT**, **Datasets** (Hugging Face)
- **discord.py 2.2.2**, **PyArrow**
- A raw Discord chat export as `dataset.txt`
- A **GPU** is strongly recommended (LoRA fits in ~3–4 GB; full fine-tune needs ~16 GB+)
- *(Optional, voice chat)* **discord.py ≥ 2.4**, **discord-ext-voice-recv**, **faster-whisper**,
  **edge-tts**, plus **FFmpeg** on PATH — see [Voice chat](#voice-chat-optional)

> **Note:** Fine-tuning on raw, unfiltered group chat will erode the base model's
> light safety tuning — the resulting bot mirrors the group's unfiltered style.
