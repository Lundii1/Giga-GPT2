"""Voice-channel support for the Giga Discord bot.

Lets Chat sit in a Discord voice channel and hold a spoken conversation:

  1. **Listen** -- receives each speaker's audio separately (Discord tags every
     RTP stream with the member who sent it), buffers it per-user, and cuts an
     "utterance" once that user goes quiet for ``silence_seconds``.
  2. **Transcribe** -- runs the utterance through faster-whisper (local, GPU if
     available) so you get ``speaker -> text`` with proper attribution.
  3. **Reply** -- hands ``(text, speaker, guild_id)`` to a caller-supplied
     ``reply_fn`` (in practice the same persona model used for text chat).
  4. **Speak** -- synthesizes the reply with edge-tts and plays it back into the
     voice channel via FFmpeg.

This module is deliberately model-agnostic: it never imports the LLM. The bot
(``discordGiga.py``) passes in ``reply_fn`` so the voice persona is identical to
the text one. All heavy dependencies are imported lazily, so the text-only bot
keeps working even if the voice extras aren't installed.

Extra requirements (see ``requirements-voice.txt``)::

    pip install discord-ext-voice-recv faster-whisper edge-tts
    # plus FFmpeg on PATH (Discord audio playback) and libopus (usually bundled)
"""
import asyncio
import inspect
import logging
import os
import re
import tempfile
import time
import wave

import discord
import torch

# Discord delivers decoded PCM as 48 kHz, 16-bit, stereo.
_SAMPLE_RATE = 48000
_CHANNELS = 2
_SAMPLE_WIDTH = 2
_BYTES_PER_SECOND = _SAMPLE_RATE * _CHANNELS * _SAMPLE_WIDTH


class VoiceDepsMissing(RuntimeError):
    """Raised when an optional voice dependency isn't installed."""


def _install_voice_recv_resilience():
    """Stop one undecodable RTP packet from killing the whole receive thread.

    ``discord-ext-voice-recv`` wraps no error handling around the opus decode,
    and its ``PacketRouter`` catches exceptions *outside* its loop -- so a single
    packet it can't decode raises ``OpusError`` and permanently tears down voice
    receive (you see: ``OpusError: corrupted stream`` once, then silence).

    The usual culprit is a speaker on the Discord **web or mobile** client: those
    send RTP header extensions this library mis-parses, corrupting the opus
    payload. Desktop-app audio is unaffected. We wrap ``_decode_packet`` to drop
    the offending packet (returning empty PCM) instead of letting it crash, so
    one browser speaker can't take the bot's ears offline for everyone else.

    Returns True if the patch is in place (idempotent).
    """
    try:
        from discord.ext.voice_recv.opus import PacketDecoder
    except Exception:
        return False
    if getattr(PacketDecoder, "_giga_resilient", False):
        return True

    from discord.opus import OpusError
    original = PacketDecoder._decode_packet
    stats = {"drops": 0, "last_report": 0.0}

    def _decode_packet_safe(self, packet):
        try:
            return original(self, packet)
        except Exception as exc:  # OpusError, struct errors on bad extensions, etc.
            stats["drops"] += 1
            now = time.monotonic()
            # Report the first drop, then at most once every 5s, with a running
            # total -- so we can tell "one bad packet" from "audio never decodes".
            if stats["drops"] == 1 or now - stats["last_report"] >= 5.0:
                stats["last_report"] = now
                kind = "opus" if isinstance(exc, OpusError) else type(exc).__name__
                print(
                    f"[voice] dropped {stats['drops']} undecodable audio packet(s) "
                    f"so far (latest {kind}: {exc}). A few at the start of speech is "
                    "normal; a constant stream means that speaker's audio can't be "
                    "decoded (e.g. web/mobile client)."
                )
            return packet, b""

    PacketDecoder._decode_packet = _decode_packet_safe
    PacketDecoder._giga_resilient = True

    # Silence receive keepalive / gateway compatibility spam that otherwise
    # buries the useful audio diagnostics.
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
    return True


def _require(module, pip_name, extra=""):
    """Import ``module`` lazily, with a friendly install hint on failure."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        hint = f"pip install {pip_name}"
        if extra:
            hint += f"  ({extra})"
        raise VoiceDepsMissing(
            f"missing '{module}'. Install it with: {hint}"
        ) from exc


class _VoiceSession:
    """One active voice connection in a single guild."""

    def __init__(self, owner, voice_client, text_channel):
        self.owner = owner                       # the VoiceChat instance
        self.vc = voice_client                   # discord voice client (recv-capable)
        self.text_channel = text_channel         # where to mirror transcripts/replies
        self.loop = asyncio.get_running_loop()

        # Per-user rolling PCM buffer. Written from the voice-recv worker thread,
        # drained from the asyncio flush loop, so guard it with our own state and
        # only ever mutate buffers from those two well-defined places.
        self._buffers = {}                       # user_id -> {"buf", "last", "name"}
        self._pending = {}                       # user_id -> finalized audio waiting for a continuation
        self._queue = asyncio.Queue(maxsize=4)   # finalized utterances (kept small to stay current)
        self.active = True
        self._flush_task = None
        self._consume_task = None

    # -- audio in (called from the voice-recv thread) -------------------------
    def on_voice(self, user, data):
        """voice-recv callback: append this packet to the speaker's buffer."""
        if user is None or not self.active:
            return
        pcm = getattr(data, "pcm", None)
        if not pcm:
            return
        now = time.monotonic()
        entry = self._buffers.get(user.id)
        if entry is None:
            entry = {"buf": bytearray(), "last": 0.0, "name": _display_name(user)}
            self._buffers[user.id] = entry
        pending = self._pending.pop(user.id, None)
        if pending:
            entry["buf"].extend(pending["buf"])
        entry["buf"].extend(pcm)
        entry["last"] = now

    def _queue_utterance(self, speaker, pcm, queued_at):
        # Keep the freshest audio: if we're backed up, drop the OLDEST queued
        # utterance rather than this new one.
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait((speaker, pcm, queued_at))
        except asyncio.QueueFull:
            pass

    # -- background loops -----------------------------------------------------
    def start(self):
        self._flush_task = self.loop.create_task(self._flush_loop())
        self._consume_task = self.loop.create_task(self._consume_loop())

    async def _flush_loop(self):
        """Finalize an utterance when a speaker falls silent (or talks too long)."""
        cfg = self.owner
        while self.active:
            await asyncio.sleep(0.2)
            now = time.monotonic()
            for uid, pending in list(self._pending.items()):
                if now >= pending["deadline"]:
                    pending = self._pending.pop(uid, None)
                    if pending:
                        self._queue_utterance(
                            pending["name"],
                            bytes(pending["buf"]),
                            now,
                        )

            for uid, entry in list(self._buffers.items()):
                buf = entry["buf"]
                if not buf:
                    continue
                silent_for = now - entry["last"]
                too_long = len(buf) >= cfg.max_bytes
                if too_long or silent_for >= cfg.silence_seconds:
                    if len(buf) >= cfg.min_bytes:
                        if too_long or cfg.merge_pause_seconds <= 0:
                            self._queue_utterance(entry["name"], bytes(buf), now)
                        else:
                            self._pending[uid] = {
                                "buf": bytearray(buf),
                                "name": entry["name"],
                                "deadline": now + cfg.merge_pause_seconds,
                            }
                    entry["buf"] = bytearray()

    async def _consume_loop(self):
        """Process finalized utterances one at a time (STT -> reply -> TTS)."""
        while self.active:
            try:
                speaker, pcm, queued_at = await self._queue.get()
            except asyncio.CancelledError:  # pragma: no cover
                break
            # Drop stale audio: never reply to something said many seconds ago.
            # This is what stops the bot from falling 40s behind a busy channel.
            age = time.monotonic() - queued_at
            if age > self.owner.max_latency_seconds:
                print(f"[voice] skipping stale utterance from {speaker} ({age:.0f}s behind)")
                continue
            try:
                await self._handle_utterance(speaker, pcm)
            except Exception as exc:  # keep the session alive on any single failure
                print(f"[voice] error handling utterance from {speaker}: {exc}")

    async def _handle_utterance(self, speaker, pcm):
        seconds = len(pcm) / _BYTES_PER_SECOND
        # Don't freeze on a slow first model load (esp. 'small'): if STT isn't
        # ready yet, make sure the load is running and skip this one with a note,
        # rather than blocking the whole consume loop until it finishes.
        if not self.owner.whisper_ready():
            self.owner.loop_safe_warm()
            print(f"[voice] speech-to-text still loading; skipping {seconds:.1f}s from {speaker}")
            return
        if self.owner.debug_audio_dir:
            self.owner._save_debug_wav(pcm, speaker)
        text = await self.owner._transcribe(pcm)
        if not text:
            # Captured audio but Whisper found no speech -- usually means the
            # audio is choppy/garbled (lost packets), not actual silence.
            print(f"[voice] heard {seconds:.1f}s from {speaker} but transcribed nothing")
            return
        print(f"[voice:heard] ({seconds:.1f}s) {speaker}: {text}")
        if self.owner.post_transcripts and self.text_channel:
            await _safe_send(self.text_channel, f"🎤 **{speaker}:** {text}")

        guild_id = self.vc.guild.id if self.vc.guild else None
        try:
            await self.owner._call_transcript_fn(text, speaker, guild_id)
        except Exception as exc:
            print(f"[voice] transcript callback failed for {speaker}: {exc}")

        if not self.owner._is_addressed(text):
            print("[voice] (not addressed — say 'chat' to talk to me, "
                  "or run with GIGA_VOICE_RESPOND_ALL=1 to reply to everything)")
            return

        reply = await self.owner._call_reply_fn(text, speaker, guild_id)
        reply = (reply or "").strip()
        if not reply:
            return
        print(f"[voice:reply] {reply}")
        if self.owner.post_replies and self.text_channel:
            await _safe_send(self.text_channel, f"🤖 **Chat:** {reply}")

        await self._speak(reply)

    async def _speak(self, text):
        """Synthesize ``text`` and play it into the voice channel, blocking until done."""
        path = await self.owner._synthesize(text)
        if not path:
            return
        try:
            if self.vc.is_playing():
                self.vc.stop()
            done = asyncio.Event()

            def _after(err):  # called from the audio thread
                if err:
                    print(f"[voice] playback error: {err}")
                self.loop.call_soon_threadsafe(done.set)

            source = discord.FFmpegPCMAudio(path)
            self.vc.play(source, after=_after)
            await done.wait()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def close(self):
        self.active = False
        
        # CRITICAL: Stop listening first to prevent new audio from arriving
        try:
            if hasattr(self.vc, 'stop_listening'):
                self.vc.stop_listening()
        except Exception:
            pass
        
        # Cancel background tasks and wait for them to finish
        for task in (self._flush_task, self._consume_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        
        # Clear buffers to free memory
        self._buffers.clear()
        self._pending.clear()
        
        # Drain the queue to unblock any waiting consumers
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Finally disconnect
        try:
            if self.vc.is_connected():
                self.vc.stop()
                await self.vc.disconnect(force=True)
        except Exception:
            pass


class VoiceChat:
    """Manages Chat's voice sessions across guilds.

    Parameters
    ----------
    reply_fn:
        ``reply_fn(text, speaker_name, guild_id) -> str``. May be sync or async.
        This is where the persona model produces Chat's spoken reply.
    transcript_fn:
        Optional ``transcript_fn(text, speaker_name, guild_id)`` callback called
        for every successful transcription, including speech that does not
        address the bot.
    whisper_model_size:
        faster-whisper model name (``tiny``/``base``/``small``/``medium``/...).
    tts_voice:
        edge-tts voice id (e.g. ``en-US-GuyNeural``).
    tts_rate, tts_pitch, tts_volume:
        Edge TTS prosody controls. ``edge-tts`` no longer accepts arbitrary
        custom SSML such as ``mstts:express-as``, but it does generate the
        supported ``<voice><prosody>...`` XML internally from these values.
        Examples: ``tts_rate="-8%"``, ``tts_pitch="-10Hz"``,
        ``tts_volume="+0%"``.
    device:
        ``"cuda"`` or ``"cpu"`` for the speech-to-text model.
    language:
        Forced transcription language (``"en"``), or ``None`` to auto-detect.
    initial_prompt:
        Text shown to Whisper as preceding context to bias its vocabulary. Use
        it to spell in-group proper nouns it would otherwise mangle -- above all
        the wake word "Giga". ``None`` disables biasing.
    wake_words:
        Only reply when one of these appears in the transcription, unless
        ``respond_to_all`` is True. Voice analogue of @-mentioning the bot.
    merge_pause_seconds:
        Extra time to wait after a silence cut before sending audio to Whisper.
        If the same speaker resumes during this window, the audio is merged into
        one utterance. Higher values handle longer thinking pauses but add reply
        latency.
    """

    def __init__(
        self,
        reply_fn,
        *,
        transcript_fn=None,
        whisper_model_size="medium",
        tts_voice="en-US-ChristopherNeural",
        tts_rate="+0%",
        tts_pitch="+0Hz",
        tts_volume="+0%",
        device="cuda" if torch.cuda.is_available() else "cpu",
        language="en",
        initial_prompt=(
            "Casual voice chat in the Giga Discord group with Chat, also "
            "called chat-ai."
        ),
        wake_words=(
            "chat",
            "chat-ai",
            "chat ai",
            "chat_ai",
            "chatai",
            "chat a i",
            "chat a.i.",
        ),
        respond_to_all=False,
        silence_seconds=0.8,
        min_seconds=0.4,
        max_seconds=20.0,
        max_latency_seconds=10.0,
        merge_pause_seconds=1.2,
        whisper_compute="int8",
        post_transcripts=True,
        post_replies=True,
        debug_audio_dir=None,
    ):
        self.reply_fn = reply_fn
        self.transcript_fn = transcript_fn
        self.whisper_model_size = whisper_model_size
        self.tts_voice = tts_voice
        self.tts_rate = tts_rate
        self.tts_pitch = tts_pitch
        self.tts_volume = tts_volume
        self.device = device
        self.language = language
        self.initial_prompt = initial_prompt or None
        self.wake_words = tuple(w.lower() for w in wake_words)
        self.respond_to_all = respond_to_all
        # When set, every captured utterance is saved here as a WAV so you can
        # listen to exactly what the bot received (best capture/decode diagnostic).
        self.debug_audio_dir = debug_audio_dir
        if debug_audio_dir:
            os.makedirs(debug_audio_dir, exist_ok=True)

        self.silence_seconds = silence_seconds
        self.min_bytes = int(min_seconds * _BYTES_PER_SECOND)
        self.max_bytes = int(max_seconds * _BYTES_PER_SECOND)
        self.merge_pause_seconds = merge_pause_seconds
        # Skip any utterance that waited longer than this to be processed, so the
        # bot always answers recent speech instead of a growing backlog.
        self.max_latency_seconds = max_latency_seconds
        self.whisper_compute = whisper_compute  # override faster-whisper compute_type
        self.post_transcripts = post_transcripts
        self.post_replies = post_replies

        self._sessions = {}        # guild_id -> _VoiceSession
        self._whisper = None       # lazily loaded faster-whisper model
        self._whisper_lock = asyncio.Lock()
        self._whisper_warm_task = None

    # -- public API -----------------------------------------------------------
    def is_connected(self, guild):
        session = self._sessions.get(getattr(guild, "id", guild))
        return bool(session and session.vc.is_connected())

    def whisper_ready(self):
        """Return True once the STT model has finished loading."""
        return self._whisper is not None

    async def join(self, voice_channel, text_channel=None):
        """Join (or move to) ``voice_channel`` and start listening."""
        try:
            from discord.ext import voice_recv
        except ImportError as exc:
            raise VoiceDepsMissing(
                "missing 'discord-ext-voice-recv'. Install it with: "
                "pip install -U --pre discord-ext-voice-recv  (needs discord.py >= 2.5)"
            ) from exc

        # Survive packets from web/mobile speakers instead of crashing on them.
        _install_voice_recv_resilience()

        guild = voice_channel.guild
        await self.leave(guild)  # ensure a clean slate if already connected

        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        # Diagnostic: which encryption mode Discord negotiated. If receive yields
        # only 'corrupted stream', confirm voice-recv actually supports this mode.
        print(f"[voice] connected to '{voice_channel.name}'; "
              f"encryption mode = {getattr(voice_client, 'mode', '?')}")
        session = _VoiceSession(self, voice_client, text_channel)
        self._sessions[guild.id] = session

        voice_client.listen(voice_recv.BasicSink(session.on_voice))
        session.start()

        # Warm up the STT model now so the first reply isn't slow.
        self.loop_safe_warm()
        return session

    def loop_safe_warm(self):
        if self.whisper_ready():
            return None
        if self._whisper_warm_task and not self._whisper_warm_task.done():
            return self._whisper_warm_task
        try:
            self._whisper_warm_task = asyncio.create_task(self._ensure_whisper())
            return self._whisper_warm_task
        except RuntimeError:  # no running loop (shouldn't happen from join)
            return None

    async def leave(self, guild):
        session = self._sessions.pop(getattr(guild, "id", guild), None)
        if session:
            await session.close()

    async def close_all(self):
        for guild_id in list(self._sessions):
            await self.leave(guild_id)

    # -- speech to text -------------------------------------------------------
    async def _ensure_whisper(self):
        if self._whisper is not None:
            return self._whisper
        async with self._whisper_lock:
            if self._whisper is not None:
                return self._whisper
            _require("faster_whisper", "faster-whisper")
            from faster_whisper import WhisperModel
            compute_type = self.whisper_compute or (
                "float16" if self.device == "cuda" else "int8"
            )
            print(
                f"[voice] loading faster-whisper '{self.whisper_model_size}' "
                f"on {self.device} ({compute_type})..."
            )
            try:
                self._whisper = await asyncio.to_thread(
                    WhisperModel, self.whisper_model_size,
                    device=self.device, compute_type=compute_type,
                )
            except Exception as exc:
                # Bigger models (e.g. 'small') can run the GPU out of memory when
                # sharing it with the LLM. int8 uses far less VRAM -- retry with it
                # before giving up (this is the usual cause of "small breaks").
                if self.device == "cuda" and compute_type != "int8":
                    print(f"[voice] whisper load failed ({exc}).")
                    print("[voice] retrying with compute_type=int8 (lower VRAM)...")
                    self._whisper = await asyncio.to_thread(
                        WhisperModel, self.whisper_model_size,
                        device=self.device, compute_type="int8",
                    )
                else:
                    raise
            print("[voice] speech-to-text ready.")
        return self._whisper

    async def _transcribe(self, pcm):
        """Transcribe raw 48 kHz stereo PCM bytes -> text (faster-whisper)."""
        model = await self._ensure_whisper()
        return await asyncio.to_thread(self._transcribe_sync, model, pcm)

    def _transcribe_sync(self, model, pcm):
        path = _write_temp_wav(pcm)
        try:
            segments, _info = model.transcribe(
                path,
                language=self.language,
                vad_filter=True,
                beam_size=5,                       # was 1 (greedy); 5 is far more accurate
                initial_prompt=self.initial_prompt,  # bias toward "Giga" + member names
                condition_on_previous_text=False,  # stop hallucination drift on short clips
                # Drop silence-only / low-confidence / repetitive segments instead
                # of emitting confident garbage on noisy Discord audio. These are
                # faster-whisper's defaults, surfaced here so they're easy to tune.
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            return " ".join(seg.text for seg in segments).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _save_debug_wav(self, pcm, speaker):
        """Dump a captured utterance to debug_audio_dir so it can be listened to."""
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", speaker)[:40] or "speaker"
        name = f"{time.strftime('%H%M%S')}_{safe}.wav"
        path = os.path.join(self.debug_audio_dir, name)
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(_CHANNELS)
                wav.setsampwidth(_SAMPLE_WIDTH)
                wav.setframerate(_SAMPLE_RATE)
                wav.writeframes(pcm)
            print(f"[voice] saved capture -> {path}")
        except OSError as exc:
            print(f"[voice] could not save debug wav: {exc}")

    def _is_addressed(self, text):
        if self.respond_to_all:
            return True
        collapsed = re.sub(r"[^a-z]", "", text.lower())
        if any(re.sub(r"[^a-z]", "", w) in collapsed for w in self.wake_words):
            return True
        if re.search(r"\bchat\s*a\s*i\b", text.lower()):
            return True
        if re.search(r"chat(?:ai|a\s*i|a\.i\.|ay)", collapsed):
            return True
        return bool(re.search(r"\bch[ae]+t\b", text.lower()))

    # -- reply ----------------------------------------------------------------
    async def _call_transcript_fn(self, text, speaker, guild_id):
        if self.transcript_fn is None:
            return None
        result = self.transcript_fn(text, speaker, guild_id)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _call_reply_fn(self, text, speaker, guild_id):
        if inspect.iscoroutinefunction(self.reply_fn):
            return await self.reply_fn(text, speaker, guild_id)
        # Persona generation is GPU-heavy and blocking. Actually run it OFF the
        # event loop -- the old code called it inline first, freezing audio and
        # the gateway for the whole generation.
        return await asyncio.to_thread(self.reply_fn, text, speaker, guild_id)

    # -- text to speech -------------------------------------------------------
    async def _synthesize(self, text):
        """Render ``text`` to an audio file path. edge-tts first, pyttsx3 fallback."""
        try:
            return await self._synthesize_edge(text)
        except VoiceDepsMissing:
            return await asyncio.to_thread(self._synthesize_pyttsx3, text)

    async def _synthesize_edge(self, text):
        _require("edge_tts", "edge-tts")
        import edge_tts
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="giga_tts_")
        os.close(fd)
        await edge_tts.Communicate(
            text,
            self.tts_voice,
            rate=self.tts_rate,
            pitch=self.tts_pitch,
            volume=self.tts_volume,
        ).save(path)
        return path

    def _synthesize_pyttsx3(self, text):
        _require("pyttsx3", "pyttsx3")
        import pyttsx3
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="giga_tts_")
        os.close(fd)
        engine = pyttsx3.init()
        engine.save_to_file(text, path)
        engine.runAndWait()
        return path


# -- module helpers -----------------------------------------------------------
def _display_name(member):
    return (
        getattr(member, "display_name", None)
        or getattr(member, "global_name", None)
        or getattr(member, "name", None)
        or str(member)
    )


def _write_temp_wav(pcm):
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="giga_stt_")
    os.close(fd)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(_CHANNELS)
        wav.setsampwidth(_SAMPLE_WIDTH)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(pcm)
    return path


async def _safe_send(channel, content):
    try:
        await channel.send(content)
    except discord.DiscordException as exc:
        print(f"[voice] failed to post to text channel: {exc}")
