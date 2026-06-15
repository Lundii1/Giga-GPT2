"""Shared constants and helpers for the Giga pipeline.

Both the trainer (``general_trainer.py``) and the Discord bot (``discordGiga.py``)
import from here so the system prompt and transcript formatting can never drift
apart. The whole point of the project is that speaker names are used only as
*context* (inside the prompt the model conditions on) and never appear in the
text the model generates, so the formatting must be identical on both sides.
"""
import re

# The fixed persona anchor. This is what lets the model "know it is imitating
# specifically" the Giga group, while the per-message speaker labels live only in
# the (loss-masked) context so they never leak into generated replies.
SYSTEM_PROMPT = (
    "You are Giga-AI, the meanest, a member of the 'Giga' Discord group. "
 
)


def render_transcript(turns):
    """Render context turns as ``author: content`` lines.

    ``turns`` is a list of dicts with ``author`` and ``content`` keys. This is the
    only place speaker names enter the prompt; the trainer masks every token of
    this transcript out of the loss.
    """
    return "\n".join(f"{t['author']}: {t['content']}" for t in turns)


# Matches an accidental leading speaker label the model might still emit.
# This is the single source of truth for cleaning bot replies.
# It catches normal usernames + every variant of "Giga-AI" (with or without separator,
# with or without discriminator, with or without brackets).
_NAME_PREFIX_RE = re.compile(
    r"^\s*[<\[]?("
    r"[a-z0-9._-]{2,32}"          # normal username (lundiilundii, trolltusk, etc.)
    r"|giga[-_\s.]*ai"            # Giga-AI, Giga AI, Giga_AI, GigaAI, Giga A.I., etc.
    r"|gigaai"                    # GigaAI (compact)
    r")([#0-9]{0,6})?[:>\]]?\s*:\s+",
    re.IGNORECASE
)


def strip_name_prefix(text):
    """Safety net: drop a leading ``name:`` / ``<name>`` / ``[name]`` prefix.

    Training masks names out of the target, so this should rarely fire, but it
    guarantees the bot never posts a reply that starts with a friend's name.
    """
    return _NAME_PREFIX_RE.sub("", text, count=1).lstrip()
