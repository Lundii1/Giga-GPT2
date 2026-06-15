"""Parse a raw Giga Discord export into trainer-friendly conversation sessions.

Input  : dataset.txt   (the raw export)
Output : data/conversations.jsonl   (one JSON session per line)

Each output line looks like::

    {"session_id": 12, "turns": [
        {"author": "trolltusk", "content": "is zatken your mc username"},
        {"author": "bitpumpkin", "content": "yea it is"}]}

Speaker names are kept here because the trainer needs to know who said what to
build conversational context. They are *not* meant to end up in the model's
output -- ``general_trainer.py`` masks them out of the training loss.

Raw export format (CRLF line endings)::

    ==============================================================
    Guild: Giga
    Channel: Text Channels / general
    ==============================================================

    [2020-03-15 19:24] bitpumpkin
    can we vote to kick aidan

Run::

    python parser.py --input dataset.txt --output data/conversations.jsonl
"""
import argparse
import json
import os
import re
from datetime import datetime, timedelta

# Header line of a message: "[YYYY-MM-DD HH:MM] author name"
HEADER_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] (.+)$")
TS_FORMAT = "%Y-%m-%d %H:%M"

# Banner / metadata lines that are never message content.
BANNER_RE = re.compile(r"^(={5,}|Guild:|Channel:)")

# A "{Attachments}" / "{Embed}" / "{Reactions}" / "{Stickers}" /
# "{Forwarded Message}" marker. Everything from the marker up to the next blank
# line is non-text noise (URLs, reaction emoji, sticker names, embed dumps).
MARKER_RE = re.compile(r"^\{[A-Za-z ]+\}$")

# System messages: the export renders these as ordinary one-line "messages".
SYSTEM_MESSAGES = {"Joined the server.", "Pinned a message."}

URL_RE = re.compile(r"https?://\S+")
# Discord markup: user/role/channel mentions, and custom (animated) emoji.
MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
CUSTOM_EMOJI_RE = re.compile(r"<a?(:[A-Za-z0-9_]+:)\d+>")
MULTI_BLANK_RE = re.compile(r"\n\s*\n+")


def clean_content(lines):
    """Turn the raw lines between two headers into a clean message string.

    Drops media/reaction/embed blocks, strips URLs and Discord markup, and
    collapses whitespace. Returns "" if nothing meaningful is left.
    """
    kept = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if MARKER_RE.match(stripped):
            in_block = True  # skip the marker and its block body
            continue
        if in_block:
            if stripped == "":
                in_block = False  # blank line ends the block
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = CUSTOM_EMOJI_RE.sub(r"\1", text)  # <:kek:123> -> :kek:
    # Normalise whitespace: trim each line, collapse blank runs, trim ends.
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = MULTI_BLANK_RE.sub("\n", text)
    return text.strip()


def parse_messages(path, drop_authors, min_chars):
    """Yield (datetime, author, content) for every real message in the export."""
    with open(path, encoding="utf-8-sig") as fh:
        raw = fh.read()
    lines = raw.splitlines()

    cur_ts = None
    cur_author = None
    buf = []

    def flush():
        if cur_author is None:
            return None
        content = clean_content(buf)
        if not content or len(content) < min_chars:
            return None
        if content in SYSTEM_MESSAGES:
            return None
        if cur_author in drop_authors:
            return None
        try:
            dt = datetime.strptime(cur_ts, TS_FORMAT)
        except (ValueError, TypeError):
            return None
        return (dt, cur_author, content)

    stats = {"raw_messages": 0}
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            stats["raw_messages"] += 1
            out = flush()
            if out is not None:
                yield out
            cur_ts, cur_author = m.group(1), m.group(2).strip()
            buf = []
        elif cur_author is not None:
            if BANNER_RE.match(line):
                continue  # a banner block landed mid-stream; ignore it
            buf.append(line)
    out = flush()
    if out is not None:
        yield out
    parse_messages.raw_count = stats["raw_messages"]


def build_sessions(messages, gap_minutes, merge_minutes, max_turns):
    """Group merged turns into sessions split on conversational time gaps."""
    gap = timedelta(minutes=gap_minutes)
    merge = timedelta(minutes=merge_minutes)

    sessions = []
    cur = []
    prev_dt = None
    for dt, author, content in messages:
        new_session = prev_dt is None or (dt - prev_dt) > gap or len(cur) >= max_turns
        if new_session and cur:
            if len(cur) >= 2:
                sessions.append(cur)
            cur = []
        # Merge bursts: same author within the merge window -> one turn.
        if cur and cur[-1]["author"] == author and prev_dt is not None \
                and (dt - prev_dt) <= merge:
            cur[-1]["content"] += "\n" + content
        else:
            cur.append({"author": author, "content": content})
        prev_dt = dt
    if len(cur) >= 2:
        sessions.append(cur)
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="dataset.txt")
    ap.add_argument("--output", default="data/conversations.jsonl")
    ap.add_argument("--gap-minutes", type=int, default=15,
                    help="gap above which a new conversation session starts")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="hard cap on turns per session")
    ap.add_argument("--merge-minutes", type=int, default=2,
                    help="merge consecutive same-author messages within this window")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="drop messages shorter than this after cleaning")
    ap.add_argument("--drop-authors", nargs="*", default=[],
                    help="usernames to exclude entirely, e.g. --drop-authors 'Deleted User'")
    args = ap.parse_args()

    drop_authors = set(args.drop_authors)
    messages = list(parse_messages(args.input, drop_authors, args.min_chars))
    sessions = build_sessions(messages, args.gap_minutes, args.merge_minutes,
                              args.max_turns)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n_turns = 0
    authors = set()
    with open(args.output, "w", encoding="utf-8") as fh:
        for i, turns in enumerate(sessions):
            n_turns += len(turns)
            authors.update(t["author"] for t in turns)
            fh.write(json.dumps({"session_id": i, "turns": turns},
                                ensure_ascii=False) + "\n")

    raw = getattr(parse_messages, "raw_count", 0)
    print(f"raw message headers : {raw}")
    print(f"kept messages       : {len(messages)}")
    print(f"sessions written    : {len(sessions)}")
    print(f"turns (after merge) : {n_turns}")
    print(f"unique authors      : {len(authors)}")
    if sessions:
        print(f"avg turns/session   : {n_turns / len(sessions):.1f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
