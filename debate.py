#!/usr/bin/env python3
"""
Claude vs GPT-4 debate: they argue for N rounds, then each contributes
a synthesis, then a final unified answer is assembled from both.

Usage:
    python debate.py "Is Python better than JavaScript?"
    python debate.py  (uses a default topic)

Requires env vars:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
"""
import os
import sys
import textwrap

import anthropic
import openai

claude = anthropic.Anthropic()
gpt = openai.OpenAI()

CLAUDE_MODEL = "claude-opus-4-7"
GPT_MODEL = "gpt-4o"

# ── Prompt templates ──────────────────────────────────────────────────────────

DEBATE_SYSTEM = (
    "You are {name}, made by {maker}. "
    'You\'re in a head-to-head debate with {other} on: "{topic}". '
    "Argue your perspective confidently. Push back on your opponent's weak points directly. "
    "Be sharp and a little competitive. 2–3 punchy paragraphs, no bullet points."
)

SYNTHESIS_SYSTEM = (
    "You are {name}, made by {maker}. "
    "You just finished debating {other}. "
    "The debate is over — now you're collaborating, not competing."
)

SYNTHESIS_USER = (
    "Here is the full debate transcript:\n\n{transcript}\n\n"
    "---\n"
    "The debate is done. Your job now is to contribute your best thinking toward "
    'a final answer to: "{topic}"\n\n'
    "Be honest: acknowledge the strongest points made by your opponent, "
    "concede anything you argued that was weaker, and distill what you genuinely "
    "believe is correct and important. "
    "2–3 paragraphs. This will be combined with your opponent's synthesis."
)

FINAL_SYSTEM = (
    "You are an expert synthesizer. Claude (Anthropic) and GPT-4 (OpenAI) debated "
    "a topic and each contributed their best synthesis. Your job is to produce one "
    "definitive, well-reasoned answer by combining the strongest insights from both. "
    "Be thorough but concise. No preamble about 'two AIs debated' — just give "
    "the answer directly. Use paragraphs, not bullet points."
)

FINAL_USER = (
    'Question / topic: "{topic}"\n\n'
    "Synthesis from Claude:\n{claude_synth}\n\n"
    "Synthesis from GPT-4:\n{gpt_synth}\n\n"
    "---\n"
    "Produce the single best answer to the question above, drawing on both "
    "syntheses. Resolve any remaining disagreements with your own judgment. "
    "This is the final answer the user will read."
)

# ── Speaker config ─────────────────────────────────────────────────────────────

SPEAKERS = [
    {
        "display": "Claude 🟠",
        "key": "Claude",
        "name": "Claude",
        "maker": "Anthropic",
        "other": "GPT-4 (OpenAI)",
    },
    {
        "display": "GPT-4 🟢",
        "key": "GPT-4",
        "name": "GPT-4",
        "maker": "OpenAI",
        "other": "Claude (Anthropic)",
    },
]

# ── API call helpers ──────────────────────────────────────────────────────────


def _call_claude(system: str, user: str, max_tokens: int = 500) -> str:
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _call_gpt(system: str, user: str, max_tokens: int = 500) -> str:
    resp = gpt.chat.completions.create(
        model=GPT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


CALLERS = {"Claude": _call_claude, "GPT-4": _call_gpt}

# ── Debate turn ───────────────────────────────────────────────────────────────


def debate_turn(history: list, topic: str, speaker: dict) -> str:
    system = DEBATE_SYSTEM.format(
        name=speaker["name"],
        maker=speaker["maker"],
        other=speaker["other"],
        topic=topic,
    )
    if history:
        lines = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
        user = f"Transcript so far:\n\n{lines}\n\n---\nYour turn. Respond directly to what your opponent said."
    else:
        user = f'Open the debate on: "{topic}"'
    return CALLERS[speaker["key"]](system, user, max_tokens=450)


# ── Synthesis turn ────────────────────────────────────────────────────────────


def synthesis_turn(history: list, topic: str, speaker: dict) -> str:
    system = SYNTHESIS_SYSTEM.format(
        name=speaker["name"],
        maker=speaker["maker"],
        other=speaker["other"],
    )
    transcript = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
    user = SYNTHESIS_USER.format(transcript=transcript, topic=topic)
    return CALLERS[speaker["key"]](system, user, max_tokens=550)


# ── Final answer ──────────────────────────────────────────────────────────────


def final_answer(syntheses: dict, topic: str) -> str:
    user = FINAL_USER.format(
        topic=topic,
        claude_synth=syntheses["Claude"],
        gpt_synth=syntheses["GPT-4"],
    )
    return _call_claude(FINAL_SYSTEM, user, max_tokens=800)


# ── Formatting ────────────────────────────────────────────────────────────────

W = 70


def wrap(text: str) -> str:
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(textwrap.fill(p, W - 2) for p in paragraphs)


def header(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}\n")


def subheader(title: str) -> None:
    print(f"── {title} {'─' * max(0, W - len(title) - 4)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def run_debate(topic: str, rounds: int = 2) -> None:
    header(f"TOPIC: {topic}")

    history = []

    for r in range(1, rounds + 1):
        subheader(f"Round {r}")
        for speaker in SPEAKERS:
            print(f"  {speaker['display']}")
            print(f"  {'─' * 40}")
            text = debate_turn(history, topic, speaker)
            print(wrap(text))
            print()
            history.append({"key": speaker["key"], "text": text})

    header("SYNTHESIS — Finding common ground")
    syntheses = {}
    for speaker in SPEAKERS:
        print(f"  {speaker['display']} synthesizes:")
        print(f"  {'─' * 40}")
        text = synthesis_turn(history, topic, speaker)
        print(wrap(text))
        print()
        syntheses[speaker["key"]] = text

    header("✦ FINAL ANSWER")
    answer = final_answer(syntheses, topic)
    print(wrap(answer))
    print(f"\n{'═' * W}\n")


if __name__ == "__main__":
    topic = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Which AI assistant is the most capable and trustworthy?"
    )
    try:
        run_debate(topic, rounds=2)
    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
