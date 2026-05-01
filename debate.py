#!/usr/bin/env python3
"""
Three-way AI debate: Claude vs GPT-4 vs Gemini
→ They argue for N rounds, then each contributes a synthesis,
  then a final unified answer is assembled from all three.

Usage:
    python debate.py "Is Python better than JavaScript?"
    python debate.py  (uses a default topic)

Requires env vars:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GEMINI_API_KEY
"""
import os
import sys
import textwrap

import anthropic
import openai
from google import genai as google_genai
from google.genai import types as google_types

claude = anthropic.Anthropic()
gpt = openai.OpenAI()
gemini = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CLAUDE_MODEL = "claude-opus-4-7"
GPT_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash"

# ── Prompt templates ──────────────────────────────────────────────────────────

DEBATE_SYSTEM = (
    "You are {name}, made by {maker}. "
    'You\'re in a three-way debate with {other1} and {other2} on: "{topic}". '
    "Argue your perspective confidently. Push back on the others' weak points directly. "
    "Be sharp and a little competitive. 2–3 punchy paragraphs, no bullet points."
)

SYNTHESIS_SYSTEM = (
    "You are {name}, made by {maker}. "
    "You just finished debating {other1} and {other2}. "
    "The debate is over — now you're collaborating, not competing."
)

SYNTHESIS_USER = (
    "Here is the full debate transcript:\n\n{transcript}\n\n"
    "---\n"
    "The debate is done. Your job now is to contribute your best thinking toward "
    'a final answer to: "{topic}"\n\n'
    "Be honest: acknowledge the strongest points made by your opponents, "
    "concede anything you argued that was weaker, and distill what you genuinely "
    "believe is correct and important. "
    "2–3 paragraphs. This will be combined with the other AIs' syntheses."
)

FINAL_SYSTEM = (
    "You are an expert synthesizer. Three AI systems — Claude (Anthropic), "
    "GPT-4 (OpenAI), and Gemini (Google) — debated a topic and then each "
    "contributed their best synthesis. Your job is to produce one definitive, "
    "well-reasoned answer by combining the strongest insights from all three. "
    "Be thorough but concise. No preamble about 'three AIs debated' — just give "
    "the answer directly. Use paragraphs, not bullet points."
)

FINAL_USER = (
    'Question / topic: "{topic}"\n\n'
    "Synthesis from Claude:\n{claude_synth}\n\n"
    "Synthesis from GPT-4:\n{gpt_synth}\n\n"
    "Synthesis from Gemini:\n{gemini_synth}\n\n"
    "---\n"
    "Produce the single best answer to the question above, drawing on all three "
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
        "others": ("GPT-4 (OpenAI)", "Gemini (Google)"),
    },
    {
        "display": "GPT-4 🟢",
        "key": "GPT-4",
        "name": "GPT-4",
        "maker": "OpenAI",
        "others": ("Claude (Anthropic)", "Gemini (Google)"),
    },
    {
        "display": "Gemini 🔵",
        "key": "Gemini",
        "name": "Gemini",
        "maker": "Google DeepMind",
        "others": ("Claude (Anthropic)", "GPT-4 (OpenAI)"),
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


def _call_gemini(system: str, user: str, max_tokens: int = 500) -> str:
    resp = gemini.models.generate_content(
        model=GEMINI_MODEL,
        config=google_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
        contents=user,
    )
    return resp.text


CALLERS = {"Claude": _call_claude, "GPT-4": _call_gpt, "Gemini": _call_gemini}

# ── Debate turn ───────────────────────────────────────────────────────────────


def debate_turn(history: list[dict], topic: str, speaker: dict) -> str:
    system = DEBATE_SYSTEM.format(
        name=speaker["name"],
        maker=speaker["maker"],
        other1=speaker["others"][0],
        other2=speaker["others"][1],
        topic=topic,
    )
    if history:
        lines = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
        user = f"Transcript so far:\n\n{lines}\n\n---\nYour turn. Respond directly to what the others said."
    else:
        user = f'Open the debate on: "{topic}"'
    return CALLERS[speaker["key"]](system, user, max_tokens=450)


# ── Synthesis turn ────────────────────────────────────────────────────────────


def synthesis_turn(history: list[dict], topic: str, speaker: dict) -> str:
    system = SYNTHESIS_SYSTEM.format(
        name=speaker["name"],
        maker=speaker["maker"],
        other1=speaker["others"][0],
        other2=speaker["others"][1],
    )
    transcript = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
    user = SYNTHESIS_USER.format(transcript=transcript, topic=topic)
    return CALLERS[speaker["key"]](system, user, max_tokens=550)


# ── Final answer ──────────────────────────────────────────────────────────────


def final_answer(syntheses: dict[str, str], topic: str) -> str:
    user = FINAL_USER.format(
        topic=topic,
        claude_synth=syntheses["Claude"],
        gpt_synth=syntheses["GPT-4"],
        gemini_synth=syntheses["Gemini"],
    )
    # Claude produces the final answer, but it's explicitly told to synthesize
    # all three — not just argue its own position.
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

    history: list[dict] = []

    # ── Debate rounds ──
    for r in range(1, rounds + 1):
        subheader(f"Round {r}")
        for speaker in SPEAKERS:
            print(f"  {speaker['display']}")
            print(f"  {'─' * 40}")
            text = debate_turn(history, topic, speaker)
            print(wrap(text))
            print()
            history.append({"key": speaker["key"], "text": text})

    # ── Synthesis round ──
    header("SYNTHESIS — Finding common ground")
    syntheses: dict[str, str] = {}
    for speaker in SPEAKERS:
        print(f"  {speaker['display']} synthesizes:")
        print(f"  {'─' * 40}")
        text = synthesis_turn(history, topic, speaker)
        print(wrap(text))
        print()
        syntheses[speaker["key"]] = text

    # ── Final answer ──
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
