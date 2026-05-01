#!/usr/bin/env python3
"""
AI Thunderdome — Streamlit web app
Claude vs GPT-4: debate → synthesize → final answer
With persistent memory: learns key insights from past debates

Run:  streamlit run app.py
"""
import os
import json
import uuid
import datetime
import re
import anthropic
import openai
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="beef.exe",
    page_icon="🥩",
    layout="centered",
)

st.markdown(
    """
    <style>
    .final-answer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: #eee;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .phase-header {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin: 2rem 0 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Models ─────────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-opus-4-7"
GPT_MODEL = "gpt-4o"

SPEAKERS = [
    {
        "key": "Claude",
        "display": "Claude",
        "maker": "Anthropic",
        "avatar": "🟠",
        "other": "GPT-4 (OpenAI)",
    },
    {
        "key": "GPT-4",
        "display": "GPT-4",
        "maker": "OpenAI",
        "avatar": "🟢",
        "other": "Claude (Anthropic)",
    },
]

# ── Memory ─────────────────────────────────────────────────────────────────────

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "or", "and", "but", "if", "it", "its", "this",
    "that", "than", "so", "what", "which", "who", "how", "why", "when",
    "where", "not", "no", "vs", "between", "better", "more", "less",
}


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_memory_entry(entry):
    memories = load_memory()
    memories.append(entry)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def _topic_words(text):
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    return words - _STOP_WORDS


def find_relevant_memories(topic, memories, n=3):
    if not memories:
        return []
    tw = _topic_words(topic)
    scored = []
    for m in memories:
        mw = _topic_words(m.get("topic", "")) | set(m.get("tags", []))
        score = len(tw & mw)
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:n]]


def format_memory_context(memories):
    if not memories:
        return ""
    lines = ["Relevant findings from past debates:"]
    for m in memories:
        lines.append(f'  • "{m["topic"]}": {m["consensus"]} — {m["insight"]}')
    return "\n".join(lines)


# ── Prompts ────────────────────────────────────────────────────────────────────

DEBATE_SYSTEM = (
    "You are {name}, made by {maker}. "
    "You're in a head-to-head debate with {other} on: \"{topic}\". "
    "Argue your perspective confidently. Push back on your opponent's weak points. "
    "Be sharp and a little competitive. 2–3 punchy paragraphs, no bullet points."
)

SYNTHESIS_SYSTEM = (
    "You are {name}, made by {maker}. "
    "You just finished debating {other}. The debate is over — now you're collaborating."
    "{memory_note}"
)

SYNTHESIS_USER = (
    "Full debate transcript:\n\n{transcript}\n\n---\n"
    "The debate is done. Contribute your best thinking toward a final answer to: \"{topic}\"\n\n"
    "Be honest: acknowledge the strongest points made by your opponent, concede anything weaker "
    "in your own arguments, and distill what you genuinely believe is correct and important. "
    "2–3 paragraphs. This will be combined with your opponent's synthesis."
)

FINAL_SYSTEM = (
    "You are an expert synthesizer. Claude and GPT-4 debated a topic and each contributed a synthesis. "
    "Produce one definitive, well-reasoned answer by combining the strongest insights from both. "
    "Be thorough but clear. Start directly with the answer — no preamble about 'two AIs debated'. "
    "Use paragraphs, not bullet points."
)

FINAL_USER = (
    "Question: \"{topic}\"\n\n"
    "{memory_section}"
    "Claude's synthesis:\n{claude_synth}\n\n"
    "GPT-4's synthesis:\n{gpt_synth}\n\n---\n"
    "Produce the single best answer to the question above. "
    "Resolve any remaining disagreements with your best judgment."
)

MEMORY_EXTRACT_SYSTEM = (
    "You are a debate memory distiller. Given a debate topic and its final synthesized answer, "
    "extract the essential learning as compact JSON with exactly these keys:\n"
    '  "consensus": one sentence — the main conclusion or winning view\n'
    '  "insight": one sentence — the most important nuance, caveat, or surprising finding\n'
    '  "tags": array of 3-5 single lowercase words for retrieval\n'
    "Return only the JSON object, no markdown fences, no explanation."
)

MEMORY_EXTRACT_USER = 'Topic: "{topic}"\n\nFinal answer:\n{answer}'

# ── API helpers ────────────────────────────────────────────────────────────────


def call_claude(clients, system, user, max_tokens=500):
    resp = clients["claude"].messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_gpt(clients, system, user, max_tokens=500):
    resp = clients["gpt"].chat.completions.create(
        model=GPT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


CALLERS = {"Claude": call_claude, "GPT-4": call_gpt}


def debate_turn(clients, history, topic, speaker):
    system = DEBATE_SYSTEM.format(
        name=speaker["key"],
        maker=speaker["maker"],
        other=speaker["other"],
        topic=topic,
    )
    if history:
        lines = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
        user = f"Transcript:\n\n{lines}\n\n---\nYour turn. Respond directly to what your opponent said."
    else:
        user = f'Open the debate on: "{topic}"'
    return CALLERS[speaker["key"]](clients, system, user, max_tokens=450)


def synthesis_turn(clients, history, topic, speaker, memory_context=""):
    memory_note = f"\n\n{memory_context}" if memory_context else ""
    system = SYNTHESIS_SYSTEM.format(
        name=speaker["key"],
        maker=speaker["maker"],
        other=speaker["other"],
        memory_note=memory_note,
    )
    transcript = "\n\n".join(f"[{e['key']}]: {e['text']}" for e in history)
    user = SYNTHESIS_USER.format(transcript=transcript, topic=topic)
    return CALLERS[speaker["key"]](clients, system, user, max_tokens=550)


def get_final_answer(clients, syntheses, topic, memory_context=""):
    memory_section = f"{memory_context}\n\n" if memory_context else ""
    user = FINAL_USER.format(
        topic=topic,
        memory_section=memory_section,
        claude_synth=syntheses["Claude"],
        gpt_synth=syntheses["GPT-4"],
    )
    return call_claude(clients, FINAL_SYSTEM, user, max_tokens=900)


def extract_and_save_memory(clients, topic, answer, history):
    user = MEMORY_EXTRACT_USER.format(topic=topic, answer=answer)
    try:
        raw = call_claude(clients, MEMORY_EXTRACT_SYSTEM, user, max_tokens=200)
        raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        entry = {
            "id": str(uuid.uuid4()),
            "date": datetime.date.today().isoformat(),
            "topic": topic,
            "consensus": data.get("consensus", ""),
            "insight": data.get("insight", ""),
            "tags": data.get("tags", []),
            "final_answer": answer,
            "transcript": [{"speaker": e["key"], "text": e["text"]} for e in history],
        }
        save_memory_entry(entry)
        return entry
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    rounds = st.slider("Debate rounds", min_value=1, max_value=4, value=2)

    st.divider()
    st.subheader("API Keys")
    st.caption("Leave blank to use environment variables.")

    anthropic_key = st.text_input(
        "Anthropic", type="password",
        placeholder=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-..."),
    )
    openai_key = st.text_input(
        "OpenAI", type="password",
        placeholder=os.environ.get("OPENAI_API_KEY", "sk-..."),
    )

    st.divider()
    st.caption("**How it works**")
    st.caption(
        "1. Claude and GPT-4 argue for N rounds\n"
        "2. Each contributes a synthesis\n"
        "3. A final answer is assembled from both\n"
        "4. Key learnings are saved to memory"
    )

    # ── Past Debates ──
    st.divider()
    st.subheader("📚 Past Debates")
    memories = load_memory()
    if memories:
        st.caption(f"{len(memories)} debate{'s' if len(memories) != 1 else ''} on record")
        for m in reversed(memories[-15:]):
            label = m["topic"][:45] + ("…" if len(m["topic"]) > 45 else "")
            with st.expander(label):
                st.markdown(f"**{m['date']}**")
                st.markdown(f"**Consensus:** {m['consensus']}")
                st.markdown(f"**Key insight:** {m['insight']}")
                if m.get("tags"):
                    st.caption("Tags: " + ", ".join(m["tags"]))
                if m.get("transcript"):
                    with st.expander("Full transcript"):
                        for turn in m["transcript"]:
                            st.markdown(f"**{turn['speaker']}:** {turn['text']}")
                if m.get("final_answer"):
                    with st.expander("Final answer"):
                        st.markdown(m["final_answer"])
    else:
        st.caption("No past debates yet.")

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🥩 beef.exe")
st.caption("Claude · GPT-4 debate your question — then give you the best possible answer.")

if "topic_input" not in st.session_state:
    st.session_state["topic_input"] = ""

topic = st.text_area(
    "What's the question?",
    placeholder="e.g. Is remote work better than going to the office?",
    label_visibility="collapsed",
    key="topic_input",
    height=80,
)

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button(
        "⚡  Start Debate",
        type="primary",
        disabled=not topic.strip(),
        use_container_width=True,
    )
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["topic_input"] = ""
        st.rerun()

# ── Run ────────────────────────────────────────────────────────────────────────

if run and topic.strip():
    try:
        clients = {
            "claude": anthropic.Anthropic(api_key=anthropic_key or os.environ.get("ANTHROPIC_API_KEY")),
            "gpt": openai.OpenAI(api_key=openai_key or os.environ.get("OPENAI_API_KEY")),
        }
    except Exception as e:
        st.error(f"Could not initialize API clients: {e}")
        st.stop()

    # Load relevant memories for this topic
    all_memories = load_memory()
    relevant = find_relevant_memories(topic, all_memories)
    memory_context = format_memory_context(relevant)

    if memory_context:
        with st.expander("🧠 Memory — what they've learned from past debates on this topic", expanded=True):
            for m in relevant:
                st.markdown(f"**{m['topic']}** *({m['date']})*")
                st.caption(f"{m['consensus']} — {m['insight']}")

    history = []

    # ── Debate rounds ──
    for r in range(1, rounds + 1):
        st.markdown(f'<p class="phase-header">Round {r}</p>', unsafe_allow_html=True)

        for speaker in SPEAKERS:
            with st.chat_message(speaker["display"], avatar=speaker["avatar"]):
                placeholder = st.empty()
                placeholder.markdown("*marinating...*")
                try:
                    text = debate_turn(clients, history, topic, speaker)
                except Exception as e:
                    placeholder.error(f"Error: {e}")
                    st.stop()
                placeholder.markdown(text)
            history.append({"key": speaker["key"], "text": text})

    # ── Synthesis ──
    st.markdown('<p class="phase-header">Synthesis — finding common ground</p>', unsafe_allow_html=True)

    syntheses = {}
    for speaker in SPEAKERS:
        with st.chat_message(speaker["display"], avatar=speaker["avatar"]):
            placeholder = st.empty()
            placeholder.markdown("*slow roasting...*")
            try:
                text = synthesis_turn(clients, history, topic, speaker, memory_context)
            except Exception as e:
                placeholder.error(f"Error: {e}")
                st.stop()
            placeholder.markdown(text)
        syntheses[speaker["key"]] = text

    # ── Final answer ──
    st.markdown('<p class="phase-header">✦ Final Answer</p>', unsafe_allow_html=True)

    with st.spinner("Searing the final cut..."):
        try:
            answer = get_final_answer(clients, syntheses, topic, memory_context)
        except Exception as e:
            st.error(f"Error generating final answer: {e}")
            st.stop()

    st.markdown(
        f'<div class="final-answer">{answer.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # ── Save memory ──
    with st.spinner("Storing leftovers..."):
        entry = extract_and_save_memory(clients, topic, answer, history)

    if entry:
        st.success(f"✓ Learned: {entry['consensus']}")

    components.html("""
    <canvas id="fw" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;"></canvas>
    <script>
    const c = document.getElementById('fw');
    const ctx = c.getContext('2d');
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    const colors = ['#ff0000','#ff3300','#ff6600','#ff9900','#ffcc00','#ff0044','#cc0000'];
    let particles = [];
    function burst(x, y) {
        for (let i = 0; i < 120; i++) {
            const angle = (Math.PI * 2 / 120) * i;
            const speed = Math.random() * 8 + 2;
            particles.push({
                x, y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                alpha: 1,
                color: colors[Math.floor(Math.random() * colors.length)],
                r: Math.random() * 3 + 1
            });
        }
    }
    let count = 0;
    function launch() {
        burst(Math.random() * c.width, Math.random() * c.height * 0.6);
        count++;
        if (count < 8) setTimeout(launch, 350);
    }
    function draw() {
        ctx.clearRect(0, 0, c.width, c.height);
        particles = particles.filter(p => p.alpha > 0.01);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.12;
            p.alpha -= 0.018;
            ctx.globalAlpha = p.alpha;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;
        if (particles.length > 0 || count < 8) requestAnimationFrame(draw);
    }
    launch();
    draw();
    </script>
    """, height=0)
