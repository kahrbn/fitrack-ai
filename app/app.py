import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# Config & Client
# ─────────────────────────────────────────────
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MEMORY_FILE = "fittrack_memory.json"
LOG_FILE = "fitness_log.json"

# ─────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────
def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# System Prompt builder (with memory injection)
# ─────────────────────────────────────────────
def build_system_prompt(memory: dict) -> str:
    memory_block = ""
    if memory:
        lines = []
        if memory.get("name"):
            lines.append(f"- User's name: {memory['name']}")
        if memory.get("goal"):
            lines.append(f"- Fitness goal: {memory['goal']}")
        if memory.get("level"):
            lines.append(f"- Experience level: {memory['level']}")
        if memory.get("injuries"):
            lines.append(f"- Known injuries/conditions: {memory['injuries']}")
        if memory.get("notes"):
            lines.append(f"- Additional notes: {memory['notes']}")
        if lines:
            memory_block = "\n\n📋 REMEMBERED USER PROFILE:\n" + "\n".join(lines)

    return f"""
🧠 System Prompt: AI Fitness Companion

If someone asks who made you or who developed you, always reply:
DEVELOPED BY RAZ ZALTE AND HITESH GOLHAR FROM LOKMANYA TILAK COLLEGE OF ENGINEERING.

You are a friendly, supportive, and knowledgeable AI fitness companion designed to help users
improve their physical and mental well-being. Your role is to guide, motivate, and educate users
on fitness, health, and lifestyle habits in a clear, practical, and encouraging way.

You behave like a reliable gym partner — motivating but realistic — balancing energy with evidence.

🎯 Core Responsibilities:
- Provide personalized fitness tips based on the user's goals (weight gain, fat loss, strength, endurance).
- Suggest workouts, routines, and exercises suitable for the user's experience level.
- Offer basic nutrition guidance (calories, protein intake, hydration, meal timing).
- Encourage consistency, discipline, and healthy habits.
- Help users track progress and stay accountable.
- REMEMBER and USE the user's profile info when giving advice.
- When the user shares personal info (name, goal, injury, level), acknowledge and remember it naturally.

⚠️ Safety Guidelines:
- Do not provide medical diagnoses or replace professional medical advice.
- If a user mentions injury, pain, or health conditions, recommend consulting a professional.
- Promote balanced, sustainable fitness habits over quick results.

🚫 Restrictions:
- Do not give dangerous fitness challenges.
- Do not promote unrealistic body standards.
- Do not act like a doctor or certified nutritionist.

💡 Goal: Make the user feel supported, consistent, and confident in their fitness journey.
{memory_block}
""".strip()

# ─────────────────────────────────────────────
# Memory extractor (simple keyword parse)
# ─────────────────────────────────────────────
def extract_memory_from_message(text: str, memory: dict) -> dict:
    """Cheaply extract key facts from user messages without extra API calls."""
    t = text.lower()
    updated = dict(memory)

    # Name
    for phrase in ["my name is ", "i am ", "i'm ", "call me "]:
        if phrase in t:
            idx = t.index(phrase) + len(phrase)
            word = text[idx:].split()[0].strip(".,!?")
            if len(word) > 1 and word[0].isupper():
                updated["name"] = word

    # Goal
    goal_keywords = {
        "lose weight": "fat loss", "fat loss": "fat loss", "weight loss": "fat loss",
        "gain muscle": "muscle gain", "build muscle": "muscle gain", "bulk": "muscle gain",
        "get stronger": "strength", "strength training": "strength",
        "endurance": "endurance", "run a marathon": "endurance",
        "stay fit": "general fitness", "stay active": "general fitness",
    }
    for kw, label in goal_keywords.items():
        if kw in t:
            updated["goal"] = label
            break

    # Level
    for kw in ["beginner", "intermediate", "advanced", "never worked out", "just starting"]:
        if kw in t:
            updated["level"] = kw
            break

    # Injuries
    injury_keywords = ["injury", "injured", "pain", "sprain", "fracture", "surgery", "bad knee", "bad back", "shoulder issue"]
    for kw in injury_keywords:
        if kw in t:
            snippet = text[max(0, t.index(kw)-10):t.index(kw)+50].strip()
            updated["injuries"] = snippet
            break

    return updated

# ─────────────────────────────────────────────
# Page config & Global CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitTrack AI 💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0c0f14;
    --surface:  #141820;
    --card:     #1a2030;
    --border:   #2a3348;
    --accent:   #f97316;
    --accent2:  #fb923c;
    --green:    #22c55e;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --font-display: 'Bebas Neue', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.2s;
    animation: fadeSlideUp 0.4s ease both;
}
.metric-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--font-mono);
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 2rem;
    color: var(--accent);
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

/* Log entries */
.log-entry {
    background: var(--card);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    animation: fadeSlideUp 0.3s ease both;
    font-size: 14px;
}
.log-entry .log-date {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 2px;
}

/* Streak badge */
.streak-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    font-family: var(--font-display);
    font-size: 1.4rem;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 1px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(249,115,22,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(249,115,22,0); }
}

/* Memory tag */
.mem-tag {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    margin: 3px;
    color: var(--text);
    animation: fadeSlideUp 0.3s ease both;
}
.mem-tag span { color: var(--accent); font-weight: 600; }

/* Animations */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    animation: fadeSlideUp 0.35s ease both !important;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.2) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(249,115,22,0.08) !important;
}

/* Selectbox / text_input */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Sidebar title */
.sidebar-title {
    font-family: var(--font-display);
    font-size: 1.6rem;
    letter-spacing: 2px;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 16px;
}

/* Main title */
.main-title {
    font-family: var(--font-display);
    font-size: 3rem;
    letter-spacing: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), #fff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin-bottom: 0;
}
.main-sub {
    color: var(--muted);
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: var(--font-mono);
    margin-bottom: 20px;
}

/* Progress bar custom */
.prog-bar-bg {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.6s ease;
}

/* Tabs */
[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Label colors */
label, .stMarkdown p { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {role, content, ts}

if "memory" not in st.session_state:
    st.session_state.memory = load_json(MEMORY_FILE, {})

if "fitness_log" not in st.session_state:
    st.session_state.fitness_log = load_json(LOG_FILE, [])

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# ─────────────────────────────────────────────
# Sidebar — Memory & Fitness Stats
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏋️ FITTRACK AI</div>', unsafe_allow_html=True)
    st.markdown("*by Raj Zalte & Hitesh Golhar*")
    st.markdown("---")

    # ── Memory panel ──────────────────────────
    st.markdown("### 🧠 User Memory")
    mem = st.session_state.memory
    if mem:
        html_tags = ""
        for k, v in mem.items():
            label = k.capitalize()
            html_tags += f'<div class="mem-tag"><span>{label}:</span> {v}</div>'
        st.markdown(html_tags, unsafe_allow_html=True)
    else:
        st.caption("Chat with the bot — it will remember you automatically.")

    if st.button("🗑 Clear Memory", use_container_width=True):
        st.session_state.memory = {}
        save_json(MEMORY_FILE, {})
        st.rerun()

    st.markdown("---")

    # ── Manual profile ─────────────────────────
    with st.expander("✏️ Edit Profile Manually"):
        name_in = st.text_input("Name", value=mem.get("name", ""), placeholder="e.g. Raj")
        goal_in = st.selectbox("Goal", ["", "fat loss", "muscle gain", "strength", "endurance", "general fitness"],
                               index=["", "fat loss", "muscle gain", "strength", "endurance", "general fitness"].index(mem.get("goal", "")) if mem.get("goal","") in ["","fat loss","muscle gain","strength","endurance","general fitness"] else 0)
        level_in = st.selectbox("Level", ["", "beginner", "intermediate", "advanced"],
                                index=["","beginner","intermediate","advanced"].index(mem.get("level","")) if mem.get("level","") in ["","beginner","intermediate","advanced"] else 0)
        injuries_in = st.text_input("Injuries / conditions", value=mem.get("injuries", ""), placeholder="e.g. bad knee")
        if st.button("💾 Save Profile", use_container_width=True):
            if name_in: st.session_state.memory["name"] = name_in
            if goal_in: st.session_state.memory["goal"] = goal_in
            if level_in: st.session_state.memory["level"] = level_in
            if injuries_in: st.session_state.memory["injuries"] = injuries_in
            save_json(MEMORY_FILE, st.session_state.memory)
            st.success("Saved!")

    st.markdown("---")

    # ── Streak calc ────────────────────────────
    log = st.session_state.fitness_log
    streak = 0
    if log:
        dates = sorted(set(e["date"] for e in log), reverse=True)
        today = datetime.date.today().isoformat()
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        if dates[0] in (today, yesterday):
            streak = 1
            for i in range(1, len(dates)):
                d1 = datetime.date.fromisoformat(dates[i-1])
                d2 = datetime.date.fromisoformat(dates[i])
                if (d1 - d2).days == 1:
                    streak += 1
                else:
                    break

    st.markdown("### 🔥 Current Streak")
    st.markdown(f'<div class="streak-badge">🔥 {streak} DAY{"S" if streak != 1 else ""}</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Quick stats ────────────────────────────
    total_workouts = len(set(e["date"] for e in log))
    total_exercises = len(log)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Workouts</div>
            <div class="metric-value">{total_workouts}</div>
        </div>''', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Exercises</div>
            <div class="metric-value">{total_exercises}</div>
        </div>''', unsafe_allow_html=True)

    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">FITTRACK AI</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Your intelligent gym companion — by Raj Zalte & Hitesh Golhar</div>', unsafe_allow_html=True)

tab_chat, tab_log, tab_history = st.tabs(["💬  Chat", "📊  Fitness Log", "📜  Chat History"])

# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════
with tab_chat:
    chat_container = st.container()

    with chat_container:
        # Render current session messages
        for msg in st.session_state.chat_history:
            role = msg["role"]
            if role == "system":
                continue
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(msg["content"])
                ts = msg.get("ts", "")
                if ts:
                    st.caption(f"🕐 {ts}")

    # ── Input ──────────────────────────────────
    user_input = st.chat_input("Ask about workouts, nutrition, goals...")

    if user_input:
        ts_now = datetime.datetime.now().strftime("%b %d, %H:%M")

        # Update memory from message
        st.session_state.memory = extract_memory_from_message(user_input, st.session_state.memory)
        save_json(MEMORY_FILE, st.session_state.memory)

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input, "ts": ts_now})

        # Build messages for API (include last 20 turns for context)
        sys_prompt = build_system_prompt(st.session_state.memory)
        api_messages = [{"role": "system", "content": sys_prompt}]
        # Keep last 20 messages (10 exchanges) for context window
        recent = [m for m in st.session_state.chat_history if m["role"] != "system"][-20:]
        for m in recent[:-1]:  # exclude the latest user msg we'll add fresh
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": user_input})

        # Show user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
                st.caption(f"🕐 {ts_now}")

        # Stream AI response
        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_reply = ""
                try:
                    stream = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=api_messages,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_reply += delta
                        placeholder.markdown(full_reply + "▌")
                    placeholder.markdown(full_reply)
                    ts_reply = datetime.datetime.now().strftime("%b %d, %H:%M")
                    st.caption(f"🕐 {ts_reply}")
                except Exception as e:
                    full_reply = f"⚠️ API error: {e}"
                    placeholder.markdown(full_reply)
                    ts_reply = ts_now

        # Save AI reply
        st.session_state.chat_history.append({"role": "assistant", "content": full_reply, "ts": ts_reply})

# ══════════════════════════════════════════════
# TAB 2 — FITNESS LOG
# ══════════════════════════════════════════════
with tab_log:
    st.markdown("### 📝 Log Today's Activity")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        exercise = st.text_input("Exercise", placeholder="e.g. Bench Press, Running, Squats")
    with col2:
        sets_reps = st.text_input("Sets × Reps / Duration", placeholder="e.g. 4×8 or 30 min")
    with col3:
        weight_kg = st.text_input("Weight (kg) optional", placeholder="e.g. 60")

    notes_log = st.text_area("Notes (optional)", placeholder="How did it feel? Any PRs?", height=68)

    if st.button("➕ Add Entry", use_container_width=False):
        if exercise.strip():
            entry = {
                "date": datetime.date.today().isoformat(),
                "time": datetime.datetime.now().strftime("%H:%M"),
                "exercise": exercise.strip(),
                "sets_reps": sets_reps.strip(),
                "weight": weight_kg.strip(),
                "notes": notes_log.strip(),
            }
            st.session_state.fitness_log.append(entry)
            save_json(LOG_FILE, st.session_state.fitness_log)
            st.success(f"✅ Logged: {exercise}")
            st.rerun()
        else:
            st.warning("Please enter an exercise name.")

    st.markdown("---")

    # ── Display log ───────────────────────────
    if st.session_state.fitness_log:
        # Group by date
        from collections import defaultdict
        grouped = defaultdict(list)
        for e in reversed(st.session_state.fitness_log):
            grouped[e["date"]].append(e)

        for date, entries in sorted(grouped.items(), reverse=True):
            st.markdown(f"#### 📅 {date}")
            for e in entries:
                badge = f"{e['sets_reps']}" if e["sets_reps"] else ""
                weight_badge = f" @ {e['weight']} kg" if e["weight"] else ""
                note_line = f"<br><span style='color:#64748b;font-size:12px'>{e['notes']}</span>" if e["notes"] else ""
                st.markdown(f'''
                <div class="log-entry">
                    <div class="log-date">🕐 {e["time"]}</div>
                    <strong>{e["exercise"]}</strong>
                    {"  <code style='background:#2a3348;padding:2px 6px;border-radius:4px;font-size:12px'>" + badge + weight_badge + "</code>" if badge or weight_badge else ""}
                    {note_line}
                </div>''', unsafe_allow_html=True)

        # ── Weekly summary ─────────────────────
        st.markdown("---")
        st.markdown("### 📊 Weekly Summary")
        week_start = datetime.date.today() - datetime.timedelta(days=6)
        week_entries = [e for e in st.session_state.fitness_log
                        if datetime.date.fromisoformat(e["date"]) >= week_start]
        week_days = len(set(e["date"] for e in week_entries))
        pct = min(week_days / 7 * 100, 100)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">Days Active</div>
                <div class="metric-value">{week_days}/7</div>
                <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct}%"></div></div>
            </div>''', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">Exercises This Week</div>
                <div class="metric-value">{len(week_entries)}</div>
            </div>''', unsafe_allow_html=True)
        with col_s3:
            unique_ex = len(set(e["exercise"] for e in week_entries))
            st.markdown(f'''<div class="metric-card">
                <div class="metric-label">Unique Exercises</div>
                <div class="metric-value">{unique_ex}</div>
            </div>''', unsafe_allow_html=True)

        if st.button("🗑 Clear All Log Data"):
            st.session_state.fitness_log = []
            save_json(LOG_FILE, [])
            st.rerun()
    else:
        st.info("No entries yet. Start logging your workouts above! 💪")

# ══════════════════════════════════════════════
# TAB 3 — CHAT HISTORY VIEWER
# ══════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Full Chat History")

    conv = [m for m in st.session_state.chat_history if m["role"] != "system"]
    if conv:
        for msg in conv:
            icon = "🧑" if msg["role"] == "user" else "🤖"
            bg = "#1a2030" if msg["role"] == "user" else "#141820"
            border = "#f97316" if msg["role"] == "user" else "#3b82f6"
            ts = msg.get("ts", "")
            st.markdown(f'''
            <div style="background:{bg};border-left:3px solid {border};border-radius:0 8px 8px 0;
                        padding:12px 16px;margin-bottom:10px;animation:fadeSlideUp 0.3s ease both">
                <div style="font-size:11px;color:#64748b;margin-bottom:4px;font-family:monospace">
                    {icon} {msg["role"].upper()}  {f"· {ts}" if ts else ""}
                </div>
                <div style="font-size:14px;line-height:1.6">{msg["content"]}</div>
            </div>''', unsafe_allow_html=True)

        # Export
        history_text = "\n\n".join(
            f"[{m.get('ts','')}] {m['role'].upper()}:\n{m['content']}"
            for m in conv
        )
        st.download_button(
            label="⬇️ Export Chat as .txt",
            data=history_text,
            file_name=f"fittrack_chat_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("No chat history yet. Start a conversation in the Chat tab!")