# =============================================================================
# ui/app.py — RACE Reading Comprehension & Quiz System
# All four required screens (spec §9.2):
#   Screen 1 — Article Input
#   Screen 2 — Question & Answer Quiz View
#   Screen 3 — Hint Panel
#   Screen 4 — Developer / Analytics Dashboard
#
# Run: streamlit run ui/app.py
# =============================================================================

import os
import sys
import time
import json
import datetime
import traceback
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Make src/ importable from ui/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Streamlit page config — must be FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RACE Quiz System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

/* ── Root variables ─────────────────────────────────────────────────────── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --accent:    #e6a817;
    --accent2:   #c08010;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --green:     #3fb950;
    --red:       #f85149;
    --blue:      #58a6ff;
    --radius:    10px;
}

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'Playfair Display', serif; color: var(--text); }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.2s, transform 0.1s;
    cursor: pointer;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ── Text areas & inputs ────────────────────────────────────────────────── */
textarea, .stTextArea textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
textarea:focus { border-color: var(--accent) !important; outline: none !important; }

/* ── Selectbox / radio ──────────────────────────────────────────────────── */
.stSelectbox > div, .stRadio > div { color: var(--text) !important; }
[data-baseweb="select"] { background: var(--surface2) !important; border-color: var(--border) !important; }
[data-baseweb="radio"] label { font-size: 14px !important; }

/* ── Cards ──────────────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 4px solid var(--accent); }
.card-green  { border-left: 4px solid var(--green); }
.card-red    { border-left: 4px solid var(--red); }
.card-blue   { border-left: 4px solid var(--blue); }

/* ── Metric boxes ───────────────────────────────────────────────────────── */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
}
.metric-box .metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-box .metric-lbl {
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Option buttons ─────────────────────────────────────────────────────── */
.option-btn {
    display: block;
    width: 100%;
    background: var(--surface2);
    border: 2px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1.2rem;
    margin: 0.4rem 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: var(--text);
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s, background 0.15s;
}
.option-btn:hover { border-color: var(--accent); background: #1d2229; }
.option-btn.selected { border-color: var(--accent); background: #1d2229; }
.option-btn.correct  { border-color: var(--green); background: #0d2116; }
.option-btn.wrong    { border-color: var(--red);   background: #2a0d0d; }

/* ── Hint cards ─────────────────────────────────────────────────────────── */
.hint-card {
    background: var(--surface2);
    border-left: 3px solid var(--blue);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 14px;
    line-height: 1.6;
}
.hint-card .hint-badge {
    display: inline-block;
    background: var(--blue);
    color: #0d1117;
    font-size: 11px;
    font-weight: 700;
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Article display ────────────────────────────────────────────────────── */
.article-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    font-size: 14px;
    line-height: 1.75;
    max-height: 280px;
    overflow-y: auto;
    color: var(--text);
}
.article-box::-webkit-scrollbar { width: 5px; }
.article-box::-webkit-scrollbar-track { background: var(--surface); }
.article-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Section titles ─────────────────────────────────────────────────────── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Tag badge ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
.badge-gold   { background: #3a2c00; color: var(--accent); border: 1px solid var(--accent); }
.badge-green  { background: #0d2116; color: var(--green);  border: 1px solid var(--green); }
.badge-red    { background: #2a0d0d; color: var(--red);    border: 1px solid var(--red); }
.badge-blue   { background: #0d1f3a; color: var(--blue);   border: 1px solid var(--blue); }

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ── Dataframe ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div { color: var(--accent) !important; }

/* ── Info / warning / error ─────────────────────────────────────────────── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}

/* WCAG AA contrast enforcement */
* { min-font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy-load pipeline (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline():
    try:
        from src.inference import load_pipeline
        return load_pipeline(), None
    except Exception as exc:
        return None, str(exc)


def _pipeline_ready():
    pipeline, err = get_pipeline()
    return pipeline is not None and pipeline.is_ready, pipeline, err


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        # Navigation
        "screen": "input",           # input | quiz | hints | dashboard
        # Article / question
        "article":    "",
        "question":   "",
        "options":    {},            # {"A":…,"B":…,"C":…,"D":…}
        "gold_answer":"",            # ground-truth key (if RACE sample)
        "correct_answer_text": "",   # text of correct option
        # Quiz state
        "selected_option":  None,    # key chosen by user
        "quiz_checked":     False,   # whether Check was pressed
        "predict_result":   None,    # dict from predict_answer
        "supporting_sent":  "",
        # Hints
        "hints":           [],
        "hints_revealed":  0,        # how many hints shown
        "answer_revealed": False,
        # Distractors
        "distractors":     [],
        # Analytics log
        "log": [],                   # list of dicts
        "session_start": datetime.datetime.now().isoformat(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Analytics logger
# ---------------------------------------------------------------------------

def _log(event: str, latency_ms: float = 0.0, extra: dict = None):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "event":     event,
        "latency_ms": round(latency_ms, 1),
    }
    if extra:
        entry.update(extra)
    st.session_state.log.append(entry)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:"Playfair Display",serif;font-size:1.4rem;
                    font-weight:900;color:#e6a817;line-height:1.2;'>
            RACE Quiz
        </div>
        <div style='font-size:12px;color:#8b949e;margin-top:4px;'>
            AI-Powered Reading Comprehension
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pages = {
        "📄  Article Input":      "input",
        "❓  Quiz View":          "quiz",
        "💡  Hint Panel":         "hints",
        "📊  Analytics Dashboard":"dashboard",
    }

    for label, screen_id in pages.items():
        active = st.session_state.screen == screen_id
        style  = "background:#21262d;border-left:3px solid #e6a817;" if active else ""
        if st.button(
            label,
            key=f"nav_{screen_id}",
            use_container_width=True,
        ):
            st.session_state.screen = screen_id
            st.rerun()

    st.markdown("---")

    # Pipeline status
    ready, pipeline, pipeline_err = _pipeline_ready()
    if ready:
        st.markdown('<span class="badge badge-green">● Pipeline Ready</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● Pipeline Offline</span>',
                    unsafe_allow_html=True)
        if pipeline_err:
            st.caption(pipeline_err)

    st.markdown(f"""
    <div style='font-size:11px;color:#8b949e;margin-top:1.5rem;'>
        Session started<br>{st.session_state.session_start[:19]}<br>
        Inferences: {len(st.session_state.log)}
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# SCREEN 1 — Article Input
# ===========================================================================

if st.session_state.screen == "input":

    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:2rem;
               margin-bottom:0.2rem;'>
        Reading Comprehension Quiz
    </h1>
    <p style='color:#8b949e;font-size:14px;margin-bottom:1.5rem;'>
        Paste a passage or load a random RACE sample — the AI will generate a
        quiz question, distractors, and graduated hints automatically.
    </p>
    """, unsafe_allow_html=True)

    # ── Random sample loader ────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 3])
    with col_l:
        split_choice = st.selectbox(
            "Dataset split",
            ["val", "train", "test"],
            label_visibility="collapsed",
        )
    with col_r:
        if st.button("🎲  Load Random RACE Sample", use_container_width=True):
            try:
                from src.inference import random_race_sample
                with st.spinner("Loading sample…"):
                    sample = random_race_sample(split_choice)
                st.session_state.article     = sample["article"]
                st.session_state.question    = sample["question"]
                st.session_state.options     = sample["options"]
                st.session_state.gold_answer = sample["answer"]
                st.session_state.correct_answer_text = sample["options"][sample["answer"]]
                # Reset downstream state
                st.session_state.quiz_checked     = False
                st.session_state.selected_option  = None
                st.session_state.predict_result   = None
                st.session_state.hints            = []
                st.session_state.hints_revealed   = 0
                st.session_state.answer_revealed  = False
                st.session_state.distractors      = []
                st.success("Sample loaded!")
                _log("load_sample", extra={"split": split_choice})
            except FileNotFoundError:
                st.error("RACE CSV not found. Place files in data/raw/ and reload.")
            except Exception as exc:
                st.error(f"Error loading sample: {exc}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Article text area ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📖 Reading Passage</div>',
                unsafe_allow_html=True)
    article_input = st.text_area(
        "Paste your article here",
        value=st.session_state.article,
        height=220,
        placeholder="Paste a reading passage here, or load a random RACE sample above…",
        label_visibility="collapsed",
    )
    st.session_state.article = article_input

    # ── Question (optional override) ────────────────────────────────────────
    with st.expander("✏️  Custom question (optional — leave blank to auto-generate)", expanded=False):
        q_input = st.text_input(
            "Question",
            value=st.session_state.question,
            placeholder="What did the protagonist decide to do?",
            label_visibility="collapsed",
        )
        st.session_state.question = q_input

        # Manual options (only shown if question is filled)
        if q_input.strip():
            st.markdown("**Answer options** (leave blank to auto-generate)")
            c1, c2 = st.columns(2)
            with c1:
                a_in = st.text_input("A", value=st.session_state.options.get("A",""), key="opt_A")
                c_in = st.text_input("C", value=st.session_state.options.get("C",""), key="opt_C")
            with c2:
                b_in = st.text_input("B", value=st.session_state.options.get("B",""), key="opt_B")
                d_in = st.text_input("D", value=st.session_state.options.get("D",""), key="opt_D")
            if any([a_in, b_in, c_in, d_in]):
                st.session_state.options = {"A": a_in, "B": b_in, "C": c_in, "D": d_in}

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Submit button ───────────────────────────────────────────────────────
    submit_col, _ = st.columns([1, 3])
    with submit_col:
        submit = st.button("🚀  Generate Quiz", use_container_width=True)

    if submit:
        if not st.session_state.article.strip():
            st.warning("Please paste an article or load a RACE sample first.")
        elif not ready:
            st.error(
                f"Pipeline not ready — train your models first.\n\n{pipeline_err or ''}"
            )
        else:
            with st.spinner("Running Model A & B inference…"):
                try:
                    from src.inference import (
                        predict_answer, generate_distractors, generate_hints
                    )
                    article  = st.session_state.article
                    question = st.session_state.question

                    # ── Auto-generate question if not provided ──────────────
                    if not question.strip():
                        # Template-based generation (spec §6.4)
                        from src.inference import _split_sentences, _clean, _tokenize
                        sentences = _split_sentences(article)
                        if sentences:
                            # pick sentence with highest token count as stem
                            best = max(sentences, key=lambda s: len(_tokenize(s)))
                            question = f"What is the main idea described in: '{best[:80]}…'?"
                        else:
                            question = "What is the main idea of the passage?"
                        st.session_state.question = question

                    # ── Options ─────────────────────────────────────────────
                    options = st.session_state.options
                    if not all(options.get(k, "").strip() for k in "ABCD"):
                        # Need to build options from correct answer + distractors
                        # First we need a correct answer candidate
                        from src.inference import _split_sentences
                        sents = _split_sentences(article)
                        correct_cand = sents[0][:100] if sents else article[:100]
                        st.session_state.correct_answer_text = correct_cand

                        t0 = time.perf_counter()
                        dists = generate_distractors(article, question, correct_cand)
                        dist_latency = (time.perf_counter() - t0) * 1000

                        all_opts = [correct_cand] + dists
                        import random as _random
                        _random.shuffle(all_opts)
                        keys = ["A", "B", "C", "D"]
                        options = {keys[i]: all_opts[i] for i in range(4)}
                        # Find which key has the correct answer
                        for k, v in options.items():
                            if v == correct_cand:
                                st.session_state.gold_answer = k
                                break
                        st.session_state.options = options
                        st.session_state.distractors = dists
                    else:
                        # Options already set (RACE sample)
                        correct_key  = st.session_state.gold_answer or "A"
                        correct_cand = options.get(correct_key, "")
                        t0 = time.perf_counter()
                        dists = generate_distractors(article, question, correct_cand)
                        dist_latency = (time.perf_counter() - t0) * 1000
                        st.session_state.distractors = dists

                    # ── Hints ────────────────────────────────────────────────
                    correct_text = st.session_state.correct_answer_text or options.get(
                        st.session_state.gold_answer or "A", ""
                    )
                    t0 = time.perf_counter()
                    hints = generate_hints(article, question, correct_text)
                    hint_latency = (time.perf_counter() - t0) * 1000

                    st.session_state.hints           = hints
                    st.session_state.hints_revealed  = 0
                    st.session_state.answer_revealed = False
                    st.session_state.quiz_checked    = False
                    st.session_state.selected_option = None
                    st.session_state.predict_result  = None

                    _log("generate_quiz",
                         latency_ms=dist_latency + hint_latency,
                         extra={"question_len": len(question)})

                    st.session_state.screen = "quiz"
                    st.rerun()

                except Exception as exc:
                    st.error(f"Inference error: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())


# ===========================================================================
# SCREEN 2 — Quiz View
# ===========================================================================

elif st.session_state.screen == "quiz":

    if not st.session_state.article.strip():
        st.warning("No article loaded. Go to **Article Input** first.")
        st.stop()

    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:1.8rem;
               margin-bottom:0.2rem;'>
        Quiz
    </h1>
    """, unsafe_allow_html=True)

    # ── Article display ─────────────────────────────────────────────────────
    with st.expander("📖  Reading Passage", expanded=False):
        st.markdown(
            f'<div class="article-box">{st.session_state.article}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Question ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="card card-accent">'
        f'<div class="section-title">❓ Question</div>'
        f'<p style="font-size:1.05rem;line-height:1.6;margin:0;">'
        f'{st.session_state.question}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Option selection ─────────────────────────────────────────────────────
    options = st.session_state.options
    if not options:
        st.error("No options available. Please go back to Article Input.")
        st.stop()

    st.markdown('<div class="section-title">📋 Choose an Answer</div>',
                unsafe_allow_html=True)

    checked   = st.session_state.quiz_checked
    selected  = st.session_state.selected_option
    gold      = st.session_state.gold_answer

    for key in ["A", "B", "C", "D"]:
        opt_text = options.get(key, "")
        if not opt_text.strip():
            continue

        if checked:
            if key == gold:
                css = "correct"
                prefix = "✅"
            elif key == selected and key != gold:
                css = "wrong"
                prefix = "❌"
            else:
                css = ""
                prefix = ""
            label = f"{prefix} **{key}.** {opt_text}"
        else:
            css    = "selected" if key == selected else ""
            label  = f"**{key}.** {opt_text}"

        col_btn, _ = st.columns([6, 1])
        with col_btn:
            if st.button(
                f"{key}.  {opt_text}",
                key=f"opt_btn_{key}",
                use_container_width=True,
                disabled=checked,
            ):
                st.session_state.selected_option = key
                st.rerun()

    # Highlight selection feedback
    if selected and not checked:
        st.markdown(
            f'<div class="card" style="margin-top:0.5rem;">'
            f'Selected: <strong>{selected}</strong> — {options.get(selected,"")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Check button ─────────────────────────────────────────────────────────
    col_check, col_hint, col_reset = st.columns([2, 2, 2])

    with col_check:
        if not checked:
            if st.button("✔️  Check Answer", use_container_width=True,
                         disabled=selected is None):
                if selected is None:
                    st.warning("Select an option first.")
                else:
                    ready, pipeline, _ = _pipeline_ready()
                    if not ready:
                        st.error("Pipeline not ready.")
                    else:
                        with st.spinner("Verifying…"):
                            from src.inference import predict_answer
                            t0  = time.perf_counter()
                            res = predict_answer(
                                st.session_state.article,
                                st.session_state.question,
                                st.session_state.options,
                            )
                            lat = (time.perf_counter() - t0) * 1000

                        # supporting sentence
                        supp = ""
                        if ready and pipeline:
                            supp = pipeline.supporting_sentence(
                                st.session_state.article,
                                st.session_state.correct_answer_text
                                or options.get(gold or "A", ""),
                            )

                        st.session_state.predict_result  = res
                        st.session_state.quiz_checked    = True
                        st.session_state.supporting_sent = supp

                        is_correct = selected == gold
                        _log(
                            "check_answer",
                            latency_ms=lat,
                            extra={
                                "selected": selected,
                                "gold": gold,
                                "correct": is_correct,
                                "model_predicted": res.get("predicted"),
                                "confidence": res.get("confidence"),
                            },
                        )
                        st.rerun()

    with col_hint:
        if st.button("💡  Get a Hint", use_container_width=True):
            st.session_state.screen = "hints"
            st.rerun()

    with col_reset:
        if st.button("🔄  New Question", use_container_width=True):
            st.session_state.screen = "input"
            st.rerun()

    # ── Result display ────────────────────────────────────────────────────────
    if checked and st.session_state.predict_result:
        res       = st.session_state.predict_result
        is_correct = selected == gold
        supp      = st.session_state.supporting_sent

        if is_correct:
            st.markdown(
                f'<div class="card card-green" style="margin-top:1rem;">'
                f'<div style="font-size:1.3rem;font-weight:700;color:#3fb950;">'
                f'🎉 Correct!</div>'
                f'<p style="margin:0.5rem 0 0 0;color:#8b949e;font-size:13px;">'
                f'Confidence: {res.get("confidence", 0):.0%} &nbsp;·&nbsp; '
                f'Latency: {res.get("latency_ms",0):.0f} ms</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card card-red" style="margin-top:1rem;">'
                f'<div style="font-size:1.3rem;font-weight:700;color:#f85149;">'
                f'✗ Incorrect</div>'
                f'<p style="margin:0.3rem 0 0 0;">The correct answer was '
                f'<strong>{gold}</strong>: {options.get(gold,"")}</p>'
                f'<p style="margin:0.2rem 0 0 0;color:#8b949e;font-size:13px;">'
                f'Model predicted: <strong>{res.get("predicted")}</strong> '
                f'(confidence {res.get("confidence",0):.0%})</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if supp:
            st.markdown(
                f'<div class="card card-blue" style="margin-top:0.6rem;">'
                f'<div class="section-title" style="font-size:0.9rem;">📌 Supporting Evidence</div>'
                f'<p style="font-size:13px;line-height:1.65;margin:0;font-style:italic;">'
                f'"{supp}"</p></div>',
                unsafe_allow_html=True,
            )

        # All four option scores bar chart
        scores = res.get("scores", {})
        if scores:
            fig = go.Figure(go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=[
                    "#3fb950" if k == gold else
                    "#f85149" if k == selected and k != gold else
                    "#58a6ff"
                    for k in scores
                ],
                text=[f"{v:.0%}" for v in scores.values()],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                yaxis=dict(range=[0, 1.1], gridcolor="#30363d", tickformat=".0%"),
                xaxis=dict(tickfont=dict(size=14, family="DM Mono")),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # AI disclaimer (spec §15.2)
        st.markdown(
            '<p style="font-size:11px;color:#8b949e;margin-top:0.5rem;">'
            '⚠️ Answers and distractors are AI-generated. Errors are possible. '
            'Not suitable for real exam use without expert review.</p>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# SCREEN 3 — Hint Panel
# ===========================================================================

elif st.session_state.screen == "hints":

    if not st.session_state.article.strip():
        st.warning("No article loaded. Go to **Article Input** first.")
        st.stop()

    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:1.8rem;
               margin-bottom:0.2rem;'>
        Hint Panel
    </h1>
    <p style='color:#8b949e;font-size:14px;margin-bottom:1.2rem;'>
        Hints go from broad context clues to near-explicit guidance.
        Use them wisely before revealing the answer.
    </p>
    """, unsafe_allow_html=True)

    # Show question
    st.markdown(
        f'<div class="card card-accent">'
        f'<div class="section-title">❓ Question</div>'
        f'<p style="font-size:1rem;margin:0;">{st.session_state.question}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    hints = st.session_state.hints
    if not hints:
        st.info("No hints generated yet. Submit an article first.")
    else:
        hint_labels = [
            ("Hint 1 — General Context",     "🔵", "#58a6ff"),
            ("Hint 2 — Narrowed Focus",      "🟡", "#e6a817"),
            ("Hint 3 — Near-Explicit Clue",  "🟠", "#f0883e"),
        ]

        n_revealed = st.session_state.hints_revealed

        for i, hint in enumerate(hints[:3]):
            label, icon, color = hint_labels[i] if i < len(hint_labels) else (f"Hint {i+1}", "●", "#58a6ff")

            if i < n_revealed:
                st.markdown(
                    f'<div class="hint-card" style="border-left-color:{color};">'
                    f'<div class="hint-badge" style="background:{color};">{label}</div><br>'
                    f'{hint}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="hint-card" style="border-left-color:#30363d;'
                    f'opacity:0.35;filter:blur(2px);">'
                    f'<div class="hint-badge" style="background:#30363d;color:#8b949e;">'
                    f'{label}</div><br>{'▓' * 60}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        col_reveal_hint, col_back = st.columns([2, 2])
        with col_reveal_hint:
            if n_revealed < len(hints):
                next_label = hint_labels[n_revealed][0] if n_revealed < len(hint_labels) else f"Hint {n_revealed+1}"
                if st.button(f"💡  Reveal {next_label}", use_container_width=True):
                    st.session_state.hints_revealed += 1
                    _log("reveal_hint", extra={"hint_number": st.session_state.hints_revealed})
                    st.rerun()
            else:
                st.markdown(
                    '<span class="badge badge-gold">All hints revealed</span>',
                    unsafe_allow_html=True,
                )

        # Reveal Answer — only after ALL hints shown (spec §9.2)
        if n_revealed >= len(hints) and not st.session_state.answer_revealed:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:13px;color:#8b949e;">'
                'You have used all hints. You may now reveal the answer.</p>',
                unsafe_allow_html=True,
            )
            if st.button("🔓  Reveal Answer", use_container_width=True):
                st.session_state.answer_revealed = True
                _log("reveal_answer")
                st.rerun()

        if st.session_state.answer_revealed:
            gold = st.session_state.gold_answer
            opts = st.session_state.options
            correct_text = opts.get(gold, st.session_state.correct_answer_text)
            st.markdown(
                f'<div class="card card-green" style="margin-top:1rem;">'
                f'<div style="font-size:1.1rem;font-weight:700;color:#3fb950;">'
                f'✅ Correct Answer: {gold}</div>'
                f'<p style="margin:0.4rem 0 0 0;">{correct_text}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_back:
            if st.button("← Back to Quiz", use_container_width=True):
                st.session_state.screen = "quiz"
                st.rerun()


# ===========================================================================
# SCREEN 4 — Analytics Dashboard
# ===========================================================================

elif st.session_state.screen == "dashboard":

    st.markdown("""
    <h1 style='font-family:"Playfair Display",serif;font-size:1.8rem;
               margin-bottom:0.2rem;'>
        Analytics Dashboard
    </h1>
    <p style='color:#8b949e;font-size:14px;margin-bottom:1.2rem;'>
        Live performance metrics for Model A (answer verification) and
        Model B (distractor &amp; hint generation) across this session.
    </p>
    """, unsafe_allow_html=True)

    log = st.session_state.log

    # ── Session summary ───────────────────────────────────────────────────────
    check_events = [e for e in log if e["event"] == "check_answer"]
    n_total   = len(check_events)
    n_correct = sum(1 for e in check_events if e.get("correct"))
    accuracy  = n_correct / n_total if n_total else 0.0
    avg_lat   = np.mean([e["latency_ms"] for e in check_events]) if check_events else 0.0
    n_hints   = len([e for e in log if e["event"] == "reveal_hint"])

    col1, col2, col3, col4 = st.columns(4)
    for col, val, lbl in [
        (col1, n_total,            "Quizzes Taken"),
        (col2, f"{accuracy:.0%}", "Session Accuracy"),
        (col3, f"{avg_lat:.0f}ms","Avg Latency"),
        (col4, n_hints,            "Hints Used"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model A metrics ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Model A — Answer Verification")

    ready, pipeline, _ = _pipeline_ready()
    if ready and pipeline:
        available = {}
        model_map = {
            "Logistic Regression": pipeline.lr_classifier,
            "SVM":                 pipeline.svm_classifier,
            "Naive Bayes":         pipeline.nb_classifier,
            "Random Forest":       pipeline.rf_classifier,
            "XGBoost":             pipeline.xgb_classifier,
            "Ensemble":            pipeline.ensemble_model,
        }
        for name, mdl in model_map.items():
            if mdl is not None:
                available[name] = "✅ Loaded"
            else:
                available[name] = "⬜ Not trained"

        model_df = pd.DataFrame(
            {"Model": list(available.keys()), "Status": list(available.values())}
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    if check_events:
        # Confusion matrix from session log
        tp = sum(1 for e in check_events if e.get("correct") and e.get("selected") == e.get("gold"))
        fp = sum(1 for e in check_events if not e.get("correct") and e.get("model_predicted") != e.get("gold"))
        fn = sum(1 for e in check_events if e.get("correct") and e.get("model_predicted") != e.get("selected"))
        tn = n_total - tp - fp - fn
        tn = max(tn, 0)

        cm = np.array([[tp, fn], [fp, tn]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Correct", "Predicted Wrong"],
            y=["Actually Correct",  "Actually Wrong"],
            colorscale=[[0,"#161b22"],[0.5,"#214a7a"],[1,"#58a6ff"]],
            text=cm.astype(str), texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            margin=dict(t=20, b=20, l=10, r=10),
            height=240,
            title="Session Confusion Matrix",
            title_font=dict(size=13, family="DM Sans"),
        )
        st.plotly_chart(fig_cm, use_container_width=True,
                        config={"displayModeBar": False})

        # Confidence distribution
        confs = [e.get("confidence", 0) for e in check_events]
        if confs:
            fig_conf = go.Figure(go.Histogram(
                x=confs, nbinsx=10,
                marker_color="#e6a817", opacity=0.8,
            ))
            fig_conf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                xaxis=dict(title="Confidence", gridcolor="#30363d"),
                yaxis=dict(title="Count",      gridcolor="#30363d"),
                margin=dict(t=20, b=20, l=10, r=10),
                height=220,
                title="Confidence Distribution",
                title_font=dict(size=13, family="DM Sans"),
            )
            st.plotly_chart(fig_conf, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info("No quiz events logged yet this session.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model B metrics ───────────────────────────────────────────────────────
    st.markdown("### 🎯 Model B — Distractor & Hint Generation")

    if ready and pipeline:
        b_items = {
            "Distractor Ranker": pipeline.distractor_ranker,
            "Hint Scorer":       pipeline.hint_scorer,
            "Word2Vec":          pipeline.w2v,
        }
        b_df = pd.DataFrame({
            "Component": list(b_items.keys()),
            "Status": ["✅ Loaded" if v is not None else "⬜ Not trained"
                       for v in b_items.values()],
        })
        st.dataframe(b_df, use_container_width=True, hide_index=True)

    hint_events = [e for e in log if e["event"] == "reveal_hint"]
    if hint_events:
        hint_nums = [e.get("hint_number", 1) for e in hint_events]
        hc = Counter(hint_nums)
        fig_hints = go.Figure(go.Bar(
            x=[f"Hint {k}" for k in sorted(hc)],
            y=[hc[k] for k in sorted(hc)],
            marker_color=["#58a6ff", "#e6a817", "#f0883e"][:len(hc)],
            text=[str(hc[k]) for k in sorted(hc)],
            textposition="outside",
        ))
        fig_hints.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            margin=dict(t=20, b=20, l=10, r=10),
            height=220,
            yaxis=dict(gridcolor="#30363d"),
            title="Hints Requested per Level",
            title_font=dict(size=13, family="DM Sans"),
        )
        st.plotly_chart(fig_hints, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Inference latency chart ───────────────────────────────────────────────
    st.markdown("### ⏱ Inference Latency (ms)")

    latency_events = [e for e in log if e["latency_ms"] > 0]
    if latency_events:
        fig_lat = go.Figure(go.Scatter(
            x=list(range(1, len(latency_events)+1)),
            y=[e["latency_ms"] for e in latency_events],
            mode="lines+markers",
            line=dict(color="#e6a817", width=2),
            marker=dict(size=6, color="#e6a817"),
            text=[e["event"] for e in latency_events],
            hovertemplate="%{text}<br>%{y:.0f} ms<extra></extra>",
        ))
        # 10 s limit line
        fig_lat.add_hline(
            y=10_000, line_dash="dot", line_color="#f85149",
            annotation_text="10s limit", annotation_position="right",
        )
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3",
            xaxis=dict(title="Request #", gridcolor="#30363d"),
            yaxis=dict(title="Latency (ms)", gridcolor="#30363d"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
        )
        st.plotly_chart(fig_lat, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Session log table + CSV export ───────────────────────────────────────
    st.markdown("### 📋 Session Event Log")

    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        col_exp, col_clr = st.columns([2, 1])
        with col_exp:
            csv_data = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Export Session Log (CSV)",
                data=csv_data,
                file_name=f"race_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️  Clear Log", use_container_width=True):
                st.session_state.log = []
                st.rerun()
    else:
        st.info("No events logged yet. Take a quiz to see data here.")

    # AI transparency notice (spec §15.2)
    st.markdown(
        '<p style="font-size:11px;color:#8b949e;margin-top:1rem;">'
        '⚠️ All questions, answers, distractors, and hints displayed in this '
        'application are AI-generated. Results should be reviewed by a human '
        'expert before use in any formal educational setting.</p>',
        unsafe_allow_html=True,
    )