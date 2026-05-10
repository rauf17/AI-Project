# =============================================================================
# ui/app.py — RACE Reading Comprehension & Quiz System
# Revamped UI: Deep-sea Glassmorphism · Teal/Emerald palette
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
# Global CSS — Deep-sea Glassmorphism + Teal/Emerald
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,700;0,9..144,900;1,9..144,300;1,9..144,500&family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ─────────────────────────────────────────────────────── */
:root {
    --bg0:          #020d0b;
    --bg1:          #061410;
    --bg2:          #0a1e19;
    --bg3:          #0d2820;
    --glass:        rgba(13, 40, 32, 0.60);
    --glass-heavy:  rgba(6, 20, 16, 0.80);
    --glass-light:  rgba(20, 60, 48, 0.35);
    --glow:         rgba(45, 212, 191, 0.18);
    --glow2:        rgba(16, 185, 129, 0.12);
    --border:       rgba(45, 212, 191, 0.14);
    --border2:      rgba(45, 212, 191, 0.28);
    --teal:         #2dd4bf;
    --teal2:        #14b8a6;
    --emerald:      #10b981;
    --emerald2:     #059669;
    --mint:         #6ee7b7;
    --sky:          #67e8f9;
    --amber:        #f59e0b;
    --coral:        #fb7185;
    --text:         #f0fdf9;
    --text2:        #99f6e4;
    --muted:        #5eead4;
    --dim:          #2d6b5e;
    --blur:         blur(24px);
    --blur2:        blur(14px);
    --r-sm:         8px;
    --r:            14px;
    --r-lg:         22px;
    --r-xl:         32px;
}

/* ── Animated radial mesh background ───────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
}

[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 900px 600px at 10% 15%, rgba(45,212,191,0.07) 0%, transparent 65%),
        radial-gradient(ellipse 600px 800px at 90% 85%, rgba(16,185,129,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 700px 400px at 50% 50%, rgba(20,184,166,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: meshDrift 18s ease-in-out infinite alternate;
}

@keyframes meshDrift {
    0%   { opacity: 0.8; }
    100% { opacity: 1.0; }
}

/* ── Noise grain overlay ────────────────────────────────────────────────── */
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1;
    opacity: 0.4;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--glass-heavy) !important;
    backdrop-filter: var(--blur) !important;
    -webkit-backdrop-filter: var(--blur) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Main block padding ─────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1100px;
}

/* ── Page entry animation ───────────────────────────────────────────────── */
[data-testid="stVerticalBlock"] {
    animation: pageIn 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes pageIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Headers ────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Fraunces', serif;
    color: var(--text);
    letter-spacing: -0.02em;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(145deg, var(--teal2), var(--emerald2)) !important;
    color: var(--bg0) !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    letter-spacing: 0.03em !important;
    padding: 0.58rem 1.5rem !important;
    box-shadow: 0 0 0 1px rgba(45,212,191,0.3), 0 4px 20px rgba(20,184,166,0.25) !important;
    transition: all 0.2s cubic-bezier(0.22,1,0.36,1) !important;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.stButton > button:hover {
    background: linear-gradient(145deg, var(--teal), var(--teal2)) !important;
    box-shadow: 0 0 0 1px rgba(45,212,191,0.5), 0 8px 32px rgba(45,212,191,0.35) !important;
    transform: translateY(-2px) scale(1.01) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 0 0 1px rgba(45,212,191,0.3), 0 2px 8px rgba(20,184,166,0.2) !important;
}
.stButton > button:disabled {
    background: rgba(45,212,191,0.1) !important;
    color: var(--dim) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Text areas ─────────────────────────────────────────────────────────── */
textarea, .stTextArea textarea {
    background: var(--glass) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    backdrop-filter: var(--blur2) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: vertical;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.15), 0 2px 16px rgba(45,212,191,0.1) !important;
    outline: none !important;
}

/* ── Text input ─────────────────────────────────────────────────────────── */
input[type="text"], .stTextInput input {
    background: var(--glass) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Syne', sans-serif !important;
    backdrop-filter: var(--blur2) !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.15) !important;
    outline: none !important;
}

/* ── Selectbox ──────────────────────────────────────────────────────────── */
.stSelectbox > div { color: var(--text) !important; }
[data-baseweb="select"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    backdrop-filter: var(--blur2) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; background: var(--bg2) !important; }

/* ── Glass card ─────────────────────────────────────────────────────────── */
.gcard {
    background: var(--glass);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}
.gcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.3), transparent);
}
.gcard:hover {
    border-color: rgba(45,212,191,0.25);
    box-shadow: 0 12px 50px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}
.gcard-teal   { border-left: 3px solid var(--teal); }
.gcard-emerald{ border-left: 3px solid var(--emerald); background: rgba(16,185,129,0.06); }
.gcard-coral  { border-left: 3px solid var(--coral); background: rgba(251,113,133,0.05); }
.gcard-sky    { border-left: 3px solid var(--sky); }
.gcard-amber  { border-left: 3px solid var(--amber); }

/* ── Metric boxes ───────────────────────────────────────────────────────── */
.mbox {
    background: var(--glass);
    backdrop-filter: var(--blur2);
    -webkit-backdrop-filter: var(--blur2);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 1.4rem 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.mbox::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    opacity: 0.5;
}
.mbox:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(45,212,191,0.12);
    border-color: rgba(45,212,191,0.25);
}
.mbox .mv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: var(--teal);
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.mbox .ml {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 6px;
    font-family: 'Syne', sans-serif;
}

/* ── Answer option buttons — purely via st.button CSS overrides ─────────── */

/* Base option style — overrides the global button style */
div[data-testid="stButton"]:has(button[kind="secondary"]) > button,
div[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--glass) !important;
    backdrop-filter: var(--blur2) !important;
    -webkit-backdrop-filter: var(--blur2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 0.9rem 1.4rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
    transition: all 0.18s cubic-bezier(0.22,1,0.36,1) !important;
    margin: 0.22rem 0 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(45,212,191,0.38) !important;
    background: rgba(45,212,191,0.07) !important;
    box-shadow: 0 4px 20px rgba(45,212,191,0.1) !important;
    transform: translateX(4px) !important;
    color: var(--text) !important;
}

/* Selected state */
.opt-selected-btn + div button,
.opt-selected-btn ~ div[data-testid="stButton"] button {
    border-color: rgba(45,212,191,0.5) !important;
    background: rgba(45,212,191,0.1) !important;
    box-shadow: 0 0 0 2px rgba(45,212,191,0.2), 0 4px 20px rgba(45,212,191,0.12) !important;
    color: var(--teal) !important;
}

/* Correct state */
.opt-correct-btn + div button,
.opt-correct-btn ~ div[data-testid="stButton"] button {
    border-color: rgba(16,185,129,0.55) !important;
    background: rgba(16,185,129,0.1) !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.2) !important;
    color: #4ade80 !important;
    cursor: default !important;
}

/* Wrong state */
.opt-wrong-btn + div button,
.opt-wrong-btn ~ div[data-testid="stButton"] button {
    border-color: rgba(251,113,133,0.5) !important;
    background: rgba(251,113,133,0.07) !important;
    color: var(--coral) !important;
    cursor: default !important;
}

/* Neutral (other options after check) */
.opt-neutral-btn + div button,
.opt-neutral-btn ~ div[data-testid="stButton"] button {
    opacity: 0.5 !important;
    cursor: default !important;
}

.opt-btn-wrap { margin: 0; padding: 0; height: 0; overflow: hidden; }

/* ── Hint card ──────────────────────────────────────────────────────────── */
.hcard {
    background: var(--glass);
    backdrop-filter: var(--blur2);
    -webkit-backdrop-filter: var(--blur2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--sky);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 1.1rem 1.5rem;
    margin: 0.6rem 0;
    font-size: 14.5px;
    line-height: 1.75;
    color: var(--text2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    animation: hintSlide 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes hintSlide {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
.hbadge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(103,232,249,0.1);
    border: 1px solid rgba(103,232,249,0.25);
    color: var(--sky);
    font-size: 10.5px;
    font-weight: 700;
    border-radius: 6px;
    padding: 3px 10px;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Syne', sans-serif;
}
.hcard-locked {
    background: rgba(6,20,16,0.4);
    border: 1px solid rgba(45,212,191,0.06);
    border-left: 3px solid rgba(45,212,191,0.1);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 1.1rem 1.5rem;
    margin: 0.6rem 0;
    filter: blur(2.5px);
    opacity: 0.3;
    user-select: none;
}

/* ── Article box ────────────────────────────────────────────────────────── */
.art-box {
    background: rgba(2,13,11,0.75);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem 1.8rem;
    font-size: 14px;
    line-height: 1.9;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text2);
    font-family: 'Syne', sans-serif;
}
.art-box::-webkit-scrollbar { width: 4px; }
.art-box::-webkit-scrollbar-track { background: transparent; }
.art-box::-webkit-scrollbar-thumb {
    background: var(--teal2);
    border-radius: 4px;
    opacity: 0.5;
}

/* ── Section label ──────────────────────────────────────────────────────── */
.slabel {
    font-family: 'Syne', sans-serif;
    font-size: 10.5px;
    font-weight: 700;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.slabel::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.04em;
}
.bg-teal    { background: rgba(45,212,191,0.12); color: var(--teal);   border: 1px solid rgba(45,212,191,0.3); }
.bg-emerald { background: rgba(16,185,129,0.12); color: #4ade80;       border: 1px solid rgba(16,185,129,0.3); }
.bg-coral   { background: rgba(251,113,133,0.12);color: var(--coral);  border: 1px solid rgba(251,113,133,0.3); }
.bg-amber   { background: rgba(245,158,11,0.12); color: var(--amber);  border: 1px solid rgba(245,158,11,0.3); }
.bg-sky     { background: rgba(103,232,249,0.12);color: var(--sky);    border: 1px solid rgba(103,232,249,0.3); }

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
}
.hr-glow {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.3), transparent);
    margin: 1.4rem 0;
}

/* ── Dataframe ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--r) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
.stSpinner > div { color: var(--teal) !important; }

/* ── Streamlit alerts ───────────────────────────────────────────────────── */
.stAlert {
    background: var(--glass) !important;
    backdrop-filter: var(--blur2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div { background: var(--teal) !important; }

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--glass) !important;
    backdrop-filter: var(--blur2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text2) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13.5px !important;
}

/* ── Loading overlay ────────────────────────────────────────────────────── */
.lov {
    position: fixed;
    inset: 0;
    background: rgba(2, 13, 11, 0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: lovIn 0.2s ease;
}
@keyframes lovIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.lring {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    border: 2px solid rgba(45,212,191,0.1);
    border-top-color: var(--teal);
    border-right-color: rgba(45,212,191,0.4);
    animation: lspin 0.85s linear infinite;
    box-shadow: 0 0 24px rgba(45,212,191,0.2);
}
.lring2 {
    position: absolute;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-bottom-color: rgba(103,232,249,0.5);
    animation: lspin 1.4s linear infinite reverse;
}
@keyframes lspin { to { transform: rotate(360deg); } }
.ltxt {
    margin-top: 1.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-align: center;
}
.ldots::after {
    content: '';
    animation: ldot 1.6s steps(4, end) infinite;
}
@keyframes ldot {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
}
.lpulse {
    position: relative;
    width: 52px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.lpulse::before {
    content: '';
    position: absolute;
    inset: -8px;
    border-radius: 50%;
    border: 1px solid rgba(45,212,191,0.15);
    animation: lpulseAnim 2s ease-in-out infinite;
}
@keyframes lpulseAnim {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.15); opacity: 0.1; }
}

/* ── Nav active state ───────────────────────────────────────────────────── */
.nav-active {
    background: rgba(45,212,191,0.1);
    border-left: 3px solid var(--teal);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 0.6rem 0.9rem;
    margin: 3px 0;
    font-size: 13.5px;
    font-weight: 700;
    color: var(--teal);
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.02em;
    cursor: default;
}

/* ── Sidebar logo ───────────────────────────────────────────────────────── */
.sidebar-logo {
    padding: 1.8rem 0.5rem 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.sidebar-logo .brand {
    font-family: 'Fraunces', serif;
    font-size: 1.85rem;
    font-weight: 900;
    color: var(--teal);
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.sidebar-logo .sub {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 5px;
}
.sidebar-logo .tagline {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 12.5px;
    color: rgba(45,212,191,0.5);
    margin-top: 2px;
}

/* ── Page header ────────────────────────────────────────────────────────── */
.ph-wrap {
    margin-bottom: 2rem;
}
.ph-eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 10.5px;
    font-weight: 700;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.35rem;
}
.ph-title {
    font-family: 'Fraunces', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin: 0;
}
.ph-title span { color: var(--teal); }
.ph-sub {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    color: var(--dim);
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* scrollbar global */
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(45,212,191,0.25) transparent;
}

/* ── Stray stApp z-index fix ────────────────────────────────────────────── */
.main { position: relative; z-index: 2; }
[data-testid="stSidebar"] { position: relative; z-index: 3; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading overlay helper
# ---------------------------------------------------------------------------

def _show_loading(message="Processing"):
    st.markdown(
        f"""
        <div class="lov">
            <div class="lpulse">
                <div class="lring"></div>
                <div class="lring2"></div>
            </div>
            <div class="ltxt">{message}<span class="ldots"></span></div>
            <div style="margin-top:0.5rem;font-family:'JetBrains Mono',monospace;
                        font-size:10px;color:rgba(45,212,191,0.25);letter-spacing:0.1em;">
                RACE QUIZ SYSTEM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        "screen": "input",
        # Content
        "article":              "",
        "question":             "",
        "options":              {},
        "gold_answer":          "",
        "correct_answer_text":  "",
        # Passage version — bump on every new passage load so all widget keys refresh
        "passage_version":      0,
        # Quiz state
        "selected_option":      None,
        "quiz_checked":         False,
        "predict_result":       None,
        "supporting_sent":      "",
        # Hints
        "hints":                [],
        "hints_revealed":       0,
        "answer_revealed":      False,
        # Distractors
        "distractors":          [],
        # Analytics log
        "log":                  [],
        "session_start":        datetime.datetime.now().isoformat(),
        # Loading
        "_loading":             False,
        "_loading_msg":         "Processing",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _reset_quiz_state():
    """Wipe ALL question/answer-dependent state — called on every new passage."""
    st.session_state.quiz_checked       = False
    st.session_state.selected_option    = None
    st.session_state.predict_result     = None
    st.session_state.supporting_sent    = ""
    st.session_state.hints              = []
    st.session_state.hints_revealed     = 0
    st.session_state.answer_revealed    = False
    st.session_state.distractors        = []
    st.session_state.question           = ""
    st.session_state.options            = {}
    st.session_state.gold_answer        = ""
    st.session_state.correct_answer_text= ""
    # Bump passage_version so every widget key tied to it becomes a new widget
    st.session_state.passage_version   += 1


# ---------------------------------------------------------------------------
# Analytics logger
# ---------------------------------------------------------------------------

def _log(event: str, latency_ms: float = 0.0, extra: dict = None):
    entry = {
        "timestamp":   datetime.datetime.now().strftime("%H:%M:%S"),
        "event":       event,
        "latency_ms":  round(latency_ms, 1),
    }
    if extra:
        entry.update(extra)
    st.session_state.log.append(entry)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="brand">RACE</div>
        <div class="tagline">Reading Comprehension</div>
        <div class="sub">AI Quiz System</div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "📄  Article Input":        "input",
        "❓  Quiz View":            "quiz",
        "💡  Hint Panel":           "hints",
        "📊  Analytics":            "dashboard",
    }

    for label, screen_id in pages.items():
        active = st.session_state.screen == screen_id
        if active:
            st.markdown(
                f'<div class="nav-active">{label}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{screen_id}", use_container_width=True):
                st.session_state.screen = screen_id
                st.rerun()

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # Pipeline status
    ready, pipeline, pipeline_err = _pipeline_ready()
    if ready:
        st.markdown('<span class="badge bg-emerald">● Pipeline Ready</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge bg-coral">● Pipeline Offline</span>',
                    unsafe_allow_html=True)
        if pipeline_err:
            with st.expander("Error details", expanded=False):
                st.caption(pipeline_err)

    st.markdown(f"""
    <div style='margin-top:1.4rem;font-family:"JetBrains Mono",monospace;
                font-size:10.5px;color:{('rgba(45,212,191,0.35)')};
                line-height:2;letter-spacing:0.04em;'>
        SESSION STARTED<br>
        <span style='color:rgba(45,212,191,0.6);'>
            {st.session_state.session_start[:19]}
        </span><br>
        INFERENCES &nbsp;
        <span style='color:rgba(45,212,191,0.6);'>{len(st.session_state.log)}</span>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# SCREEN 1 — Article Input
# ===========================================================================

if st.session_state.screen == "input":

    st.markdown("""
    <div class="ph-wrap">
        <div class="ph-eyebrow">Step 1</div>
        <h1 class="ph-title">Reading <span>Passage</span></h1>
        <p class="ph-sub">
            Paste any article or load a random sample from the RACE dataset —
            the pipeline will generate a question, distractors and graduated hints.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Random sample loader ────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 3])
    with col_l:
        split_choice = st.selectbox(
            "Dataset split",
            ["val", "train", "test"],
            label_visibility="collapsed",
            key="split_select",
        )
    with col_r:
        load_btn = st.button(
            "🎲  Load Random RACE Sample",
            use_container_width=True,
            key="load_sample_btn",
        )

    if load_btn:
        _show_loading("Loading passage from RACE dataset")
        try:
            from src.inference import random_race_sample
            sample = random_race_sample(split_choice)

            # ── CRITICAL: reset all quiz state BEFORE setting new content ──
            _reset_quiz_state()

            st.session_state.article              = sample["article"]
            st.session_state.question             = sample["question"]
            st.session_state.options              = sample["options"]
            st.session_state.gold_answer          = sample["answer"]
            st.session_state.correct_answer_text  = sample["options"][sample["answer"]]

            _log("load_sample", extra={"split": split_choice})
            st.rerun()

        except FileNotFoundError:
            st.error("RACE CSV not found. Place files in data/raw/ and reload.")
        except Exception as exc:
            st.error(f"Error loading sample: {exc}")

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Article text area ───────────────────────────────────────────────────
    st.markdown('<div class="slabel">📖 Reading Passage</div>', unsafe_allow_html=True)

    # Key tied to passage_version → widget is completely recreated on new passage
    article_input = st.text_area(
        "Paste your article here",
        value=st.session_state.article,
        height=230,
        placeholder="Paste a reading passage here, or load a random RACE sample above…",
        label_visibility="collapsed",
        key=f"article_area_{st.session_state.passage_version}",
    )
    # Only reset downstream if user manually edits
    if article_input != st.session_state.article:
        st.session_state.article = article_input
        _reset_quiz_state()
    else:
        st.session_state.article = article_input

    # ── Custom question override ────────────────────────────────────────────
    with st.expander(
        "✏️  Custom question  (optional — leave blank to auto-generate)",
        expanded=False,
    ):
        q_input = st.text_input(
            "Question",
            value=st.session_state.question,
            placeholder="What did the protagonist decide to do?",
            label_visibility="collapsed",
            key=f"q_input_{st.session_state.passage_version}",
        )
        st.session_state.question = q_input

        if q_input.strip():
            st.markdown("**Answer options** (leave blank to auto-generate)")
            c1, c2 = st.columns(2)
            with c1:
                a_in = st.text_input(
                    "A", value=st.session_state.options.get("A", ""),
                    key=f"opt_A_{st.session_state.passage_version}",
                )
                c_in = st.text_input(
                    "C", value=st.session_state.options.get("C", ""),
                    key=f"opt_C_{st.session_state.passage_version}",
                )
            with c2:
                b_in = st.text_input(
                    "B", value=st.session_state.options.get("B", ""),
                    key=f"opt_B_{st.session_state.passage_version}",
                )
                d_in = st.text_input(
                    "D", value=st.session_state.options.get("D", ""),
                    key=f"opt_D_{st.session_state.passage_version}",
                )
            if any([a_in, b_in, c_in, d_in]):
                st.session_state.options = {
                    "A": a_in, "B": b_in, "C": c_in, "D": d_in,
                }

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Submit ──────────────────────────────────────────────────────────────
    submit_col, _ = st.columns([1, 3])
    with submit_col:
        submit = st.button(
            "🚀  Generate Quiz",
            use_container_width=True,
            key="generate_quiz_btn",
        )

    if submit:
        if not st.session_state.article.strip():
            st.warning("Please paste an article or load a RACE sample first.")
        elif not ready:
            st.error(f"Pipeline not ready — train your models first.\n\n{pipeline_err or ''}")
        else:
            _show_loading("Generating quiz · running inference")
            with st.spinner("Running Model A & B inference…"):
                try:
                    from src.inference import (
                        predict_answer, generate_distractors, generate_hints
                    )
                    article  = st.session_state.article
                    question = st.session_state.question

                    if not question.strip():
                        from src.inference import _split_sentences, _clean, _tokenize
                        sentences = _split_sentences(article)
                        if sentences:
                            best = max(sentences, key=lambda s: len(_tokenize(s)))
                            question = f"What is the main idea described in: '{best[:80]}…'?"
                        else:
                            question = "What is the main idea of the passage?"
                        st.session_state.question = question

                    options = st.session_state.options
                    if not all(options.get(k, "").strip() for k in "ABCD"):
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
                        for k, v in options.items():
                            if v == correct_cand:
                                st.session_state.gold_answer = k
                                break
                        st.session_state.options    = options
                        st.session_state.distractors = dists
                    else:
                        correct_key  = st.session_state.gold_answer or "A"
                        correct_cand = options.get(correct_key, "")
                        t0 = time.perf_counter()
                        dists = generate_distractors(article, question, correct_cand)
                        dist_latency = (time.perf_counter() - t0) * 1000
                        st.session_state.distractors = dists

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

                    _log(
                        "generate_quiz",
                        latency_ms=dist_latency + hint_latency,
                        extra={"question_len": len(question)},
                    )

                    st.session_state.screen = "quiz"
                    st.rerun()

                except Exception as exc:
                    st.error(f"Inference error: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    # ── Passage preview ─────────────────────────────────────────────────────
    if st.session_state.article.strip():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">👁  Passage Preview</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="art-box">{st.session_state.article}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.question:
            st.markdown(
                f'<div class="gcard gcard-teal" style="margin-top:0.9rem;">'
                f'<div class="slabel">❓ Loaded Question</div>'
                f'<p style="font-size:1rem;margin:0;color:var(--text2);line-height:1.65;">'
                f'{st.session_state.question}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ===========================================================================
# SCREEN 2 — Quiz View
# ===========================================================================

elif st.session_state.screen == "quiz":

    if not st.session_state.article.strip():
        st.warning("No article loaded. Go to **Article Input** first.")
        if st.button("← Go to Article Input"):
            st.session_state.screen = "input"
            st.rerun()
        st.stop()

    st.markdown("""
    <div class="ph-wrap">
        <div class="ph-eyebrow">Step 2</div>
        <h1 class="ph-title">Answer the <span>Question</span></h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Article (collapsible) ───────────────────────────────────────────────
    with st.expander("📖  Reading Passage", expanded=False):
        st.markdown(
            f'<div class="art-box">{st.session_state.article}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Question card ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="gcard gcard-teal">'
        f'<div class="slabel">❓ Question</div>'
        f'<p style="font-size:1.12rem;line-height:1.7;margin:0;font-weight:500;'
        f'color:var(--text);">'
        f'{st.session_state.question}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Options ─────────────────────────────────────────────────────────────
    options = st.session_state.options
    if not options:
        st.error("No options available. Please go back to Article Input and generate a quiz.")
        if st.button("← Back to Article Input"):
            st.session_state.screen = "input"
            st.rerun()
        st.stop()

    st.markdown(
        '<div class="slabel" style="margin-top:1rem;">📋 Choose Your Answer</div>',
        unsafe_allow_html=True,
    )

    checked  = st.session_state.quiz_checked
    selected = st.session_state.selected_option
    gold     = st.session_state.gold_answer
    pv       = st.session_state.passage_version  # for unique widget keys

    for key in ["A", "B", "C", "D"]:
        opt_text = options.get(key, "")
        if not opt_text.strip():
            continue

        if checked:
            if key == gold:
                btn_cls  = "opt-correct-btn"
                prefix   = "✓"
            elif key == selected and key != gold:
                btn_cls  = "opt-wrong-btn"
                prefix   = "✗"
            else:
                btn_cls  = "opt-neutral-btn"
                prefix   = key
        else:
            btn_cls  = "opt-selected-btn" if key == selected else "opt-default-btn"
            prefix   = key

        # Inject per-button CSS class via a container ID trick
        st.markdown(
            f'<style>'
            f'div[data-testid="stButton"] > button[kind="secondary"]'
            f'#opt_{pv}_{key} {{ display:none; }}'
            f'</style>'
            f'<div class="opt-btn-wrap {btn_cls}" id="opt_wrap_{pv}_{key}"></div>',
            unsafe_allow_html=True,
        )

        col_btn, _ = st.columns([6, 1])
        with col_btn:
            if st.button(
                f"{prefix}  ·  {opt_text}",
                key=f"opt_btn_{pv}_{key}",
                use_container_width=True,
                disabled=checked,
            ):
                st.session_state.selected_option = key
                st.rerun()

    # Selection confirmation strip
    if selected and not checked:
        st.markdown(
            f'<div class="gcard" style="padding:0.75rem 1.2rem;margin-top:0.4rem;">'
            f'<span style="color:var(--dim);font-size:12.5px;font-family:\'Syne\',sans-serif;">'
            f'SELECTED &nbsp;·&nbsp; </span>'
            f'<strong style="color:var(--teal);font-family:\'JetBrains Mono\',monospace;">'
            f'{selected}</strong>'
            f'<span style="color:var(--text2);font-size:13px;"> — {options.get(selected,"")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Action row ────────────────────────────────────────────────────────────
    col_check, col_hint, col_reset = st.columns(3)

    with col_check:
        if not checked:
            check_clicked = st.button(
                "✔  Check Answer",
                use_container_width=True,
                disabled=(selected is None),
                key=f"check_btn_{pv}",
            )
            if check_clicked:
                ready2, pipeline2, _ = _pipeline_ready()
                if not ready2:
                    st.error("Pipeline not ready.")
                else:
                    _show_loading("Verifying answer")
                    with st.spinner("Verifying…"):
                        from src.inference import predict_answer
                        t0  = time.perf_counter()
                        res = predict_answer(
                            st.session_state.article,
                            st.session_state.question,
                            st.session_state.options,
                        )
                        lat = (time.perf_counter() - t0) * 1000

                    supp = ""
                    if ready2 and pipeline2:
                        supp = pipeline2.supporting_sentence(
                            st.session_state.article,
                            st.session_state.correct_answer_text
                            or options.get(gold or "A", ""),
                        )

                    st.session_state.predict_result  = res
                    st.session_state.quiz_checked    = True
                    st.session_state.supporting_sent = supp

                    is_correct = (selected == gold)
                    _log(
                        "check_answer",
                        latency_ms=lat,
                        extra={
                            "selected":        selected,
                            "gold":            gold,
                            "correct":         is_correct,
                            "model_predicted": res.get("predicted"),
                            "confidence":      res.get("confidence"),
                        },
                    )
                    st.rerun()

    with col_hint:
        if st.button("💡  Get a Hint", use_container_width=True,
                     key=f"hint_btn_{pv}"):
            st.session_state.screen = "hints"
            st.rerun()

    with col_reset:
        if st.button("🔄  New Question", use_container_width=True,
                     key=f"reset_btn_{pv}"):
            st.session_state.screen = "input"
            st.rerun()

    # ── Result display ────────────────────────────────────────────────────────
    if checked and st.session_state.predict_result:
        res        = st.session_state.predict_result
        is_correct = (selected == gold)
        supp       = st.session_state.supporting_sent

        st.markdown("<br>", unsafe_allow_html=True)

        if is_correct:
            st.markdown(
                f'<div class="gcard gcard-emerald">'
                f'<div style="font-family:\'Fraunces\',serif;font-size:1.5rem;'
                f'font-weight:700;color:#4ade80;letter-spacing:-0.02em;">'
                f'🎉 Correct!</div>'
                f'<div style="margin-top:0.5rem;display:flex;gap:1rem;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:12px;color:var(--dim);">'
                f'<span>CONFIDENCE &nbsp;<strong style="color:var(--mint);">'
                f'{res.get("confidence", 0):.0%}</strong></span>'
                f'<span>LATENCY &nbsp;<strong style="color:var(--mint);">'
                f'{res.get("latency_ms",0):.0f} ms</strong></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="gcard gcard-coral">'
                f'<div style="font-family:\'Fraunces\',serif;font-size:1.5rem;'
                f'font-weight:700;color:var(--coral);letter-spacing:-0.02em;">'
                f'✗ Incorrect</div>'
                f'<p style="margin:0.5rem 0 0;color:var(--text2);font-size:14px;">'
                f'The correct answer was '
                f'<strong style="color:var(--teal);font-family:\'JetBrains Mono\',monospace;">'
                f'{gold}</strong> — {options.get(gold,"")}</p>'
                f'<div style="margin-top:0.5rem;font-family:\'JetBrains Mono\',monospace;'
                f'font-size:11.5px;color:var(--dim);">'
                f'MODEL PREDICTED &nbsp;<strong style="color:var(--text2);">'
                f'{res.get("predicted")}</strong>'
                f'&nbsp;·&nbsp; CONFIDENCE &nbsp;<strong style="color:var(--text2);">'
                f'{res.get("confidence",0):.0%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if supp:
            st.markdown(
                f'<div class="gcard gcard-sky">'
                f'<div class="slabel">📌 Supporting Evidence</div>'
                f'<p style="font-size:13.5px;line-height:1.75;margin:0;'
                f'font-style:italic;color:var(--text2);">"{supp}"</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Score bar chart
        scores = res.get("scores", {})
        if scores:
            bar_colors = []
            for k in scores:
                if k == gold:
                    bar_colors.append("#10b981")
                elif k == selected and k != gold:
                    bar_colors.append("#fb7185")
                else:
                    bar_colors.append("rgba(45,212,191,0.25)")

            fig = go.Figure(go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=bar_colors,
                text=[f"{v:.0%}" for v in scores.values()],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12, color="#5eead4"),
                marker_line_width=0,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#5eead4",
                font_family="Syne",
                margin=dict(t=24, b=8, l=8, r=8),
                height=210,
                yaxis=dict(
                    range=[0, 1.2],
                    gridcolor="rgba(45,212,191,0.08)",
                    tickformat=".0%",
                    tickfont=dict(size=11, family="JetBrains Mono"),
                    zeroline=False,
                ),
                xaxis=dict(
                    tickfont=dict(size=15, family="JetBrains Mono", color="#99f6e4"),
                    ticklabelposition="outside",
                ),
                showlegend=False,
                bargap=0.38,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown(
            '<p style="font-size:11px;color:var(--dim);margin-top:0.4rem;">'
            '⚠ Answers and distractors are AI-generated. '
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
    <div class="ph-wrap">
        <div class="ph-eyebrow">Step 3</div>
        <h1 class="ph-title">Hint <span>Panel</span></h1>
        <p class="ph-sub">
            Hints progress from broad context clues to near-explicit guidance.
            Use them wisely before revealing the answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Question recap
    st.markdown(
        f'<div class="gcard gcard-teal">'
        f'<div class="slabel">❓ Question</div>'
        f'<p style="font-size:1.05rem;margin:0;font-weight:500;'
        f'color:var(--text);line-height:1.65;">'
        f'{st.session_state.question}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    hints = st.session_state.hints
    if not hints:
        st.info("No hints generated yet. Submit an article first.")
    else:
        hint_meta = [
            ("Hint 1",  "General Context",    "#67e8f9", "rgba(103,232,249,0.14)"),
            ("Hint 2",  "Narrowed Focus",     "#2dd4bf", "rgba(45,212,191,0.14)"),
            ("Hint 3",  "Near-Explicit Clue", "#f59e0b", "rgba(245,158,11,0.14)"),
        ]
        n_revealed = st.session_state.hints_revealed

        for i, hint in enumerate(hints[:3]):
            if i < len(hint_meta):
                num, desc, border_col, badge_bg = hint_meta[i]
            else:
                num, desc, border_col, badge_bg = f"Hint {i+1}", "", "#5eead4", "rgba(94,234,212,0.1)"

            if i < n_revealed:
                st.markdown(
                    f'<div class="hcard" style="border-left-color:{border_col};">'
                    f'<div class="hbadge" style="background:{badge_bg};'
                    f'color:{border_col};border-color:{border_col}55;">'
                    f'{num} &mdash; {desc}</div><br>'
                    f'<span style="color:var(--text2);">{hint}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="hcard-locked">'
                    f'<div style="background:rgba(45,212,191,0.06);color:var(--dim);'
                    f'font-size:10px;font-weight:700;border-radius:5px;padding:3px 10px;'
                    f'display:inline-block;text-transform:uppercase;letter-spacing:0.12em;'
                    f'margin-bottom:0.5rem;">{num} &mdash; {desc}</div><br>'
                    f'<span style="color:var(--dim);">{"▓" * 55}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_reveal, col_back = st.columns(2)

        with col_reveal:
            if n_revealed < len(hints):
                next_num  = hint_meta[n_revealed][0] if n_revealed < len(hint_meta) else f"Hint {n_revealed+1}"
                next_desc = hint_meta[n_revealed][1] if n_revealed < len(hint_meta) else ""
                if st.button(f"💡  Reveal {next_num}", use_container_width=True):
                    st.session_state.hints_revealed += 1
                    _log("reveal_hint",
                         extra={"hint_number": st.session_state.hints_revealed})
                    st.rerun()
            else:
                st.markdown(
                    '<span class="badge bg-emerald">✓ All hints revealed</span>',
                    unsafe_allow_html=True,
                )

        with col_back:
            if st.button("← Back to Quiz", use_container_width=True):
                st.session_state.screen = "quiz"
                st.rerun()

        # Reveal answer — only after all hints used
        if n_revealed >= len(hints) and not st.session_state.answer_revealed:
            st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:13px;color:var(--dim);">'
                'All hints have been used. You may now reveal the correct answer.</p>',
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
                f'<div class="gcard gcard-emerald" style="margin-top:1rem;">'
                f'<div style="font-family:\'Fraunces\',serif;font-size:1.3rem;'
                f'font-weight:700;color:#4ade80;letter-spacing:-0.02em;">'
                f'✅ Correct Answer: '
                f'<span style="font-family:\'JetBrains Mono\',monospace;">{gold}</span>'
                f'</div>'
                f'<p style="margin:0.55rem 0 0;color:var(--text2);font-size:14px;">'
                f'{correct_text}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ===========================================================================
# SCREEN 4 — Analytics Dashboard
# ===========================================================================

elif st.session_state.screen == "dashboard":

    st.markdown("""
    <div class="ph-wrap">
        <div class="ph-eyebrow">Developer View</div>
        <h1 class="ph-title">Analytics <span>Dashboard</span></h1>
        <p class="ph-sub">
            Live performance metrics for Model A (answer verification) and
            Model B (distractor &amp; hint generation) across this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    log = st.session_state.log

    # ── Session summary metrics ───────────────────────────────────────────────
    check_events = [e for e in log if e["event"] == "check_answer"]
    n_total   = len(check_events)
    n_correct = sum(1 for e in check_events if e.get("correct"))
    accuracy  = n_correct / n_total if n_total else 0.0
    avg_lat   = np.mean([e["latency_ms"] for e in check_events]) if check_events else 0.0
    n_hints   = len([e for e in log if e["event"] == "reveal_hint"])

    col1, col2, col3, col4 = st.columns(4)
    for col, val, lbl in [
        (col1, n_total,             "Quizzes Taken"),
        (col2, f"{accuracy:.0%}",  "Session Accuracy"),
        (col3, f"{avg_lat:.0f}ms", "Avg Latency"),
        (col4, n_hints,             "Hints Used"),
    ]:
        with col:
            st.markdown(
                f'<div class="mbox"><div class="mv">{val}</div>'
                f'<div class="ml">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="hr-glow" style="margin-top:1.4rem;"></div>',
                unsafe_allow_html=True)

    # ── Model A ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slabel" style="font-size:11px;">🤖 Model A — Answer Verification</div>',
        unsafe_allow_html=True,
    )

    ready, pipeline, _ = _pipeline_ready()
    if ready and pipeline:
        model_map = {
            "Logistic Regression": pipeline.lr_classifier,
            "SVM":                 pipeline.svm_classifier,
            "Naive Bayes":         pipeline.nb_classifier,
            "Random Forest":       pipeline.rf_classifier,
            "XGBoost":             pipeline.xgb_classifier,
            "Ensemble":            pipeline.ensemble_model,
        }
        model_df = pd.DataFrame({
            "Model":  list(model_map.keys()),
            "Status": ["✅ Loaded" if v is not None else "⬜ Not trained"
                       for v in model_map.values()],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    if check_events:
        tp = sum(1 for e in check_events
                 if e.get("correct") and e.get("selected") == e.get("gold"))
        fp = sum(1 for e in check_events
                 if not e.get("correct") and e.get("model_predicted") != e.get("gold"))
        fn = sum(1 for e in check_events
                 if e.get("correct") and e.get("model_predicted") != e.get("selected"))
        tn = max(n_total - tp - fp - fn, 0)

        cm = np.array([[tp, fn], [fp, tn]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Correct", "Predicted Wrong"],
            y=["Actually Correct",  "Actually Wrong"],
            colorscale=[
                [0,   "rgba(13,40,32,0.7)"],
                [0.5, "rgba(20,184,166,0.3)"],
                [1,   "#2dd4bf"],
            ],
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont=dict(size=18, family="JetBrains Mono"),
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#5eead4",
            font_family="Syne",
            margin=dict(t=24, b=24, l=12, r=12),
            height=240,
            title=dict(text="Session Confusion Matrix", font=dict(size=13)),
        )
        st.plotly_chart(fig_cm, use_container_width=True,
                        config={"displayModeBar": False})

        confs = [e.get("confidence", 0) for e in check_events]
        if confs:
            fig_conf = go.Figure(go.Histogram(
                x=confs,
                nbinsx=10,
                marker_color="#2dd4bf",
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                opacity=0.75,
            ))
            fig_conf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#5eead4",
                font_family="Syne",
                xaxis=dict(
                    title="Confidence",
                    gridcolor="rgba(45,212,191,0.08)",
                    tickformat=".0%",
                ),
                yaxis=dict(title="Count", gridcolor="rgba(45,212,191,0.08)"),
                margin=dict(t=24, b=16, l=12, r=12),
                height=210,
                title=dict(text="Confidence Distribution", font=dict(size=13)),
            )
            st.plotly_chart(fig_conf, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info("No quiz events logged yet this session.")

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Model B ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slabel" style="font-size:11px;">🎯 Model B — Distractor & Hint Generation</div>',
        unsafe_allow_html=True,
    )

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
            marker_color=["#67e8f9", "#2dd4bf", "#f59e0b"][:len(hc)],
            text=[str(hc[k]) for k in sorted(hc)],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#5eead4"),
            marker_line_width=0,
        ))
        fig_hints.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#5eead4",
            font_family="Syne",
            margin=dict(t=24, b=16, l=12, r=12),
            height=210,
            yaxis=dict(gridcolor="rgba(45,212,191,0.08)"),
            title=dict(text="Hints Requested per Level", font=dict(size=13)),
            bargap=0.4,
        )
        st.plotly_chart(fig_hints, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Latency chart ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slabel" style="font-size:11px;">⏱ Inference Latency (ms)</div>',
        unsafe_allow_html=True,
    )

    latency_events = [e for e in log if e["latency_ms"] > 0]
    if latency_events:
        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=list(range(1, len(latency_events) + 1)),
            y=[e["latency_ms"] for e in latency_events],
            mode="lines+markers",
            line=dict(color="#2dd4bf", width=2),
            marker=dict(
                size=7,
                color="#2dd4bf",
                line=dict(color="#020d0b", width=2),
            ),
            fill="tozeroy",
            fillcolor="rgba(45,212,191,0.05)",
            text=[e["event"] for e in latency_events],
            hovertemplate="%{text}<br>%{y:.0f} ms<extra></extra>",
        ))
        fig_lat.add_hline(
            y=10_000,
            line_dash="dot",
            line_color="#fb7185",
            annotation_text="10 s limit",
            annotation_position="right",
            annotation_font_color="#fb7185",
        )
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#5eead4",
            font_family="Syne",
            xaxis=dict(title="Request #", gridcolor="rgba(45,212,191,0.08)"),
            yaxis=dict(title="Latency (ms)", gridcolor="rgba(45,212,191,0.08)"),
            margin=dict(t=12, b=12, l=12, r=12),
            height=230,
        )
        st.plotly_chart(fig_lat, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown('<div class="hr-glow"></div>', unsafe_allow_html=True)

    # ── Session log ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="slabel" style="font-size:11px;">📋 Session Event Log</div>',
        unsafe_allow_html=True,
    )

    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        col_exp, col_clr = st.columns([2, 1])
        with col_exp:
            csv_data = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Export Session Log (CSV)",
                data=csv_data,
                file_name=f"race_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑  Clear Log", use_container_width=True):
                st.session_state.log = []
                st.rerun()
    else:
        st.info("No events logged yet. Take a quiz to see data here.")

    st.markdown(
        '<p style="font-size:11px;color:var(--dim);margin-top:1.2rem;">'
        '⚠ All questions, answers, distractors, and hints are AI-generated. '
        'Results should be reviewed by a human expert before use in any '
        'formal educational setting.</p>',
        unsafe_allow_html=True,
    )