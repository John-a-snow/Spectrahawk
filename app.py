from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from spectrahawk.simulator import SimConfig, simulate, metrics

st.set_page_config(page_title="SpectraHawk", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 26px 30px; border: 1px solid #26364a; border-radius: 18px; background: #0d1724;}
.kicker {color:#75a8d6; font-size:13px; letter-spacing:2px; text-transform:uppercase; font-weight:700;}
.hero h1 {font-size:48px; margin:6px 0; color:#f4f7fb;}
.hero p {color:#aebdcd; font-size:17px; max-width:850px;}
.card {border:1px solid #26364a; border-radius:14px; padding:16px; background:#101b29;}
.small {color:#94a7ba; font-size:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="kicker">Smart India Hackathon 2026 · Que Sera Sera</div>
<h1>◈ SpectraHawk</h1>
<p>Simulation-first adaptive spectrum scanning: Observe → Predict → Decide → Learn.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Scenario")
    preset = st.selectbox("Preset", ["Easy", "Medium", "Hard", "Stress", "OOD"], index=1)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=7, step=1)
    slots = st.slider("Time slots", 100, 2000, 500, 100)
    bands = st.select_slider("Bands", options=[16, 32, 64, 128], value=32)
    agent = st.selectbox("Scheduler", ["SpectraHawk-TS", "LinearSweep", "Random"])
    run = st.button("Run simulation", type="primary", use_container_width=True)

if run or "episode" not in st.session_state:
    cfg = SimConfig(slots=int(slots), bands=int(bands), seed=int(seed))
    st.session_state.episode = simulate(cfg, agent=agent)
    st.session_state.agent = agent

agent = st.session_state.agent
ep = st.session_state.episode
m = metrics(ep)

cols = st.columns(6)
for col, (label, value) in zip(cols, [
    ("Pd", f"{m['Pd']:.1%}"),
    ("Pfa", f"{m['Pfa']:.3%}"),
    ("Intercept rate", f"{m['Intercept rate / 1000']:.1f}"),
    ("Reward / slot", f"{m['Reward / slot']:.3f}"),
    ("Cumulative reward", f"{m['Cumulative reward']:.1f}"),
    ("Band coverage", f"{m['Band coverage']:.1%}"),
]):
    with col:
        st.markdown(f'<div class="card"><div class="small">{label}</div><h2>{value}</h2></div>', unsafe_allow_html=True)

st.subheader("Live Duel — receiver trajectory")
window = st.slider("Timeline", 50, len(ep.selected), min(250, len(ep.selected)), 50)
truth = ep.truth[:window]
visits = ep.selected[:window]

z = truth.astype(float)
fig = go.Figure(go.Heatmap(z=z.T, colorscale=[[0, "#101b29"], [1, "#4f86b8"]], showscale=False))
fig.add_trace(go.Scatter(x=np.arange(window), y=visits, mode="markers", marker=dict(size=4, color="#f3c969"), name="Receiver dwell"))
hits = np.flatnonzero(ep.detected[:window])
fig.add_trace(go.Scatter(x=hits, y=visits[hits], mode="markers", marker=dict(size=7, color="#62c58c", symbol="circle-open"), name="Detection"))
fig.update_layout(height=540, xaxis_title="Time slot", yaxis_title="Band", paper_bgcolor="#0d1724", plot_bgcolor="#0d1724", font_color="#d9e4ef", margin=dict(l=50,r=20,t=20,b=50))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Reward over time")
reward_df = pd.DataFrame({"slot": np.arange(len(ep.rewards)), "reward": ep.rewards, "cumulative": np.cumsum(ep.rewards)})
reward_fig = go.Figure()
reward_fig.add_trace(go.Scatter(x=reward_df.slot, y=reward_df.cumulative, name="Cumulative reward"))
reward_fig.update_layout(height=300, paper_bgcolor="#0d1724", plot_bgcolor="#0d1724", font_color="#d9e4ef")
st.plotly_chart(reward_fig, use_container_width=True)

st.subheader("Method")
st.markdown("""
**SpectraHawk-TS** in this prototype uses a deterministic closed-loop band-selection policy with posterior sampling and a staleness exploration floor. The RF world is simulated; no real RF hardware, SDR, live spectrum, or operational interception is used.

The production roadmap adds the LSTM prior, Period-Lock/Kalman tracker, Double-DQN, formal statistical benchmarking, calibration, ablations, OOD evaluation, and the full six-page dashboard described in the project specification.
""")

st.download_button("Download selected-band trace", reward_df.to_csv(index=False), "spectrahawk_trace.csv", "text/csv")
