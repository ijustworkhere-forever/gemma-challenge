"""
research_autopilot_dashboard.py

Live dashboard for autonomous inference optimization system.

Purpose:
- Visualize benchmark performance trends
- Show experiment history
- Display bottlenecks
- Show next recommended experiments
- Trigger runs manually (optional hook)

Built with Streamlit for simplicity and speed.
"""

import os
import json
import glob
import streamlit as st
import pandas as pd


# -----------------------------
# LOADERS
# -----------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_experiments(dir_path="auto_experiments"):
    files = sorted(glob.glob(os.path.join(dir_path, "*.json")))

    data = []
    for f in files:
        d = load_json(f)
        if d:
            data.append(d)

    return data


def load_meta_policy():
    return load_json("META_POLICY.json")


def load_next_experiment():
    return load_json("compiled_experiments.json")


def load_ideas():
    if not os.path.exists("IDEAS.md"):
        return ""
    with open("IDEAS.md", "r") as f:
        return f.read()


# -----------------------------
# ANALYTICS
# -----------------------------

def build_dataframe(experiments):
    rows = []

    for exp in experiments:
        bench = exp.get("benchmark", {})
        rows.append({
            "model": exp.get("model"),
            "tps_mean": bench.get("tps_mean", 0),
            "latency": bench.get("latency_mean", 0),
            "timestamp": exp.get("timestamp")
        })

    return pd.DataFrame(rows)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Autonomous Optimizer", layout="wide")

st.title("🧠 Autonomous Inference Optimization Dashboard")


# -----------------------------
# LOAD DATA
# -----------------------------

experiments = load_experiments()
meta = load_meta_policy()
next_exp = load_next_experiment()
ideas = load_ideas()

df = build_dataframe(experiments)


# -----------------------------
# TOP METRICS
# -----------------------------

st.header("📊 System Performance")

if not df.empty:
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg TPS", f"{df['tps_mean'].mean():.2f}")
    col2.metric("Best TPS", f"{df['tps_mean'].max():.2f}")
    col3.metric("Avg Latency", f"{df['latency'].mean():.4f}s")
else:
    st.warning("No experiment data found.")


# -----------------------------
# TPS CHART
# -----------------------------

st.subheader("TPS Over Experiments")

if not df.empty:
    st.line_chart(df.set_index("timestamp")["tps_mean"])


# -----------------------------
# EXPERIMENT HISTORY
# -----------------------------

st.subheader("📁 Experiment History")

if not df.empty:
    st.dataframe(df)
else:
    st.info("No experiments logged yet.")


# -----------------------------
# META POLICY
# -----------------------------

st.subheader("🧬 Meta Optimization Policy")

if meta:
    st.json(meta)
else:
    st.info("Run meta_optimizer.py first.")


# -----------------------------
# NEXT EXPERIMENTS
# -----------------------------

st.subheader("🚀 Next Suggested Experiments")

if next_exp:
    st.json(next_exp)
else:
    st.info("Run experiment_compiler.py first.")


# -----------------------------
# IDEAS
# -----------------------------

st.subheader("💡 Live IDEAS.md Feed")

st.text_area("Current IDEAS.md", ideas, height=300)


# -----------------------------
# BOTTLENECK INSIGHT PANEL
# -----------------------------

st.subheader("⚠️ Bottleneck Insight (Heuristic)")

if not df.empty:
    latest = df.iloc[-1]

    tps = latest["tps_mean"]

    if tps < 20:
        st.error("GPU Underutilized / Scheduler bottleneck likely")
    elif tps < 50:
        st.warning("Memory / Attention bottleneck likely")
    else:
        st.success("System is compute-efficient or balanced")


# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.caption("Autonomous Optimization System v1 — live research loop controller")
