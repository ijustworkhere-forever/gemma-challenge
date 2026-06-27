# 🧠 Gemma Challenge — Autonomous ML Research Backend

A distributed, queue-based ML research system for running, evaluating, and evolving LLM inference experiments across multiple providers (OpenRouter, HuggingFace, Vertex AI).

---

# 🚀 Overview

This system is a lightweight **ML research backend** designed to:

- Generate inference experiments automatically
- Execute experiments via distributed workers
- Benchmark LLM performance across providers
- Track cost, latency, and throughput
- Evolve experiment strategies over time

It is inspired by:
- internal ML evaluation systems
- distributed inference platforms
- research automation pipelines

---

# 🏗 Architecture

## System Flow

```
Controller → Redis Queue → Workers → Providers → Result Queue → Evaluator → Meta Optimizer
```

---

## Core Components

### 1. Controller
Generates experiments and submits jobs to the queue.

### 2. Redis Queue
Provides durable, retry-safe job distribution.

Queues:
- `experiment_queue`
- `result_queue`

---

### 3. Workers
Stateless execution nodes that:

- Pull jobs from queue
- Run inference via:
  - OpenRouter
  - HuggingFace
  - Vertex AI
- Return normalized results

---

### 4. Cost Tracker
Tracks:

- token usage
- provider cost per request
- total experiment cost

---

### 5. Autoscaler
Simple queue-depth-based scaling:

- scales workers up when backlog increases
- remains idle when system is stable

---

### 6. Meta Optimizer (future layer)
Learns from historical results to:

- improve experiment selection
- prune low-value strategies
- optimize system performance

---

# ⚙️ Deployment (VPS)

## Requirements

- Ubuntu VPS
- Python 3.10+
- Redis server

## Install

```bash
sudo apt update
sudo apt install redis python3-pip -y
pip install redis requests
```

## Run system

```bash
python controller.py
python worker.py
python autoscaler.py
```

---

# 📊 Features

## ✔ Distributed Execution
Workers operate independently and scale horizontally.

## ✔ Multi-Provider Benchmarking
Supports:
- OpenRouter
- HuggingFace
- Vertex AI

## ✔ Fault Tolerance
- Retry mechanism (max 3 attempts)
- Queue persistence

## ✔ Cost Awareness
Every experiment tracks estimated API cost.

## ✔ Scalable Architecture
Queue-based system allows multiple workers per VPS.

---

# 🔁 Future Extensions

Planned upgrades:

- Postgres / ClickHouse result storage
- Web dashboard (Streamlit or Next.js)
- Multi-node worker clusters
- Kubernetes deployment option
- Advanced experiment RL selection policy

---

# 🧠 Concept

This system behaves like a:

> lightweight internal ML research platform for LLM inference optimization

---

# ⚠️ Notes

- Not a production-grade distributed system yet
- Designed for experimentation + research workflows
- Optimized for simplicity on a single VPS
