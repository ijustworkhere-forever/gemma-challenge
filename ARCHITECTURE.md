# 🧠 Gemma Challenge – Lightweight ML Research Backend

This project is a **single-node, queue-based ML research system** designed to run inference experiments using free LLM APIs (OpenRouter, NVIDIA NIM, HuggingFace).

It is optimized for:
- low-cost VPS deployment
- free-tier LLM APIs
- small-scale research workloads

---

# 🏗 System Design

```
Controller → Redis Queue → Worker → LLM API → SQLite DB
```

---

# ⚙️ Components

## 1. Controller
Generates experiments and submits them to Redis.

## 2. Redis Queue
Handles asynchronous job distribution.

## 3. Worker
Executes inference jobs:
- OpenRouter
- NVIDIA NIM
- HuggingFace

## 4. SQLite Database
Stores:
- experiment metadata
- results
- latency
- cost estimates

---

# 📊 Metrics Tracked

- latency per request
- token usage
- estimated cost
- success/failure rate

---

# 💰 Cost Model

Designed for:
- free-tier APIs
- minimal VPS usage
- no GPU required

---

# 🚀 Deployment

## Requirements
- Ubuntu VPS
- Redis
- Python 3.10+

## Install
```bash
sudo apt install redis python3-pip
pip install redis requests sqlite3
```

## Run
```bash
python controller.py
python worker.py
```

---

# 🧠 Philosophy

This system prioritizes:

> simplicity, observability, and real execution over distributed complexity

It is designed to evolve into a full MLOps system only when resources allow.
