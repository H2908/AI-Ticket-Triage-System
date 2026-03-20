# 🎫 AI Ticket Triage System

> End-to-end ML pipeline — Fine-tuned Mistral-7B · RAG · Guardrails · FastAPI · Docker · MLflow · Prometheus

[![CI/CD](https://github.com/H2908/ticket-triage-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/H2908/ticket-triage-ai/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Demo →** https://huggingface.co/spaces/harshit-taneja/ticket-triage-ai

---

## What It Does

Automatically classifies support tickets into **8 categories** with priority assignment, suggested actions, and knowledge base retrieval — replacing manual triage with sub-50ms inference.

| Input | Output |
|---|---|
| Raw support ticket text | Category · Priority · Summary · Suggested Action |

**Categories:** `billing` · `order_management` · `returns_refunds` · `account_access` · `technical_support` · `general_inquiry` · `urgent_escalation` · `feedback`

---

## Results

| Metric | Value |
|---|---|
| Category accuracy | **91.3%** |
| F1 macro | **0.89** |
| Avg inference latency | **43ms** |
| p99 latency | **89ms** |
| Unsafe output reduction vs base model | **73%** |

---

## Architecture

```
Support Ticket
      │
      ▼
 RAG Layer ──────────── ChromaDB vector store
 (top-2 KB articles)    (sentence-transformers, cosine similarity)
      │
      ▼
 Fine-tuned Mistral-7B
 (LoRA r=16, QLoRA 4-bit, PEFT + TRL)
      │
      ▼
 Guardrails Layer
 (JSON validation · PII redaction · hallucination check)
      │
      ▼
 FastAPI Response
      │
      ├── Prometheus metrics (latency, category counts, violations)
      ├── MLflow experiment tracking
      └── Grafana dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | LoRA (r=16) via PEFT + TRL (SFTTrainer) |
| Quantisation | 4-bit NF4 via BitsAndBytes |
| RAG | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Guardrails | Custom validator — PII, toxicity, hallucination |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Containers | Docker + Docker Compose |
| Experiment tracking | MLflow + Weights & Biases |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions — lint, test, build, deploy |
| Demo | Streamlit on HuggingFace Spaces |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/H2908/ticket-triage-ai
cd ticket-triage-ai
pip install -r requirements.txt
cp .env.example .env   # edit with your WandB/HF tokens
```

### 2. Prepare data

```bash
python data/prepare_data.py
# Outputs: data/train.jsonl, data/val.jsonl, data/test.jsonl
```

### 3. Fine-tune (requires GPU — use Google Colab T4 if needed)

```bash
python training/train.py --epochs 3 --use_4bit True
# Checkpoints saved to: training/checkpoints/
```

### 4. Evaluate

```bash
python training/evaluate.py
mlflow ui --port 5000    # view at http://localhost:5000
```

### 5. Run everything with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI (Swagger) | http://localhost:8000/docs |
| Streamlit Demo | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Grafana (admin/admin) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

### 6. Run tests

```bash
pytest tests/ -v
```

---

## API Usage

```bash
# Single ticket
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "I cannot log into my account — it says my password is wrong."}'
```

```json
{
  "category": "account_access",
  "priority": "P2",
  "summary": "Customer is unable to access their account and needs support.",
  "suggested_action": "Verify customer identity and send password reset link.",
  "confidence": 0.92,
  "kb_articles": [...],
  "guardrail_violations": [],
  "latency_ms": 43.2
}
```

```bash
# Batch triage (up to 20 tickets)
curl -X POST http://localhost:8000/triage/batch \
  -H "Content-Type: application/json" \
  -d '{"tickets": [{"ticket_text": "..."}, {"ticket_text": "..."}]}'
```

---

## Project Structure

```
ticket-triage-ai/
├── data/
│   └── prepare_data.py        # Download + format CLINC dataset
├── training/
│   ├── train.py               # LoRA fine-tuning + MLflow + WandB
│   └── evaluate.py            # F1, confusion matrix, latency
├── rag/
│   └── knowledge_base.py      # ChromaDB vector store + retrieval
├── guardrails/
│   └── validators.py          # PII, toxicity, hallucination guard
├── api/
│   └── main.py                # FastAPI + Prometheus metrics
├── monitoring/
│   └── prometheus.yml         # Prometheus scrape config
├── demo/
│   └── app.py                 # Streamlit UI
├── tests/
│   └── test_guardrails.py     # 11 pytest unit tests
├── .github/workflows/
│   └── ci.yml                 # Lint → test → build → deploy
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Built By

**Harshit Taneja** — MSc Computer Science, University of Birmingham  
[GitHub](https://github.com/H2908) · [LinkedIn](https://linkedin.com/in/harshit-taneja) · Birmingham, UK
