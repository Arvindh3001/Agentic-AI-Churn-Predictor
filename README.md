# Churn Intelligence Platform

An end-to-end **Agentic AI** platform for customer churn prediction and retention management. A 6-agent LangGraph pipeline predicts churn, explains it with SHAP, generates counterfactuals, optimises a retention budget, and routes high-risk cases through a human-in-the-loop approval gate — all surfaced in a real-time Next.js dashboard.

---

## What It Does

| Capability | Detail |
|---|---|
| **Churn Prediction** | XGBoost ensemble with conformal 95% confidence intervals |
| **Explainability** | SHAP local + global, LIME, DiCE counterfactuals |
| **Retention Strategy** | Knapsack-optimised action selection within a cost budget |
| **HITL Gate** | Slack Block Kit approval flow for CRITICAL-tier customers |
| **Feedback Loop** | CSM outcome recording → retraining trigger at 50 records |
| **Fairness & Robustness** | Demographic parity, equalized odds, adversarial stability |
| **Survival Analysis** | Kaplan-Meier curves, Cox PH hazard ratios, cohort heatmap |
| **Real-time UI** | WebSocket step streaming, live customer broadcast, Quick Insights sidebar |

---

## Architecture

```
Next.js 14 Dashboard
        │  REST + WebSocket
        ▼
FastAPI Backend  ──►  Celery Worker
        │                   │
        ▼                   ▼
  PostgreSQL          LangGraph Pipeline
  Redis               ├── DataIntelligenceAgent
  ChromaDB            ├── PredictionAgent  (XGBoost + SHAP)
  MLflow              ├── ExplanationAgent (LLM narrative)
                      ├── CounterfactualAgent (DiCE)
                      ├── RetentionStrategistAgent (Knapsack)
                      └── HITLAgent (Slack approval)
```

---

## Quick Start (Docker)

```bash
cd docker
docker compose -f docker-compose.yml up -d
```

| Service | URL | Credentials |
|---|---|---|
| Dashboard | http://localhost:3000 | admin / admin123 |
| API Docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3001 | admin / admin123 |
| MLflow | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |

---

## Local Development

```powershell
# 1. Infrastructure only
docker compose -f docker/docker-compose.yml up -d postgres redis chromadb mlflow

# 2. Python setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 3. Generate data & train models (first time only)
python data/synthetic/generate_synthetic.py
python src/models/train.py

# 4. Start API  (Terminal 2)
python -m uvicorn app.main:app --reload --port 8000

# 5. Start Celery worker  (Terminal 3)
python -m celery -A celery_app.celery_app worker --loglevel=info

# 6. Start frontend  (Terminal 4)
cd frontend && npm install && npm run dev
```

> **Windows note:** Use `python -m celery` — the `celery.exe` in `.venv\Scripts\` may be blocked by Windows security policy on OneDrive-synced paths.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Optional | GPT-4o narrative generation |
| `ANTHROPIC_API_KEY` | Optional | Claude narrative generation |
| `SLACK_WEBHOOK_URL` | Optional | HITL Slack notifications |
| `SLACK_SIGNING_SECRET` | Optional | Slack webhook signature verification |
| `DATABASE_URL` | Auto-set in Docker | PostgreSQL connection string |
| `REDIS_URL` | Auto-set in Docker | Redis connection string |

The pipeline runs fully in **template mode** without any LLM API keys — narratives are generated from structured templates instead of an LLM.

---

## Project Structure

```
churn-intelligence-platform/
├── agents/               # LangGraph agents + tools
│   ├── orchestrator.py   # StateGraph wiring + run_sync / run_async
│   ├── state.py          # Shared AgentState TypedDict
│   ├── *_agent.py        # Six pipeline agents
│   └── tools/            # sql, model, shap, counterfactual, crm, slack
├── app/                  # FastAPI backend
│   ├── main.py           # App factory, CORS, lifespan
│   ├── routers/          # auth, customers, agent, analytics, gdpr
│   ├── websockets/       # agent_stream, customer_broadcast
│   └── hitl_webhook.py   # HITL decision + feedback endpoints
├── frontend/             # Next.js 14 dashboard
│   └── src/app/          # login, dashboard, customers/[id], hitl, models, system
├── src/                  # ML pipeline modules
│   ├── models/           # Training, registry, conformal prediction
│   ├── explainability/   # SHAP, LIME, DiCE, narrative generator
│   ├── optimization/     # Knapsack solver, A/B testing
│   ├── fairness/         # Bias detection, fairness report
│   ├── robustness/       # Adversarial + calibration testing
│   └── temporal/         # Survival analysis, cohort, seasonality
├── memory/               # Redis state manager + ChromaDB vector store
├── config/               # Dynaconf settings
├── docker/               # Dockerfiles + docker-compose
├── k8s/                  # Kubernetes manifests (10 resources)
├── monitoring/           # Prometheus config + Grafana dashboard
└── .github/workflows/    # CI, deploy, weekly retrain pipelines
```

---

## The 6-Agent Pipeline

```
[START]
   │
   ▼
DataIntelligenceAgent   — fetches features from DB, runs anomaly/drift checks
   │
   ▼
PredictionAgent         — XGBoost ensemble, conformal 95% CI, risk tier routing
   │
   ├─(LOW)──────────────► low_risk_terminal ──► [END]
   │
   ▼
ExplanationAgent        — SHAP values, LIME agreement, LLM narrative
   │
   ▼
CounterfactualAgent     — DiCE / rule-based perturbation, business constraint filter
   │
   ▼
RetentionStrategistAgent — Knapsack optimizer, A/B assignment, CRM dispatch
   │
   ▼
HITLAgent               — CRITICAL: Slack approval gate; HIGH: alert + auto-approve
   │
   ▼
[END:complete]
```

---

## Technology Stack

**AI / ML:** LangGraph, LangChain, XGBoost, SHAP, LIME, DiCE, MAPIE, Optuna, scikit-learn  
**Backend:** FastAPI, Celery, PostgreSQL, Redis, ChromaDB, MLflow, SQLAlchemy  
**Frontend:** Next.js 14, TypeScript, Tailwind CSS, IBM Carbon Design System  
**Infrastructure:** Docker, Kubernetes, Prometheus, Grafana, GitHub Actions  

---

## Documentation

See [GUIDEBOOK.md](GUIDEBOOK.md) for a complete walkthrough of every phase, every file, and every design decision — written to be understandable by anyone with basic programming knowledge.
