# The Complete Guidebook
# Agentic AI-Powered Customer Churn Intelligence Platform

> **Written for:** College students with basic programming knowledge  
> **Assumes:** You know what Python is, have heard of AI/ML, and can follow logic  
> **Goal:** By the end of this guide, you will understand every single piece of this project — what it is, why it exists, how it works, and why it matters

---

## Table of Contents

1. [The Real-World Problem](#1-the-real-world-problem)
2. [What Is Customer Churn?](#2-what-is-customer-churn)
3. [How Companies Currently Handle Churn (The Old Way)](#3-how-companies-currently-handle-churn-the-old-way)
4. [What This Project Does Differently](#4-what-this-project-does-differently)
5. [The Big Picture Architecture](#5-the-big-picture-architecture)
6. [Phase 0 — Foundation & Data Pipeline](#6-phase-0--foundation--data-pipeline)
7. [Phase 1 — Machine Learning Model Training](#7-phase-1--machine-learning-model-training)
8. [Phase 2 — Explainability Layer](#8-phase-2--explainability-layer)
9. [Phase 3 — Agentic AI Core](#9-phase-3--agentic-ai-core)
10. [Phase 4 — Advanced Agents](#10-phase-4--advanced-agents)
11. [Phase 5 — Human-in-the-Loop (HITL) + Feedback Loop](#11-phase-5--human-in-the-loop-hitl--feedback-loop-)
12. [Phase 6 — The Full-Stack Application](#12-phase-6--the-full-stack-application)
13. [Phase 7 — Optimization & Fairness](#13-phase-7--optimization--fairness)
14. [Phase 8 — Deployment & Monitoring](#14-phase-8--deployment--monitoring)
15. [Technology Stack Explained](#15-technology-stack-explained)
16. [How Everything Connects](#16-how-everything-connects)
17. [Why This Project Stands Out](#17-why-this-project-stands-out)

---

## 1. The Real-World Problem

### Imagine you run a subscription service

Think of Netflix, Spotify, a gym membership, or a SaaS software company. Your business model works like this:

- Customers **sign up** and pay you every month
- As long as they stay, you earn money
- When they **leave**, you lose that recurring revenue

This "leaving" is called **churn**. And it is one of the most expensive problems in modern business.

### The numbers are brutal

| Fact | Why It Matters |
|---|---|
| Acquiring a new customer costs **5–7× more** than retaining one | Losing a customer is financially devastating |
| A **5% reduction** in churn can increase profits by **25–95%** | Small improvements have massive financial impact |
| The average SaaS company loses **5–7% of its customers every month** | At this rate, half your customers are gone in a year |
| Most customers leave **without saying why** | You can't fix problems you don't know about |

### The core challenge

You have thousands (or millions) of customers. Some of them are unhappy right now. They are going to cancel next month. But you don't know **which ones**, and you don't know **why**, and you don't know **what to do about it**.

If only you could:
1. **Predict** which customers are about to leave — before they actually leave
2. **Understand** why each specific customer is at risk
3. **Know** exactly what action would most effectively retain them
4. **Execute** that retention action efficiently, with human approval for important cases
5. **Learn** from whether it worked, to get smarter over time

That is exactly what this platform does.

---

## 2. What Is Customer Churn?

### Simple definition

**Churn** = A customer stops paying for your service and leaves.

### Types of churn

```
Voluntary Churn          Involuntary Churn
─────────────────        ────────────────────
Customer chose to leave  Customer's card failed
Found a better option    Payment processing error
Too expensive            Technical issues prevented renewal
Bad product experience   
Poor customer support    
```

This platform focuses primarily on **voluntary churn** — the kind you can predict and prevent.

### The churn signal: customers tell you before they leave

Here's the key insight: **customers don't just disappear overnight**. They send signals for weeks or months before they cancel:

- They **log in less often** → Losing interest
- They **open fewer features** → Not getting value
- They **raise more support tickets** → Frustrated
- Their **usage drops** → Disengaging
- Their **NPS score falls** → Unhappy
- They **switch to a monthly plan** → Looking for flexibility to leave

These signals are sitting in your data. This platform collects them, learns from them, and acts on them.

---

## 3. How Companies Currently Handle Churn (The Old Way)

### Method 1: Gut feeling
A customer success manager notices a customer seems quiet and sends a check-in email. This is completely random and unscalable.

### Method 2: Simple rule-based systems
```
IF customer hasn't logged in for 30 days → send email
IF customer submitted 3 support tickets → flag for review
```
Problems:
- Rules are too simplistic (not every quiet customer is churning)
- Rules can't handle complex combinations of signals
- Someone has to manually create and maintain every rule
- No way to know which action to take, only that something is wrong

### Method 3: Basic ML models (what most companies use today)
Some companies train a basic logistic regression or decision tree that outputs a churn score. An analyst runs it once a week and produces a spreadsheet.

Problems:
- **Black box**: It says "this customer will churn" but can't say why
- **No action**: The model predicts but doesn't recommend what to do
- **Stale**: Weekly batch runs miss fast-moving situations
- **No feedback loop**: The model never learns if its recommendations worked
- **No human oversight**: Critical decisions are made automatically or not at all
- **Single model**: If it breaks or drifts, everything fails

This platform solves every single one of these problems.

---

## 4. What This Project Does Differently

### The three revolutions in this platform

#### Revolution 1: Agentic AI
Instead of a model that just outputs a number, this platform has **AI agents** — intelligent, autonomous software entities that can:
- Reason about customer situations
- Use tools (look things up, run models, query databases)
- Talk to each other and collaborate
- Make decisions and explain their reasoning
- Ask humans for approval when stakes are high

Think of it like having a team of AI specialists, each with a specific job, working together.

#### Revolution 2: Full Explainability
Every prediction comes with a **plain-English explanation**:
- *"This customer is 87% likely to churn because their usage dropped 60% in the last 30 days and they opened 4 support tickets this month"*
- *"If you reduce their price by 15%, their churn probability drops to 31%"*

No black boxes. Every decision is auditable.

#### Revolution 3: Closed Feedback Loop
The system **learns from its own interventions**. When it recommends a retention action and a human approves it, it tracks whether the customer stayed or left. Over time, it becomes smarter about which actions actually work for which types of customers.

---

## 5. The Big Picture Architecture

Think of the platform as a building with multiple floors:

```
┌─────────────────────────────────────────────────────────┐
│                    FLOOR 4 (TOP)                        │
│              Next.js Web Application                    │
│     What humans see: dashboards, alerts, approvals      │
└──────────────────────────┬──────────────────────────────┘
                           │  (HTTP requests, WebSockets)
┌──────────────────────────▼──────────────────────────────┐
│                      FLOOR 3                            │
│               FastAPI Backend Server                    │
│    Authentication, routing, real-time WebSocket stream  │
└──────────────────────────┬──────────────────────────────┘
                           │  (function calls)
┌──────────────────────────▼──────────────────────────────┐
│                      FLOOR 2                            │
│            Agentic AI Orchestration Layer               │
│     LangGraph agents reasoning, deciding, acting        │
└──────┬──────────────────────────────────────────┬───────┘
       │                                          │
┌──────▼──────────────┐              ┌────────────▼───────┐
│       FLOOR 1A      │              │      FLOOR 1B      │
│   ML + XAI Layer    │              │  Optimization Layer│
│  5 trained models   │              │  Budget optimizer  │
│  SHAP/LIME/DiCE     │              │  ROI calculator    │
└──────┬──────────────┘              └────────────┬───────┘
       │                                          │
┌──────▼──────────────────────────────────────────▼───────┐
│                    BASEMENT                             │
│                    Data Layer                           │
│  PostgreSQL (records) │ Redis (cache) │ ChromaDB (AI)   │
│  MLflow (experiments) │ Feature Store │ Drift Detector  │
└─────────────────────────────────────────────────────────┘
```

Every floor depends on the floor below it. Let's understand each floor from the ground up.

---

## 6. Phase 0 — Foundation & Data Pipeline

### What is a "data pipeline"?

A pipeline is like a factory assembly line. Raw materials (messy customer data) go in one end, and clean, standardised, analysis-ready data comes out the other end.

### 6.1 The Synthetic Data Generator

**File:** `data/synthetic/generate_synthetic.py`

#### Why do we need synthetic data?

When building a real system, you often don't have production data yet. Instead, we generate **fake-but-realistic** data that mimics real customer behaviour.

#### What data does it generate?

It creates 10,000 fake customers, each with these attributes:

| Column | What it represents | Example |
|---|---|---|
| `customer_id` | Unique identifier (UUID) | `"a3f1-bc92-..."` |
| `age` | Customer's age | `34` |
| `tenure_months` | How long they've been a customer | `18` |
| `contract_type` | Month-to-month, 1-year, or 2-year | `"Month-to-Month"` |
| `plan_tier` | Basic, Standard, or Premium | `"Standard"` |
| `payment_method` | How they pay | `"Credit Card"` |
| `monthly_charges` | What they pay monthly | `$65.00` |
| `total_charges` | Total paid lifetime | `$1,170.00` |
| `num_support_tickets_30d` | Support tickets in last 30 days | `3` |
| `login_frequency_30d` | How often they logged in | `12` |
| `feature_adoption_rate` | % of features they use (0–1) | `0.45` |
| `nps_score` | Net Promoter Score (-100 to +100) | `-20` |
| `usage_30d` | Usage in last 30 days | `28.5` |
| `usage_60d` | Usage in last 60 days | `45.2` |
| `usage_90d` | Usage in last 90 days | `62.1` |
| `days_since_last_login` | Days since last activity | `14` |
| `churn` | Did they leave? (0=No, 1=Yes) | `1` |

#### How does it make churn realistic?

This is the clever part. Churn is not random — it's correlated with the other features. The generator uses a **logistic model** to calculate a churn score for each customer:

```python
# Simplified concept:
churn_risk_score = (
    +2.5 × (month-to-month contract)       # riskiest contract type
    +1.5 × (more support tickets)           # frustrated customers leave
    -1.5 × (higher feature adoption)        # engaged customers stay
    -1.2 × (higher login frequency)         # active customers stay
    +1.8 × (days since last login)          # dormant customers leave
    -1.0 × (higher NPS score)               # happy customers stay
    +1.0 × (declining usage trend)          # disengaging customers leave
)

# Then convert score to probability using sigmoid function
churn_probability = 1 / (1 + e^(-churn_risk_score))

# Randomly assign churn based on that probability
churn = 1 if random() < churn_probability else 0
```

This produces a dataset with ~27% churn rate — realistic for SaaS businesses.

---

### 6.2 Schema Validation

**File:** `src/preprocessing/validators.py`

#### What is schema validation and why do we need it?

Imagine you're building a bridge. Before you start welding steel, you verify the steel meets specifications. Schema validation does the same for data.

Real-world data is **messy**:
- Someone entered age as `"thirty-four"` instead of `34`
- A support ticket count came in as `-2` (impossible)
- A customer record is missing the contract type entirely

If bad data enters your pipeline, your model will produce garbage predictions — and you might not even notice. Schema validation catches these problems at the door.

#### How it works (Pandera)

```python
# Define what valid data looks like:
RAW_CUSTOMER_SCHEMA = DataFrameSchema({
    "age": Column(float, checks=[
        Check.greater_than_or_equal_to(18),   # no children
        Check.less_than_or_equal_to(100)       # no immortals
    ]),
    "contract_type": Column(str, checks=
        Check.isin(["Month-to-Month", "One Year", "Two Year"])
    ),
    # ... etc
})

# Validate incoming data:
RAW_CUSTOMER_SCHEMA.validate(df)  # raises SchemaError if data is wrong
```

If data is invalid, the pipeline stops and logs exactly what went wrong — preventing silent corruption.

---

### 6.3 Feature Engineering

**File:** `src/preprocessing/feature_engineering.py`

#### What is feature engineering?

Raw data columns are often not the most useful form for a model. Feature engineering **transforms and combines** raw columns into more informative signals.

Think of it like this: you have the raw ingredients (flour, eggs, butter), and feature engineering is the act of mixing them into dough before baking.

#### The derived features this platform creates

**Usage Decline Rate:**
```
usage_decline_rate = (usage_90d - usage_30d) / usage_90d

Example: If a customer used 60 units 90 days ago but only 20 today:
decline_rate = (60 - 20) / 60 = 0.67  (67% decline — danger signal!)
```

**Support Escalation Flag:**
```
support_escalation_flag = 1  if  num_support_tickets_30d >= 5  else  0

A binary alarm: "This customer is in trouble with our product"
```

**Charge Per Month Normalised:**
```
charge_per_month_normalised = monthly_charges / (tenure_months + 1)

New customers paying high amounts are higher risk than long-tenured customers
paying the same amount — this ratio captures that.
```

**Tenure Bin:**
```
Buckets: [0-6 months] [7-12 months] [13-24 months] [25-48 months] [49-84 months]
Encoded:      0              1              2              3              4

New customers (bucket 0) churn at different rates than loyal customers (bucket 4)
```

**NPS Segment:**
```
Detractor (NPS < 0)  → 0   ← Most likely to churn
Passive   (NPS 0-50) → 1
Promoter  (NPS > 50) → 2   ← Most likely to stay
```

---

### 6.4 The Preprocessing Pipeline

**File:** `src/preprocessing/pipeline.py`

#### What is a scikit-learn Pipeline?

A `Pipeline` is a chain of data transformation steps that are applied in sequence. The key benefit: the entire chain can be **saved, loaded, and applied identically to new data**.

#### The 5-step pipeline

```
Raw Data
   │
   ▼
Step 1: Feature Engineering
   │  (creates derived columns from raw ones)
   ▼
Step 2: Outlier Capping (IQR method)
   │  (prevents extreme values from distorting the model)
   ▼
Step 3: Categorical Encoding (One-Hot)
   │  (converts text columns to numbers the model can read)
   ▼
Step 4: Missing Value Imputation (Median)
   │  (fills gaps in data with sensible defaults)
   ▼
Step 5: Feature Scaling (RobustScaler)
   │  (normalises all numbers to a similar range)
   ▼
Clean Feature Matrix (ready for ML models)
```

#### Why do we cap outliers? (IQR method explained)

IQR = Interquartile Range = the spread of the middle 50% of your data.

```
Q1 (25th percentile) = 20
Q3 (75th percentile) = 80
IQR = Q3 - Q1 = 60

Lower cap = Q1 - 1.5 × IQR = 20 - 90 = -70
Upper cap = Q3 + 1.5 × IQR = 80 + 90 = 170

Any value below -70 gets set to -70.
Any value above 170 gets set to 170.
```

Why? Because one customer who paid $10,000/month would otherwise dominate the entire model.

#### Why do we scale features?

Some ML algorithms measure "distance" between customers. Without scaling:
```
age = 45  (range: 18-75)
monthly_charges = 65  (range: 18-120)
total_charges = 2,000  (range: 0-50,000)
```

The `total_charges` column (with its large numbers) would completely dominate the algorithm, even though it's not necessarily the most important feature. Scaling makes them all comparable.

---

### 6.5 Drift Detection

**File:** `src/preprocessing/drift_detector.py`

#### What is data drift?

Imagine you train a model in January using customer data from 2024. By July, customer behaviour has changed — maybe a new competitor entered the market, or you launched a new feature. The data in July **looks different** from January. Your model was trained on January data. If July data has "drifted" too far from what the model learned, its predictions become unreliable.

#### How drift is detected

For **numeric features** (age, charges, usage): **Kolmogorov-Smirnov (KS) Test**

```
The KS test asks: "Are these two samples from the same distribution?"

Reference data (January):  [20, 25, 30, 35, 40, 45, 50]  average login = 35
Current data  (July):      [5,  8,  12, 10, 7,  9,  11]  average login = 9

KS statistic = 0.85, p-value = 0.0001
p-value < 0.05 → DRIFT DETECTED ✓
```

For **categorical features** (contract type, plan tier): **Chi-Square Test**

```
Reference: Month-to-Month=55%, One Year=25%, Two Year=20%
Current:   Month-to-Month=80%, One Year=15%, Two Year=5%

Chi² statistic = 24.3, p-value = 0.00001
p-value < 0.05 → DRIFT DETECTED (big shift to risky contracts!)
```

When drift is detected, the system triggers a **model retraining pipeline** automatically.

---

### 6.6 Docker Compose & Infrastructure

**File:** `docker/docker-compose.yml`

#### What is Docker?

Docker packages software into isolated containers — like shipping containers. Each service runs in its own container with everything it needs. You can spin up an entire infrastructure with one command: `docker compose up`.

#### The four services

**PostgreSQL 17** (the main database)
- Stores customer records, model metadata, intervention history, user accounts
- Think of it as the system's long-term memory

**Redis 7** (the fast cache + task queue)
- Stores frequently-accessed data in RAM for ultra-fast retrieval
- Also serves as the message queue for background tasks (Celery)
- Think of it as the system's short-term working memory

**ChromaDB** (the vector database)
- Stores "embeddings" — mathematical representations of text
- Used by AI agents to search their memory by semantic similarity
- Example: An agent can search "customers similar to John who churned" and find relevant past cases
- Think of it as the AI's searchable notebook

**MLflow** (the ML experiment tracker)
- Logs every model training run: what parameters were used, what metrics were achieved
- Stores trained model files and lets you compare versions
- Tracks model versions through Staging → Production lifecycle
- Think of it as the lab notebook for all ML experiments

---

## 7. Phase 1 — Machine Learning Model Training

### Why do we train multiple models?

Different algorithms have different strengths. Rather than betting everything on one approach, we train five models and let them vote together. This is called an **ensemble**.

Think of it like asking five different doctors for a diagnosis. Each has different expertise. Together they're more reliable than any one alone.

### 7.1 The Five Models

#### Model 1: Logistic Regression (the baseline)

```
Mathematical idea: Draw the best straight line (or flat plane in multiple 
dimensions) that separates churners from non-churners.

Output: A probability between 0 and 1
Example: churn_probability = sigmoid(0.3×support_tickets - 0.5×feature_adoption + ...)
```

**Strengths:** Fast, interpretable, gives clear coefficient weights  
**Weaknesses:** Can't capture non-linear patterns (e.g., "usage drop only matters for new customers")

#### Model 2: Random Forest

```
Idea: Build 300 decision trees, each trained on a random subset of data and features.
      Each tree makes a prediction. Final answer = majority vote of all 300 trees.

Individual tree example:
   Is monthly_charges > 80?
   ├── YES: Is feature_adoption_rate < 0.3?
   │        ├── YES: CHURN (prob = 0.82)
   │        └── NO:  STAY  (prob = 0.35)
   └── NO:  Is tenure_months < 6?
            ├── YES: CHURN (prob = 0.71)
            └── NO:  STAY  (prob = 0.22)
```

**Strengths:** Handles non-linear patterns, robust to outliers, provides feature importance  
**Weaknesses:** Slow to train on large datasets, less interpretable than logistic regression

#### Model 3: XGBoost (the powerhouse)

```
Idea: Build trees SEQUENTIALLY. Each new tree focuses specifically on 
the mistakes made by all previous trees — it "boosts" the ensemble 
by correcting errors.

Tree 1: Gets 70% of predictions right
Tree 2: Focuses on the 30% Tree 1 got wrong → now 82% right
Tree 3: Focuses on remaining errors → now 88% right
...continue for 300 rounds
```

**Strengths:** State-of-the-art performance on tabular data, fast prediction  
**Weaknesses:** More hyperparameters to tune, can overfit without careful tuning  

**Why `scale_pos_weight=2.7`?**  
Our dataset is 73% non-churn and 27% churn. Without correction, the model would learn to always predict "stay" and be 73% accurate — which is useless. `scale_pos_weight = 73/27 ≈ 2.7` tells XGBoost to treat churn cases as 2.7× more important.

#### Model 4: LightGBM (XGBoost's faster cousin)

Same gradient boosting idea as XGBoost, but with two key optimisations:
- **Leaf-wise tree growth** (smarter about where to split) vs. XGBoost's level-wise
- **Histogram-based binning** (groups continuous values into buckets for speed)

**Strengths:** 10× faster than XGBoost on large datasets, lower memory usage  
**Weaknesses:** Can overfit small datasets more easily

#### Model 5: Stacking Ensemble (the final production model)

```
Stage 1: Train XGBoost, LightGBM, Random Forest on training data
         Each model produces churn probability predictions

Stage 2: Take those probability outputs as NEW FEATURES
         Train Logistic Regression (meta-learner) on these meta-features

Input data → [XGB: 0.82, LGBM: 0.78, RF: 0.71] → Meta-LR → Final: 0.80
```

Why does this work? The meta-learner learns **which base model to trust more** in different situations. If XGBoost is consistently better for new customers and LightGBM for older ones, the meta-learner figures that out.

---

### 7.2 Cross-Validation (Stratified K-Fold)

#### Why can't we just train on all the data?

If you study your exam with the exact questions on the exam, you'll score 100% — but you haven't actually learned anything. Similarly, if you train and test on the same data, you'll get misleadingly optimistic performance metrics.

#### What is k-fold cross-validation?

```
Dataset split into 5 equal "folds":
│ Fold 1 │ Fold 2 │ Fold 3 │ Fold 4 │ Fold 5 │

Round 1: Train on [2,3,4,5], Test on [1] → AUC = 0.86
Round 2: Train on [1,3,4,5], Test on [2] → AUC = 0.88
Round 3: Train on [1,2,4,5], Test on [3] → AUC = 0.87
Round 4: Train on [1,2,3,5], Test on [4] → AUC = 0.89
Round 5: Train on [1,2,3,4], Test on [5] → AUC = 0.87

Final AUC = average(0.86, 0.88, 0.87, 0.89, 0.87) = 0.874
```

"Stratified" means each fold has the same ratio of churn/non-churn as the full dataset, preventing an unlucky fold that has no churn examples.

---

### 7.3 Hyperparameter Tuning (Optuna)

#### What are hyperparameters?

Regular parameters are learned during training (e.g., "how much does support tickets contribute to churn?"). Hyperparameters are **settings chosen before training** that control how training works:

```
XGBoost hyperparameters examples:
- n_estimators: How many trees to build? (100? 500? 1000?)
- max_depth: How deep should each tree grow? (3? 6? 10?)
- learning_rate: How much to correct on each step? (0.01? 0.1? 0.3?)
- subsample: What % of data to use for each tree? (0.6? 0.8? 1.0?)
```

There are thousands of possible combinations. Trying them all would take forever.

#### How Optuna solves this (Bayesian Optimisation)

Instead of random or grid search, Optuna uses **Bayesian optimisation**:

```
Trial 1: Try random settings → AUC = 0.82
Trial 2: Based on Trial 1, make an educated guess → AUC = 0.85
Trial 3: Focus search in promising region → AUC = 0.87
Trial 4: Narrow down further → AUC = 0.88
...
Trial 50: Best settings found → AUC = 0.91
```

It builds a probabilistic model of which hyperparameters work well and samples more densely from promising regions. Much smarter than guessing.

---

### 7.4 Evaluation Metrics

#### Why not just use accuracy?

If 95% of transactions are legitimate and 5% are fraud, a model that always says "not fraud" has **95% accuracy** but is completely useless.

For churn prediction (27% positive rate), we use:

**AUC-ROC** (primary metric)
```
ROC curve: plots True Positive Rate vs False Positive Rate at all thresholds
AUC (Area Under Curve): 
  1.0 = perfect model
  0.5 = random guessing
  0.9+ = excellent (our target)

Interpretation: "What is the probability that a randomly chosen churner 
                 scores higher than a randomly chosen non-churner?"
```

**F1-Score**
```
Precision = Of all customers I flagged as churning, how many actually did?
Recall    = Of all customers that actually churned, how many did I catch?

F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = harmonic mean of precision and recall

Good for imbalanced datasets where both false positives and false negatives matter
```

**Brier Score** (probability calibration)
```
Measures if predicted probabilities are realistic:

If model says "80% chance of churn" for 100 customers,
approximately 80 of them should actually churn.

Lower Brier Score = better calibrated probabilities
Range: 0 (perfect) to 1 (perfectly wrong)
```

**Confusion Matrix**
```
                 Predicted: Stay  |  Predicted: Churn
Actual: Stay         TN (✓)       |      FP (✗) 
Actual: Churn        FN (✗)       |      TP (✓)

TN = True Negative  (correctly said they'd stay)
TP = True Positive  (correctly predicted churn)
FP = False Positive (wrongly alarmed — wasted retention effort)
FN = False Negative (missed a churner — lost revenue)
```

---

### 7.5 Conformal Prediction (Uncertainty Quantification)

**File:** `src/models/uncertainty.py`

#### The problem with just a probability number

When a model says "87% churn probability", should you fully trust that? What if the model is uncertain — maybe 87% could really be anywhere from 60% to 95%?

#### What conformal prediction gives you

```
Standard output:   { "churn_prob": 0.87 }

Conformal output:  { 
    "churn_prob": 0.87, 
    "confidence_interval": [0.82, 0.91],
    "coverage": 0.90,
    "is_uncertain": False
}
```

The interval `[0.82, 0.91]` means: "We're 90% confident the true churn probability is between 82% and 91%."

When `is_uncertain: True`, both class labels are in the prediction set — the model is genuinely confused and the human-in-the-loop should be alerted.

#### How it works (MAPIE library)

1. Train the model on training data
2. Run predictions on a separate **calibration set** (data the model hasn't seen)
3. For each calibration example, calculate the "nonconformity score" — how surprising this prediction is
4. Store the distribution of these scores
5. For new predictions, use this distribution to construct intervals with guaranteed coverage

---

### 7.6 MLflow Model Registry

**File:** `src/models/registry.py`

#### Why do we need a model registry?

Imagine a pharmaceutical company developing a drug. They track every experiment, every formulation, every trial result. They have strict stages before a drug is approved for patients.

MLflow does the same for ML models:

```
Model Lifecycle:
Experiment Run → Staging → Validation → Production → Archived

Every run records:
- Exact code version
- Hyperparameters used
- All evaluation metrics
- The model file itself
- Who ran it and when
```

This makes ML development **reproducible** — anyone can look back at any experiment and understand exactly how that model was produced.

#### Champion-Challenger A/B Testing

```
Current production model (Champion): AUC = 0.88
New trained model (Challenger):      AUC = 0.91

If challenger.auc > champion.auc + 0.005:
    → Promote challenger to Production
    → Archive old champion

Traffic split during evaluation: 80% → Champion, 20% → Challenger
```

---

## 8. Phase 2 — Explainability Layer

### The "Why" problem

Imagine a doctor telling you: "You need surgery." You'd immediately ask "Why?" You'd want to understand the reasoning before agreeing.

The same applies to AI decisions. When the system says "Customer #4821 is about to churn, and you should spend $120 to retain them", every stakeholder needs to understand why — for trust, for legal compliance, and for taking the right action.

### 8.1 SHAP — Global & Local Explainability

**File:** `src/explainability/shap_explainer.py`

#### What is SHAP?

SHAP = **SH**apley **A**dditive ex**P**lanations

It's based on **Shapley values** from game theory. The core idea: if a group of players cooperate to win a prize, how much did each player contribute?

In ML terms: the "players" are features, the "prize" is the model's prediction, and SHAP values tell you how much each feature contributed.

#### Global SHAP (understanding the model overall)

```
Feature Importance (average |SHAP value| across all customers):

feature_adoption_rate    ████████████████████  0.32
days_since_last_login    ████████████████      0.28
usage_decline_rate       █████████████         0.23
num_support_tickets_30d  ██████████            0.18
contract_type            ████████              0.15
login_frequency_30d      ██████                0.12
...

Insight: Feature adoption is the single most powerful predictor of churn.
```

#### Local SHAP (explaining one specific customer)

```
Customer #4821 — Churn Probability: 87%

Base value (average prediction): 27%

+ days_since_last_login = 85     → +22% (hasn't logged in for 3 months!)
+ num_support_tickets = 7        → +15% (very frustrated)
+ usage_decline_rate = 0.68      → +12% (usage fell 68%)
+ contract_type = Month-to-Month → +11% (no commitment)
- feature_adoption_rate = 0.62   → -8%  (moderate engagement reduces risk)
- tenure_months = 36             → -5%  (3-year customer, some loyalty)
                                   ────
Final prediction:                  74% → ... → 87% (after all interactions)
```

This waterfall diagram tells the account manager exactly what to focus on.

---

### 8.2 LIME — Model-Agnostic Local Explanations

**File:** `src/explainability/lime_explainer.py`

#### How LIME works

SHAP is exact but mathematically complex. LIME takes a different, more intuitive approach:

```
Goal: Explain why the model predicted 87% churn for Customer #4821

LIME's approach:
1. Take Customer #4821's features
2. Create 5,000 "perturbed neighbours" — slightly different versions of this customer
   (change some features randomly, keeping others the same)
3. Run the black-box model on all 5,000 neighbours
4. Fit a SIMPLE interpretable model (linear regression) on these 5,000 predictions
5. The simple model's coefficients = the explanation

The linear model is only locally accurate — it explains this specific prediction,
not the model globally.
```

#### Why use both SHAP and LIME?

They use completely different mathematical approaches. When they **agree** on the top risk factors, you can be very confident in the explanation. When they **disagree**, it flags that the model's behaviour in this region is complex and deserves human scrutiny.

#### SHAP-LIME Agreement Score

```python
# Top 5 features ranked by SHAP: [adoption_rate, days_since_login, usage_decline, tickets, contract]
# Top 5 features ranked by LIME: [days_since_login, adoption_rate, tickets, usage_decline, tenure]

# Overlap at each rank position:
Rank 1: {adoption_rate} ∩ {days_since_login} = {} → 0/1 = 0.0
Rank 2: {adoption, days_login} ∩ {days_login, adoption} = {both} → 2/2 = 1.0
Rank 3: ...

Agreement score = average = 0.73  (73% agreement — good!)
```

---

### 8.3 DiCE — Counterfactual Explanations

**File:** `src/explainability/counterfactual.py`

#### What is a counterfactual?

"What would need to be different for this customer NOT to churn?"

This is incredibly valuable because it directly suggests **what action to take**.

```
Current situation (Customer #4821):
- monthly_charges: $95         Churn probability: 87%
- feature_adoption_rate: 0.62
- contract_type: Month-to-Month
- num_support_tickets_30d: 7

Counterfactual 1 (cheapest):
- monthly_charges: $80.75      Churn probability: 31%
  (15% price reduction → costs company ~$170/year)

Counterfactual 2 (most effective):
- feature_adoption_rate: 0.82  Churn probability: 24%
  (assign CSM → costs ~$120 one-time)

Counterfactual 3 (balanced):
- contract_type: One Year       Churn probability: 28%
  (loyalty discount → costs ~$80 one-time)
```

#### Business Constraint Filtering

Not all counterfactuals are practically useful. The engine enforces:
- Maximum discount ≤ 30% (business policy)
- Maximum retention cost ≤ $300 per customer
- Only actionable changes (can't change a customer's age)

#### Ranking interventions

Each intervention is scored on:
1. **Impact** — how much does it reduce churn probability?
2. **Cost** — what does it cost the company?
3. **Effort** — how hard is it to execute?
4. **Time** — how quickly will it take effect?

Final ranking = optimal balance of all four dimensions.

---

### 8.4 Narrative Generator

**File:** `src/explainability/narrative_generator.py`

#### The last mile problem

SHAP values and counterfactuals are great for data scientists, but account managers and executives need **plain English**.

The narrative generator converts technical outputs to readable summaries:

```
TEMPLATE MODE OUTPUT:

Customer C-4821 has a HIGH churn risk with a predicted probability of 87%.

Key churn drivers (SHAP analysis):
Factors increasing churn risk:
  • Days Since Last Login (contribution: +0.220) 
  • Num Support Tickets 30d (contribution: +0.150)
  • Usage Decline Rate (contribution: +0.120)
Factors reducing churn risk:
  • Feature Adoption Rate (contribution: −0.080)
  • Tenure Months (contribution: −0.050)

A model-agnostic analysis (LIME) corroborates this, highlighting 
Days Since Last Login as the strongest local driver.

Recommended Retention Actions:
  1. Reduce monthly price by 15% → reduces churn risk to 31% 
     (↓56% reduction, estimated cost: $51)
  2. Assign dedicated onboarding / CSM support → reduces churn risk to 28% 
     (↓59% reduction, estimated cost: $120)
  3. Offer annual contract upgrade with loyalty discount → reduces churn risk to 28%
     (↓59% reduction, estimated cost: $80)
```

In Phase 3, the LLM mode takes this structure and produces an even more natural, contextualised narrative using GPT-4o or Claude.

---

## 9. Phase 3 — Agentic AI Core

This is the most innovative and important phase of the entire project.

### What is an AI Agent?

A traditional ML model is like a **calculator**: you give it input, it gives you output. Done.

An AI Agent is like a **junior analyst**: you give it a goal, and it figures out what to do — looking things up, running calculations, asking questions, making decisions, and reporting back.

Formally:
```
Agent = LLM (brain) + Tools (hands) + Memory (context) + Feedback loop
```

### What is LangGraph?

LangGraph is a framework for building **networks of AI agents** that collaborate to solve complex problems. Each agent is a **node** in a directed graph. Data (the `AgentState`) flows between nodes via **edges**. Conditional edges let the graph route to different agents depending on what happened in the previous step.

Think of it like an org chart:

```
ChurnOrchestrator (LangGraph StateGraph — the wiring)
├── DataIntelligenceAgent  (data_intelligence_agent.py)
├── PredictionAgent        (prediction_agent.py)
├── ExplanationAgent       (explanation_agent.py)
├── CounterfactualAgent    (Phase 4 — counterfactual_agent.py)
├── RetentionStrategistAgent (Phase 4 — retention_strategist.py)
├── HITLAgent              (Phase 5 — hitl_agent.py)
└── FeedbackAgent          (Phase 5 — feedback_agent.py)
```

### The AgentState — the shared blackboard

Every agent reads from and writes to a single typed dictionary called `AgentState` (defined in `agents/state.py`). Think of it as a shared whiteboard that travels through the pipeline:

```python
# What the state looks like mid-pipeline:
{
    "customer_id": "C-4821",
    "run_id": "a3f1bc92",
    "customer_features": { "tenure_months": 36, "monthly_charges": 95.0, ... },
    "data_quality": { "passed": True, "anomaly_features": [], ... },
    "prediction": {
        "churn_probability": 0.87,
        "confidence_interval": [0.82, 0.91],
        "risk_tier": "HIGH",
        "model_version": "production",
        "is_uncertain": False
    },
    "explanation": {
        "narrative_text": "Customer C-4821 is at HIGH risk...",
        "top_risk_factors": [...],
        "shap_contributions": { "days_since_last_login": 0.22, ... }
    },
    "completed_steps": ["data_intelligence", "prediction", "explanation"],
    "errors": []
}
```

Each agent only modifies the keys it owns — LangGraph merges the partial updates automatically.

### The Graph Topology (Phase 3)

```
[START]
   │
   ▼
data_intelligence ──(abort?)──► [END:error]
   │
   ▼
prediction ────────(abort?)──► [END:error]
   │
   ├─(LOW risk)─────────────► [END:low_risk]   ← skip expensive explanation
   │
   └─(MEDIUM/HIGH/CRITICAL)─►
   │
   ▼
explanation
   │
   ▼
[END:complete]
```

Conditional edges (e.g., "LOW risk → skip explanation") are defined as Python functions that inspect the current state and return a routing string. This means the graph is **dynamic** — its execution path depends on real data, not just hardcoded sequence.

### The Four Agent Tools (agents/tools/)

| Tool | What it does |
|---|---|
| `sql_tool.py` | Safe read-only SQL queries to PostgreSQL — only SELECT allowed |
| `model_tool.py` | Loads production model from MLflow (or local .pkl fallback), runs prediction |
| `shap_tool.py` | Computes SHAP values for a single customer, returns structured contribution dict |
| `drift_tool.py` | Per-instance anomaly check — flags features > 3 standard deviations from training mean |

Tools are decorated with LangChain's `@tool` decorator and have Pydantic-validated input schemas. This means an LLM can call them safely — it can't pass the wrong argument types.

### The Two Memory Layers

**Short-term memory — Redis** (`memory/redis_state.py`)
- Stores the full `AgentState` snapshot for each task (24-hour TTL)
- Powers real-time status polling: `/agent/status/{run_id}`
- Streams progress events to the WebSocket layer as each step completes
- Caches enriched customer context for 1 hour (avoid re-fetching on retries)

**Long-term memory — ChromaDB** (`memory/vector_store.py`)
- Three collections: `customer_interactions`, `churn_explanations`, `customer_profiles`
- Each customer's risk profile is stored as an **embedding** — a mathematical representation of their description
- Agents can search: *"find customers similar to C-4821 who were successfully retained"*
- The explanation agent stores every generated narrative, which future runs can retrieve for context

### Async Execution (Celery + Redis)

Running the full pipeline (fetch → predict → explain) can take 5–30 seconds depending on whether an LLM is called. HTTP requests shouldn't wait that long. The solution:

```
POST /agent/run  →  Celery task submitted  →  Returns task_id immediately
                                                       │
GET /agent/status/{task_id}  ←──── polling ────────────┘

Result ready?  GET /agent/status/{task_id}  returns full result
Still running? GET /agent/status/{task_id}  returns stream_events (partial progress)
```

`tasks.py` defines two Celery tasks:
- `run_churn_pipeline` — single customer, max 2 retries on crash
- `run_batch_pipeline` — list of customers, streams progress updates

### LangSmith Observability

When `LANGCHAIN_TRACING_V2=true` is set, every LLM call, tool invocation, and agent decision is automatically traced to LangSmith. You can open the LangSmith dashboard and see:

```
Run: "Analyse C-4821"  (28.3s)
├── DataIntelligenceAgent  (1.2s)
│   └── tool: sql_query_tool  → 1 row returned
│   └── tool: drift_check_tool → passed=True
├── PredictionAgent  (0.8s)
│   └── tool: churn_prediction_tool → 0.87
└── ExplanationAgent  (24.1s)
    └── tool: shap_explanation_tool → top driver: days_since_last_login
    └── LLM call: gpt-4o → narrative generated (487 tokens)
```

This is critical for debugging agent behaviour and explaining AI decisions to stakeholders.

### How to run Phase 3

```powershell
# Template mode (no LLM key needed)
python agents/orchestrator.py --customer-id demo-customer-001

# With OpenAI (requires OPENAI_API_KEY in .env)
python agents/orchestrator.py --customer-id demo-customer-001 --llm openai

# Start Celery worker for async tasks
celery -A celery_app worker --loglevel=info --concurrency=4
```

---

## 10. Phase 4 — Advanced Agents

> **Status: ✅ Complete**

Phase 4 extends the LangGraph pipeline with two new agents that turn a churn prediction into an executed retention action. After the Explanation Agent finishes, the pipeline now continues through a full decision-and-action loop.

### New graph topology

```
data_intelligence → prediction → explanation → counterfactual → retention_strategist → END
                              ↘ (LOW risk)  → low_risk_terminal → END
```

### 4.1 — Counterfactual Agent (`agents/counterfactual_agent.py`)

The Counterfactual Agent asks: **"What is the minimum change that would prevent this customer from churning?"**

It calls the `counterfactual_tool`, which:
1. Loads the trained model from MLflow (or local pickle)
2. Tries to run the full DiCE engine from Phase 2 to generate diverse counterfactuals
3. Falls back to **rule-based perturbation** if DiCE fails — it perturbs 6 specific features (price, feature adoption, support tickets, NPS, login frequency, contract type) and scores each modified customer with the model
4. Filters by **hard business constraints** (max 30% discount, max $300 spend per customer)
5. **Ranks** surviving interventions by: `prob_reduction / cost / days_to_effect`

The result looks like:
```json
{
  "current_churn_prob": 0.87,
  "n_feasible": 3,
  "interventions": [
    {"action": "Reduce monthly price by 15%",     "new_churn_prob": 0.54, "cost_usd": 45,  "feasibility_score": 0.000042},
    {"action": "Assign dedicated CSM support",    "new_churn_prob": 0.41, "cost_usd": 120, "feasibility_score": 0.000023},
    {"action": "Loyalty upgrade annual contract", "new_churn_prob": 0.38, "cost_usd": 80,  "feasibility_score": 0.000030}
  ]
}
```

### 4.2 — Knapsack Solver (`src/optimization/knapsack_solver.py`)

Not all interventions can be applied at once — there is a **budget constraint**. The knapsack solver picks the most valuable combination of actions within that budget.

This is the classic **0/1 knapsack problem** from computer science:
- Each intervention has a **cost** (weight) and **value** (expected revenue saved = CLV × probability reduction)
- Maximise total value without exceeding the budget
- Uses **PuLP** (integer programming library) for the exact optimal solution
- **Greedy fallback**: sort by value/cost ratio, pick greedily if PuLP is unavailable

Example: budget = $300, three interventions:

| Action | Cost | Value | Select? |
|---|---|---|---|
| Price cut 15% | $45 | $380 | ✅ |
| CSM assignment | $120 | $820 | ✅ |
| Loyalty upgrade | $80 | $640 | ✅ |
| Price cut 25% | $75 | $460 | ❌ (budget hit) |

Total spent: $245 out of $300 budget. Total value: $1,840. ROI = 650%.

### 4.3 — A/B Testing Framework (`src/optimization/ab_testing.py`)

Before executing any retention action, the system assigns the customer to a **treatment** or **control** group. This lets us measure: "Does our intervention actually work?"

- **Deterministic assignment**: same customer always gets the same group (using a seeded hash of `customer_id + experiment_id`)
- **Control group**: the action is logged but NOT dispatched — this customer gets no intervention
- **Treatment group**: the action is executed
- **Analysis**: After 30 days, compare churn rates between groups using:
  - **Mann-Whitney U test** (non-parametric, works for any outcome values)
  - **Chi-squared test** (for binary outcomes: churned / retained)

### 4.4 — Retention Strategist Agent (`agents/retention_strategist.py`)

The Retention Strategist takes the ranked interventions from the Counterfactual Agent and makes the final decision:

1. **Estimates CLV** from monthly charges: `CLV = monthly_charges × 12`
2. **Converts interventions to knapsack items**: value = `CLV × probability_reduction`
3. **Runs KnapsackSolver** to find the optimal subset within the $300 budget
4. **Assigns A/B group** (treatment → action dispatched; control → logged only)
5. **Calls `crm_executor_tool`** with the best action

### 4.5 — Mock CRM Executor (`agents/tools/crm_executor_tool.py`)

The CRM Executor Tool simulates triggering a retention action in a CRM system (like Salesforce or HubSpot). In Phase 4 this is a mock — it logs the action to Redis and returns a structured record.

Supported action types: `price_reduction`, `csm_assignment`, `loyalty_upgrade`, `support_outreach`, `engagement_campaign`

Each execution returns:
```json
{
  "action_id": "crm-a3f7d1b2",
  "customer_id": "C-4821",
  "action_type": "price_reduction",
  "action_description": "Reduce monthly price by 15%",
  "estimated_cost_usd": 45.0,
  "status": "scheduled",
  "ab_group": "treatment",
  "timestamp": "2026-04-15T10:32:00Z"
}
```

### Why dedicated agents for counterfactuals and retention?

The **Counterfactual Agent** needs specialised reasoning about business constraints, feasibility, and customer-specific context. It's not just running DiCE — it's thinking about:
- "Is this customer in a market where price discounts are effective?"
- "Do we have CSM capacity right now?"
- "Has this type of intervention worked for similar customers before?"

The **Retention Strategist Agent** is connected to the optimization layer:

```python
# Agent's internal reasoning (via LLM):
"Customer C-4821 is high-value ($95/month × 3 years tenure = $3,420 LTV).
Current churn probability: 87%.
Best counterfactual: CSM assignment at $120.
Expected value if we intervene:
  - P(stay | CSM) = 72%  →  Expected revenue = $95 × 12 × 0.72 = $820.80
  - Net ROI = $820.80 - $120 = $700.80  ← Worth it
Best counterfactual if we don't intervene:
  - P(stay | no action) = 13%  →  Expected revenue = $95 × 12 × 0.13 = $148.20
  - Net ROI = $148.20 - $0 = $148.20

Decision: Recommend CSM assignment. ROI = +$700.80 vs doing nothing."
```

**Files created in Phase 4:**
```
agents/
├── counterfactual_agent.py        # DiCE + constraint filter + ranking node
├── retention_strategist.py        # Knapsack optimizer + CRM executor node
└── tools/
    ├── counterfactual_tool.py     # LangChain tool wrapping DiCE/perturbation
    └── crm_executor_tool.py       # Mock CRM action dispatcher

src/optimization/
├── __init__.py
├── knapsack_solver.py             # PuLP integer programming + greedy fallback
└── ab_testing.py                  # Deterministic A/B assignment + analysis
```

---

## 11. Phase 5 — Human-in-the-Loop (HITL) + Feedback Loop ✅

### Why humans stay in the loop

AI systems can be confidently wrong. For high-stakes decisions — spending significant budget,
contacting important customers, making company commitments — a human should review and approve
before the action fires.  Phase 5 adds a full HITL gate with Slack notifications, a CSM
feedback system, an immutable audit trail, and an auto-retraining trigger.

### 11.1 — New files

| File | Purpose |
|---|---|
| `agents/hitl_agent.py` | LangGraph HITL gate node |
| `agents/feedback_agent.py` | Records outcomes, triggers retraining |
| `agents/tools/slack_tool.py` | Slack Block Kit notifications |
| `app/hitl_webhook.py` | FastAPI router: decision + feedback endpoints |
| `src/feedback/audit_log.py` | Append-only JSONL audit trail |
| `src/feedback/__init__.py` | Package init |

### 11.2 — Updated graph topology

```
[START]
   │
   ▼
data_intelligence
   │
   ▼
prediction ──(LOW)──► low_risk_terminal ──► [END]
   │
   └─(MEDIUM/HIGH/CRITICAL)─►
   │
   ▼
explanation
   │
   ▼
counterfactual
   │
   ▼
retention_strategist        ← CRITICAL tier: selects actions but SKIPS CRM dispatch
   │                          HIGH/MEDIUM: dispatches CRM immediately
   ▼
hitl                        ← CRITICAL: blocks, waits for Slack approve/reject
   │                          HIGH: sends Slack alert, auto-approves
   │                          MEDIUM: silent auto-approve
   ▼
[END:complete]
```

### 11.3 — HITL Agent behaviour by risk tier

**CRITICAL (≥ 85% churn probability)**
1. `RetentionStrategistAgent` selects optimal actions via knapsack solver but sets `pending_hitl=True` — CRM is **not** dispatched yet
2. `HITLAgent` sends a Slack Block Kit message with customer details, top risk factors, recommended actions, estimated cost and ROI
3. Two buttons: **Approve** and **Reject** — both POST to `POST /hitl/decision` with the `run_id`
4. The agent polls Redis (`churn:hitl:{run_id}:decision`) at 2-second intervals for up to `hitl_timeout_seconds` (default: 30 min)
5. On **approval** → dispatches the CRM action and logs to audit trail
6. On **rejection** → skips CRM dispatch and logs rejection reason
7. On **timeout** → auto-approves with a warning log (configurable)

**HIGH (70–84%)**
1. `RetentionStrategistAgent` dispatches CRM immediately (no gate)
2. `HITLAgent` sends an informational Slack alert (no buttons) and auto-approves

**MEDIUM / LOW**
- MEDIUM passes through HITL silently (auto-approved, no notification)
- LOW never reaches HITL (routed to `low_risk_terminal` before explanation)

### 11.4 — Slack Block Kit message structure

```
┌─────────────────────────────────────────────────────────┐
│ 🚨 HITL Review Required — CRITICAL Churn Risk           │
├────────────────┬────────────────┬──────────┬────────────┤
│ Customer ID    │ Churn Prob     │ Risk Tier│ A/B Group  │
│ a97c0bd3...    │ 91%            │ CRITICAL │ treatment  │
├────────────────┴────────────────┴──────────┴────────────┤
│ Est. Cost: $120.00              Est. ROI: +4.2x         │
├─────────────────────────────────────────────────────────┤
│ Top Risk Factors                                        │
│   • Monthly Charges: +0.312                             │
│   • Feature Adoption Rate: +0.241                       │
│   • Support Tickets (30d): +0.198                       │
├─────────────────────────────────────────────────────────┤
│ Recommended Actions                                     │
│   • Assign dedicated onboarding / CSM support ($120)    │
│   • Proactive support outreach ($60)                    │
├─────────────────────────────────────────────────────────┤
│  [  Approve  ]          [  Reject  ]                    │
├─────────────────────────────────────────────────────────┤
│ Run ID: `abc12345` | Auto-approves after 30 min         │
└─────────────────────────────────────────────────────────┘
```

Falls back to structured console logging (`[SLACK-FALLBACK]`) when no
`SLACK_WEBHOOK_URL` is set — dev/CI mode works without a real Slack workspace.

### 11.5 — FastAPI HITL Webhook (`app/hitl_webhook.py`)

| Route | Method | Purpose |
|---|---|---|
| `/hitl/decision` | POST | CSM approve/reject (direct API call) |
| `/hitl/slack/interactive` | POST | Slack button callback (verified via HMAC-SHA256) |
| `/hitl/feedback` | POST | Record CSM outcome: retained / churned / unknown |
| `/hitl/status/{run_id}` | GET | Poll current HITL decision for a run |
| `/hitl/feedback/stats` | GET | Feedback count vs retraining threshold |
| `/hitl/audit` | GET | Last N audit log entries |

### 11.6 — Feedback Agent & Retraining Trigger

After a retention intervention has had time to take effect, a CSM submits the outcome:

```
POST /hitl/feedback
{
  "run_id": "abc12345",
  "customer_id": "a97c0bd3-...",
  "outcome": "retained",            # or "churned" / "unknown"
  "notes": "Renewed 12-month contract after CSM call",
  "submitted_by": "sarah.jones@company.com",
  "ab_group": "treatment"
}
```

`FeedbackAgent` then:
1. Persists the record to Redis with a 90-day TTL (`churn:feedback:{run_id}:{customer_id}`)
2. Updates the ChromaDB interaction record with the actual outcome
3. Logs the A/B experiment outcome (`ABTestingManager.log_outcome`)
4. Writes an audit log entry
5. Increments the global feedback counter (`churn:feedback:total_count`)
6. If counter ≥ `feedback_retrain_threshold` (default 50): fires a retraining signal
   to Redis (`churn:retrain:trigger`) and resets the counter

Phase 8 will wire the retraining signal to a Celery task that runs the full training pipeline.

### 11.7 — Audit Log (`src/feedback/audit_log.py`)

Every significant system event is recorded as a single JSON line in `logs/audit.jsonl`:

```jsonl
{"event":"hitl_decision","run_id":"abc12345","customer_id":"a97c0bd3-...","risk_tier":"CRITICAL","churn_prob":0.91,"status":"approved","decided_by":"slack:sarah.jones","slack_sent":true,"timestamp":"2026-04-16T10:23:45Z"}
{"event":"crm_action","action_id":"crm-b57dcc4a","customer_id":"a97c0bd3-...","action_type":"csm_assignment","status":"scheduled","ab_group":"treatment","cost_usd":120.0,"timestamp":"2026-04-16T10:23:46Z"}
{"event":"feedback_recorded","feedback_id":"fb-abc12345-1745...","run_id":"abc12345","customer_id":"a97c0bd3-...","outcome":"retained","submitted_by":"sarah.jones@company.com","timestamp":"2026-05-16T14:00:00Z"}
```

The file is append-only — entries are never modified or deleted. `AuditLog.read_recent(n=50)` returns the tail of the file for the `/hitl/audit` endpoint.

### 11.8 — Configuration

New settings in `config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `SLACK_WEBHOOK_URL` | `""` | Slack incoming webhook URL |
| `SLACK_SIGNING_SECRET` | `""` | App signing secret for webhook signature verification |
| `HITL_TIMEOUT_SECONDS` | `1800` | Seconds before auto-approval (30 min) |
| `FEEDBACK_RETRAIN_THRESHOLD` | `50` | Feedback records needed to trigger retraining |

---

## 12. Phase 6 — The Full-Stack Application ✅

Phase 6 wraps the entire AI pipeline in a production web application: a FastAPI backend serving REST + WebSocket APIs, and a Next.js 14 frontend that business users interact with in real time.

### 12.1 FastAPI Backend

**What is FastAPI?**

FastAPI is a Python web framework for building APIs — the communication layer between your frontend (what users see) and your backend (where logic runs).

```
User clicks "Analyse Customer" on website
         ↓
HTTP POST /api/v1/agent/analyse
         ↓
FastAPI validates JWT Bearer token
         ↓
Dispatches ChurnOrchestrator as Celery async task
         ↓
Returns run_id immediately
         ↓
Frontend opens WebSocket /ws/agent/{run_id}
         ↓
Server streams step events as they complete
         ↓
Final payload (prediction + explanation + plan) delivered
```

#### Backend files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app with lifespan model pre-warm, CORS, GZip |
| `app/middleware/auth.py` | JWT Bearer validation, `CurrentUser` dependency |
| `app/routers/auth.py` | `POST /auth/token` — OAuth2 login, demo users |
| `app/routers/customers.py` | Customer list, high-risk watchlist, single customer |
| `app/routers/agent.py` | Trigger analysis, poll status, batch endpoint |
| `app/websockets/agent_stream.py` | `GET /ws/agent/{run_id}` — live step streaming |
| `app/hitl_webhook.py` | HITL decision, Slack callback, feedback, audit |

**JWT Authentication**
```
JWT = JSON Web Token
User logs in → server creates signed token → user sends token with every request
Server validates token signature → if valid, processes request

Structure: [Header].[Payload].[Signature]
Payload contains: username, role, expiry_timestamp

Demo credentials:
  admin / admin123   → role: admin
  analyst / analyst123 → role: analyst
```

**WebSocket Streaming**
```
Normal HTTP: Request → Wait → Full Response (frozen spinner for 30s)
WebSocket:   Connection open → server pushes updates in real-time

Event types streamed:
  { step: "prediction",   status: "running" }
  { step: "prediction",   status: "completed" }
  { step: "explanation",  status: "running" }
  ...
  { step: "final", status: "final",
    prediction: {...}, explanation: {...},
    retention_plan: {...}, hitl_decision: {...} }
```

**API endpoints summary**
```
POST /auth/token                       Log in, receive JWT
GET  /auth/verify                      Validate token

GET  /api/v1/customers                 List all customers (paginated)
GET  /api/v1/customers/high-risk       Top churned by monthly charges
GET  /api/v1/customers/{id}            Single customer + last analysis

POST /api/v1/agent/analyse             Start pipeline (async Celery)
GET  /api/v1/agent/status/{run_id}     Poll pipeline status
POST /api/v1/agent/batch               Batch analysis (max 100)
GET  /api/v1/agent/batch/{task_id}     Poll batch progress

WS   /ws/agent/{run_id}                Live step event stream

POST /hitl/decision                    CSM approve/reject
POST /hitl/slack/interactive           Slack button callback
POST /hitl/feedback                    Record outcome
GET  /hitl/audit                       Audit log (last N entries)
GET  /hitl/feedback/stats              Feedback counts + A/B breakdown
GET  /health                           Health check
```

---

### 12.2 Next.js 14 Frontend

**What is Next.js?**

Next.js is a React framework for building web applications. It powers the dashboard that business users interact with.

#### Frontend architecture

```
frontend/
├── src/
│   ├── app/                          App Router (Next.js 14)
│   │   ├── layout.tsx                Root layout + AppShell
│   │   ├── page.tsx                  Dashboard — high-risk watchlist
│   │   ├── login/page.tsx            JWT login form
│   │   ├── customers/[id]/page.tsx   Customer detail + live pipeline
│   │   └── hitl/page.tsx             HITL audit log + feedback stats
│   ├── components/
│   │   ├── AppShell.tsx              Conditional nav (hides on /login)
│   │   ├── Navbar.tsx                Dark sidebar with route links
│   │   ├── PipelineStream.tsx        WebSocket step progress + results
│   │   ├── RetentionPlanCard.tsx     Cost / ROI / actions display
│   │   └── RiskBadge.tsx             Color-coded CRITICAL/HIGH/MEDIUM/LOW
│   ├── lib/
│   │   └── api.ts                    Typed API client (all endpoints)
│   └── types/
│       └── index.ts                  Shared TypeScript interfaces
```

#### The pages

**Login (`/login`)**
- JWT form: username + password → `POST /auth/token`
- Stores token in `localStorage`, redirects to dashboard
- Demo credentials shown for portfolio demo use

**Dashboard (`/`)**
- Fetches top 20 high-risk customers via `GET /api/v1/customers/high-risk`
- KPI strip: watched count, feedback collected, retention rate
- Table with risk tier badge — click any row → `/customers/{id}`

**Customer Analysis (`/customers/{id}`)**
- Shows customer profile card (monthly charges, support tickets, NPS, adoption)
- "Run Churn Analysis" button → `POST /api/v1/agent/analyse` → opens WebSocket
- `PipelineStream` renders live step progress for all 6 pipeline stages:
  Data Intelligence → Prediction → SHAP Explanation →
  Counterfactual → Retention Strategy → HITL Review
- Final results appear inline: probability bar, narrative, SHAP factors, plan
- HITL quick-action panel for CRITICAL customers (Approve / Reject buttons)
- Outcome feedback form (Retained / Churned / Unknown) after analysis completes

**HITL Queue (`/hitl`)**
- Feedback statistics: total, retained, churned, unknown counts
- A/B group breakdown (control vs treatment)
- Append-only audit log table: colour-coded event types, run IDs, timestamps

#### Key frontend patterns

**Auth guard** — every protected page checks `getToken()` on mount; redirects to `/login` if missing; `api.ts` auto-redirects on any 401 response.

**Optimistic streaming** — `PipelineStream` opens a WebSocket immediately after `run_id` is received; no polling required; all step state held locally in `useState`.

**Tailwind CSS** — utility-first styling with no external UI library; consistent slate/indigo colour palette throughout.

---

## 13. Phase 7 — Optimization Engine + Fairness Layer ✅

Phase 7 adds four advanced analytical capabilities: a portfolio-level budget optimizer, temporal survival analysis, a fairness/bias detection framework, and model robustness testing. These are exposed through new API endpoints and two new frontend pages (Optimizer and Model Intel).

### 13.1 Retention Budget Optimizer

**Files:** `src/optimization/knapsack_solver.py`, `app/routers/analytics.py`, `frontend/src/app/optimization/page.tsx`

#### The business problem

You have a $50,000 quarterly retention budget and 800 at-risk customers. You can't afford to intervene on all of them. Which customers get which actions to maximise retained revenue?

This is the classic **0/1 Knapsack Problem**:

```
Knapsack = Total Budget
Items    = (Customer × Action) pairs
Weight   = Cost of each action
Value    = Expected revenue saved = CLV × prob_reduction

Goal: Assign actions to customers to maximise total expected
      revenue retained without exceeding the budget.
```

**Example output:**
```
Customer C-4821: dedicated_csm ($150) → −20% churn → $1,840 saved  ROI: 12×
Customer C-7193: discount_10pct ($20)  → −8% churn  → $380 saved   ROI: 19×
...
Total: $48,920 spent → $142,000 expected revenue retained  ROI: 2.9×
```

Solved using **PuLP** integer programming (falls back to greedy sort if solver unavailable). The frontend Optimizer page lets you set the budget slider, max actions per customer, and risk tier filter interactively.

### 13.2 Temporal & Survival Analysis

**Files:** `src/temporal/survival_analysis.py`, `src/temporal/seasonality.py`, `src/temporal/cohort_analysis.py`

#### Kaplan-Meier survival curves

KM curves answer: "Of 1,000 customers acquired in January, how many are still active after N months?"

```python
kmf = KaplanMeierFitter()
kmf.fit(tenure_months, event_observed=churned)
# median_survival_time_ = 34 months (half of customers have churned by month 34)
```

Survival curves can be stratified by `contract_type`, `plan_tier`, etc. to reveal which segments are most at risk over time.

#### Cox Proportional Hazards

The Cox PH model quantifies *which features most accelerate churn*:

```
Hazard Ratio > 1 = this feature INCREASES churn rate
Hazard Ratio < 1 = this feature DECREASES churn rate

monthly_charges:  HR = 1.42 (p<0.001) → higher charges → faster churn
nps_score:        HR = 0.71 (p<0.001) → higher NPS → slower churn
num_support_tickets: HR = 1.28 (p=0.003) → more tickets → faster churn
```

#### Seasonality decomposition (STL)

Uses statsmodels STL to decompose the monthly churn-rate time-series into:
- **Trend** — long-term direction (churn going up or down?)
- **Seasonal** — month-of-year patterns (does churn spike in Q1 budget cycles?)
- **Residual** — unexplained noise

#### Cohort retention heatmap

Groups customers by acquisition month, then tracks the % still active at each subsequent month. Displayed as a colour-coded heatmap: green = high retention, red = high churn. Reveals product/onboarding improvements or degradations over cohorts.

### 13.3 Fairness & Bias Detection

**Files:** `src/fairness/bias_detector.py`, `src/fairness/fairness_report.py`

#### Why fairness matters in churn prediction

If the model was trained on biased historical data, it may systematically under-predict churn for certain customer segments, causing the retention team to overlook them:

```
Bad outcome example:
  Predicted churn rate: Enterprise = 18%, SMB = 12%
  Actual churn rate:    Enterprise = 17%, SMB = 22%
  
  → The model underestimates SMB churn by 10pp
  → SMB customers who should get retention calls don't get them
  → They churn → lost revenue + unfair treatment
```

#### Three fairness metrics

| Metric | What it measures | Threshold |
|---|---|---|
| **Demographic Parity Difference** | Max difference in *predicted positive rate* across groups | < 0.10 |
| **Equalized Odds Difference** | Max difference in *true positive rate* across groups (requires labels) | < 0.10 |
| **Disparate Impact Ratio** | min_group_rate / max_group_rate (EEOC 4/5ths rule) | > 0.80 |

The fairness report is available as JSON (API) or HTML (browser), with per-segment breakdown and clear PASS/FAIL per attribute.

### 13.4 Model Robustness Testing

**File:** `src/robustness/robustness_tester.py`

Four tests run against the deployed model:

| Test | Method | Pass threshold |
|---|---|---|
| **Prediction stability** | Add Gaussian noise to features; measure Spearman rank correlation between original and noisy predictions | ≥ 0.85 |
| **Feature perturbation sensitivity** | Shift each feature by ±10% of its std; measure mean absolute prob change | (reported, no threshold) |
| **Adversarial stress** | 2σ targeted perturbation in worst-case direction; report max shift | ≤ 0.20 |
| **Calibration ECE** | Expected Calibration Error across 10 probability bins | ≤ 0.10 |

A high overall robustness score (> 75%) means the model's predictions are trustworthy and consistent — not erratic near decision boundaries.

### 13.5 New API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/v1/analytics/survival` | KM curves + Cox PH hazard ratios |
| `GET /api/v1/analytics/cohort` | Monthly cohort retention matrix |
| `GET /api/v1/analytics/seasonality` | STL decomposition of churn time-series |
| `GET /api/v1/analytics/fairness` | JSON bias report for protected attributes |
| `GET /api/v1/analytics/fairness/report` | HTML fairness report |
| `GET /api/v1/analytics/robustness` | Robustness + calibration report |
| `POST /api/v1/analytics/optimize` | Portfolio budget optimizer |

### 13.6 New Frontend Pages

**Optimizer page** (`/optimization`): Budget slider + risk tier filter + max actions toggle → run knapsack optimizer → results table showing customer/action/cost/churn-reduction/expected-revenue-saved with KPI tiles for total ROI.

**Model Intelligence page** (`/models`): Four-tab interface:
- **Fairness** — per-attribute bias checks with segment breakdown
- **Robustness** — stability score, feature sensitivity bars, calibration table
- **Survival Analysis** — KM table, Cox PH hazard ratios, 12-month churn projection bar chart
- **Cohort Retention** — colour-coded heatmap + period churn rate bar chart

---

## 14. Phase 8 — Deployment & Monitoring

### 14.1 Docker for Development

Already covered in Phase 0. `docker compose up` spins up the entire stack locally.

### 14.2 Kubernetes for Production

**What is Kubernetes?**

If Docker is "run this container", Kubernetes is "run this container, keep 3 copies of it running at all times, automatically replace any that crash, scale up to 10 copies during peak load, and route traffic across all healthy copies."

```
Kubernetes Cluster (e.g., on AWS/GCP/Azure)
├── FastAPI Pods (3 replicas)
│   └── Auto-scales to 10 during peak
├── Celery Worker Pods (2 replicas)
├── Frontend (Next.js) Pod (2 replicas)
└── Monitoring Stack
    ├── Prometheus (collects metrics)
    └── Grafana (visualises metrics)
```

### 14.3 GitHub Actions CI/CD

**CI = Continuous Integration:** Every time code is pushed, automated tests run.

**CD = Continuous Deployment:** If tests pass, the new version is automatically deployed.

```
Developer pushes code to GitHub
         ↓
GitHub Actions triggers automatically
         ↓
┌──────────────────────────────────┐
│ CI Pipeline:                     │
│  1. Install dependencies         │
│  2. Run linter (flake8)          │
│  3. Run type checker (mypy)      │
│  4. Run unit tests (pytest)      │
│  5. Run integration tests        │
│  6. Build Docker image           │
└──────────────┬───────────────────┘
               │ All checks pass?
               ▼
┌──────────────────────────────────┐
│ CD Pipeline:                     │
│  1. Push image to container registry│
│  2. Update Kubernetes manifests  │
│  3. Rolling deployment (0 downtime)│
│  4. Health check new pods        │
│  5. Alert if deployment fails    │
└──────────────────────────────────┘
```

### 14.4 Prometheus + Grafana Monitoring

**Prometheus** collects metrics from all services:
```
churn_predictions_total = 15,283
churn_prediction_latency_p99 = 145ms
model_auc_roc_current = 0.891
agent_workflow_duration_p95 = 28.3s
active_websocket_connections = 47
redis_cache_hit_rate = 0.94
```

**Grafana** visualises these metrics in real-time dashboards, and sends alerts when metrics go out of range:

```
ALERT: model_auc_roc_current dropped below 0.85
ACTION: Page on-call engineer + trigger retraining pipeline
```

---

## 15. Technology Stack Explained

| Technology | Category | What it does | Analogy |
|---|---|---|---|
| **Python 3.14** | Language | Runs everything | The language you speak |
| **Pandas** | Data | Manipulates tables of data | Excel but in code |
| **NumPy** | Math | Fast array/matrix operations | Calculator on steroids |
| **Scikit-learn** | ML | ML algorithms + pipelines | Swiss Army knife for ML |
| **XGBoost / LightGBM** | ML | Gradient boosting models | Specialized prediction engines |
| **SHAP** | XAI | Shapley value explanations | "Here's why I said that" |
| **LIME** | XAI | Local surrogate explanations | "In this neighbourhood, here's the rule" |
| **DiCE** | XAI | Counterfactual generation | "Here's what needs to change" |
| **MLflow** | MLOps | Experiment tracking + model registry | Lab notebook + version control for models |
| **Optuna** | MLOps | Hyperparameter optimisation | Auto-tuner for model settings |
| **MAPIE** | MLOps | Conformal prediction intervals | "Here's my confidence range" |
| **LangChain** | AI Agents | LLM orchestration framework | Translator between Python and LLMs |
| **LangGraph** | AI Agents | Multi-agent state machines | Org chart for AI agents |
| **LangSmith** | AI Agents | Agent observability / tracing | Security camera for AI reasoning |
| **OpenAI / Anthropic** | LLMs | Language model APIs | The AI "brains" |
| **ChromaDB** | Vector DB | Semantic search for agent memory | AI's searchable notebook |
| **FastAPI** | Backend | Python web framework for APIs | Traffic manager for web requests |
| **Celery** | Backend | Distributed task queue | Background job worker |
| **PostgreSQL** | Database | Relational database | Organised filing cabinet |
| **Redis** | Cache/Queue | In-memory fast storage | Working desk (vs filing cabinet) |
| **Next.js** | Frontend | React-based web framework | The website users see |
| **TypeScript** | Frontend | Typed JavaScript | JavaScript with safety nets |
| **Tailwind CSS** | Frontend | Utility-first CSS framework | Pre-made clothing for web pages |
| **shadcn/ui** | Frontend | React component library | Pre-built UI furniture |
| **Recharts** | Frontend | React charting library | Chart factory |
| **Socket.IO** | Realtime | WebSocket communication | Persistent phone line (vs letter) |
| **Pydantic** | Validation | Data validation and settings | Contract enforcer |
| **Docker** | DevOps | Container packaging | Shipping containers for software |
| **Kubernetes** | DevOps | Container orchestration | Shipping fleet manager |
| **Prometheus** | Monitoring | Metrics collection | Health sensors |
| **Grafana** | Monitoring | Metrics visualisation | Hospital monitor screen |
| **GitHub Actions** | CI/CD | Automated testing + deployment | Quality control + auto-delivery |
| **PuLP** | Optimisation | Linear programming solver | Mathematical budget allocator |
| **Structlog** | Logging | Structured JSON logging | System diary |
| **Pandera** | Validation | DataFrame schema validation | Data quality inspector |
| **Scipy** | Science | Statistical tests | Stats textbook in code |

---

## 16. How Everything Connects

Let's trace a single complete journey through the entire system:

### A day in the life of Customer C-4821

```
Monday, 9:00 AM — Data Collection
─────────────────────────────────
CRM system exports overnight customer activity data to:
  data/raw/customers_2025_04_15.csv

9:01 AM — Data Ingestion
─────────────────────────
Preprocessing pipeline runs automatically (Celery scheduled task):
  1. Pandera validates schema → all 1,247 new records valid
  2. Feature engineering adds derived columns
  3. IQR capping, encoding, scaling applied
  4. Features written to PostgreSQL feature store

9:05 AM — Drift Check
──────────────────────
DriftDetector runs KS-test on 15 features:
  → login_frequency: p=0.003 → DRIFTED ⚠
  → usage_30d: p=0.041 → DRIFTED ⚠
  → Other 13 features: no drift
  Report saved: reports/explainability/drift_2025_04_15.html
  [No retraining triggered — only 2/15 features drifted]

9:06 AM — Batch Prediction
────────────────────────────
PredictionAgent runs ensemble on all 1,247 customers:
  C-4821: churn_prob=0.87, interval=[0.82, 0.91] → HIGH RISK

9:07 AM — High-Risk Customers Queued for Analysis
────────────────────────────────────────────────────
127 customers with prob > 0.70 added to Celery task queue

9:08 AM — C-4821 Full Agent Workflow Triggered
───────────────────────────────────────────────
[LangSmith traces every step for auditability]

  Orchestrator: "Analyse C-4821 — high priority"
  
  DataIntelligenceAgent:
  → Fetches full feature vector from PostgreSQL
  → Checks for anomalies: support_tickets=7 (3σ above mean) ⚠
  → Returns: customer_profile dict
  
  PredictionAgent:
  → Runs stacking ensemble: 0.87
  → XGB: 0.89, LGBM: 0.85, RF: 0.82 (all agree — confident!)
  → Conformal interval: [0.82, 0.91], is_uncertain=False
  → Returns: prediction dict
  
  ExplanationAgent:
  → SHAP: top drivers = [days_since_login, tickets, usage_decline]
  → LIME: corroborates SHAP, agreement_score=0.81
  → GPT-4o generates narrative (3 paragraphs, plain English)
  → Returns: explanation dict
  
  CounterfactualAgent:
  → DiCE generates 10 counterfactuals
  → 3 pass business constraints (cost < $300, discount < 30%)
  → Ranked by impact/cost ratio
  → Top 3: CSM ($120), Price cut ($51), Contract upgrade ($80)
  → Returns: interventions dict
  
  RetentionStrategistAgent:
  → ROI calculation: CSM gives 583% ROI
  → Knapsack check: budget has room → recommend CSM
  → Similar past cases: 74% success rate
  → Returns: recommendation dict

9:11 AM — HITL Decision
─────────────────────────
HITLAgent evaluates: prob=0.87 > 0.70 threshold AND cost=$120 > $100
→ Sends Slack message to Sarah Jones (Account Manager):

  "⚠ C-4821 (Acme Corp) — 87% churn risk
   Recommended: Assign dedicated CSM (Cost: $120, ROI: 583%)
   Confidence: High (both SHAP and LIME agree)
   Similar cases: 74% retained with this approach
   [Approve ✓] [Reject ✗] [View Details 🔍]"

9:14 AM — Sarah Reviews on Dashboard
───────────────────────────────────────
Sarah opens the web dashboard:
  → SHAP waterfall chart confirms usage drop and support tickets
  → Counterfactual shows: CSM assignment reduces risk to 24%
  → She's seen this pattern before with other Enterprise customers
  → Sarah clicks "Approve"

9:14 AM — Approval Logged
──────────────────────────
FeedbackAgent logs:
  {customer_id: C-4821, action: CSM_assignment, approver: sarah.jones,
   timestamp: 2025-04-15T09:14:23Z, predicted_prob: 0.87}

Schedule: Check outcome on 2025-05-15

9:15 AM — Real-Time Dashboard Update
──────────────────────────────────────
WebSocket pushes update to all connected dashboard users:
  "C-4821 intervention approved — moved to 'In Progress' queue"

May 15, 2025 — Outcome Tracking
─────────────────────────────────
FeedbackAgent checks: Did C-4821 renew?
  → PostgreSQL query: C-4821 paid invoice on May 10 ✓ RETAINED
  
Logged as: intervention_outcome = SUCCESS

Model performance tracking:
  Recent 30-day precision: 0.79 (above 0.75 threshold — no retraining needed)
```

---

## 17. Why This Project Stands Out

### Comparison with existing tools

| Feature | Basic Churn Model | Typical SaaS Tools | **This Platform** |
|---|---|---|---|
| Prediction | ✓ Single model | ✓ Basic ML | ✓ 5-model ensemble |
| Uncertainty | ✗ | ✗ | ✓ Conformal intervals |
| Explainability | ✗ | ✗ Sometimes | ✓ SHAP + LIME + DiCE |
| Counterfactuals | ✗ | ✗ | ✓ With business constraints |
| Agentic AI | ✗ | ✗ | ✓ 8-agent LangGraph system |
| Human-in-the-Loop | ✗ | Sometimes | ✓ With full context + Slack |
| Feedback loop | ✗ | ✗ | ✓ Outcome tracking + auto-retrain |
| Budget optimisation | ✗ | ✗ | ✓ Knapsack solver |
| Fairness analysis | ✗ | ✗ | ✓ Multiple fairness metrics |
| Real-time streaming | ✗ | Sometimes | ✓ WebSocket agent streaming |
| Full observability | ✗ | ✗ | ✓ LangSmith + Prometheus + Grafana |
| Production-grade | ✗ | ✓ | ✓ Docker + K8s + CI/CD |

### The five design principles that make this different

**1. Explain every decision**
The system never says "trust me". Every prediction comes with evidence, every recommendation comes with reasoning, every intervention comes with expected ROI.

**2. Keep humans in control**
AI handles the analysis and recommendations. Humans make the important calls. The system augments human judgment rather than replacing it.

**3. Learn from outcomes**
The system tracks whether its interventions actually worked, and uses that knowledge to improve. Most ML systems are frozen in time — this one evolves.

**4. Optimise holistically**
Individual predictions are combined with budget constraints, business capacity, and fairness considerations to make portfolio-level decisions — not just per-customer decisions in isolation.

**5. Production-grade from day one**
Logging, monitoring, drift detection, CI/CD, authentication, rate limiting — all the boring-but-critical pieces that turn a prototype into a system that can actually run in a business.

---

## Closing Thoughts

This platform represents the state of the art in how AI should be applied to business problems:

- Not just "predict and dump" — but **predict, explain, act, and learn**
- Not just automation — but **human-AI collaboration** with appropriate trust levels
- Not just a notebook — but a **production system** designed to run reliably at scale

Every technology in this stack exists to solve a real problem. Nothing is included just because it's trendy. And every design decision — from the conformal prediction intervals to the Knapsack optimizer — exists because real businesses have real constraints that a naive model ignores.

If you understand this entire system, you understand:
- How modern machine learning actually works (beyond "AI is magic")
- How production software systems are built and deployed
- How AI agents reason and collaborate
- How businesses make data-driven decisions at scale
- Why explainability and fairness are non-negotiable in serious AI applications

That's a genuinely comprehensive foundation in applied AI engineering.

---

*Built as a portfolio project demonstrating: Agentic AI · Production ML · Explainable AI · Full-Stack Development · MLOps · DevOps*
