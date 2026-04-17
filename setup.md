# Setup Guide: Churn Intelligence Platform

Welcome to the **Customer Churn Intelligence Platform**. This is a powerful, containerized application stack consisting of a Python FastAPI backend, a Celery-based AI worker pipeline, a PostgreSQL + Redis + ChromaDB data tier, and a Next.js (React) front-end dashboard. 

This guide details how to get everything up and running on a local development environment.

## Prerequisites

Before starting, ensure you have the following installed on your machine:
*   **Git** (for cloning the repository)
*   **Docker & Docker Compose** (highly recommended for running infrastructure dependencies)
*   **Node.js 20.x+** and `npm` (required for local frontend development)
*   **Python 3.12+** (required for local backend development)

---

## 1. Environment Configuration

Regardless of whether you choose to run via Docker or run processes locally, you must provide your environment variables.

1.  Open a terminal in the root directory: `churn-intelligence-platform/`
2.  Create a `.env` file from the example template:
    ```bash
    cp .env.example .env
    ```
    *(On Windows Command Prompt, use `copy .env.example .env`)*
3.  Open the `.env` file and populate the required API keys (e.g., OpenAI, Anthropic, Gemini API keys) and verify the connection strings for PostgreSQL, Redis, and ChromaDB.

---

## 2. Docker Deployment (Recommended)

The easiest way to spin up the entire application stack—including the API, the frontend console, the Celery worker, and all background databases / infrastructure services—is via Docker Compose.

> [!NOTE] 
> The Dockerfiles are located inside the `docker/` folder, but they build using the root `churn-intelligence-platform/` directory as the build context.

1.  Navigate to the `docker/` subdirectory:
    ```bash
    cd docker
    ```
2.  Trigger the build and start the containers in detached (`-d`) mode:
    ```bash
    docker-compose up -d --build
    ```
3.  **Access the Platform:**
    *   **Frontend Dashboard:** [http://localhost:3000](http://localhost:3000)
    *   **Backend API:** [http://localhost:8000](http://localhost:8000)
    *   **API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   **Grafana Monitoring:** [http://localhost:3001](http://localhost:3001)

4.  **Tear Down:**
    To spin down all services and optionally remove attached volumes:
    ```bash
    docker-compose down
    # To also prune persistence volumes:
    docker-compose down -v
    ```

---

## 3. Local Development Setup (Manual)

If you intend to write code on the primary services (Backend API or Next.js Frontend) and want to use local hot-reloading outside of Docker configurations, follow these instructions to run the services individually.

> [!TIP]
> You can mix and match. A common pattern is to spin up the infrastructure layer using `docker-compose up -d churn_postgres churn_redis churn_chromadb` and execute the Frontend and Backend natively.

### 3A. Backend (FastAPI) & Celery Worker

1.  **Virtual Environment Setup:** From the root folder, create and activate a Python virtual environment to isolate your dependencies.
    ```bash
    python -m venv .venv
    
    # On Windows:
    .\.venv\Scripts\activate
    
    # On macOS/Linux:
    source .venv/bin/activate
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start the Local API Server:**
    ```bash
    python -m uvicorn app.main:app --reload --port 8000
    ```
4.  **Start the Celery Worker:** (Open a separate terminal, activate the `.venv`, then run this. Ensure Redis is running!)
    ```powershell
    python -m celery -A celery_app.celery_app worker --loglevel=info
    ```
    > [!NOTE]
    > **Windows only:** Use `python -m celery` instead of calling `celery` directly. On Windows (especially with OneDrive-synced paths), `celery.exe` in the `.venv\Scripts\` folder may be blocked by Windows security policy with "Access is denied".
    > The app argument is `celery_app.celery_app` — `celery_app` (the filename) dot `celery_app` (the Celery instance variable name inside the file).

### 3B. Frontend (Next.js)

1.  **Navigate to the Frontend Directory:**
    ```bash
    cd frontend/
    ```
2.  **Install Node Modules:**
    ```bash
    npm install
    ```
3.  **Start the Development Server:**
    ```bash
    npm run dev
    ```
4.  The locally hosted frontend will stream live on [http://localhost:3000](http://localhost:3000).

> [!WARNING]
> **EADDRINUSE Error on Port 3000:**
> If you start `npm run dev` and get an error saying `address already in use :::3000`, process cleanup may be required. 
> * On Windows, search for the process using `netstat -ano | findstr :3000` and forcefully terminate the matching Process ID (`taskkill /PID <PID> /F`).
> * Alternatively, start Next.js on a different port: `npm run dev -- --port 3005`.

---

## Technical Notes
*   **Docker Ignore Profiles:** Notice the `.dockerignore` file in the root folder, as well as specific `docker/Dockerfile.*.dockerignore` files. These strictly block `node_modules` and `.venv` from being pushed into the Docker Daemon cache during builds to maintain performance. Do not delete them.
*   **Machine Learning Output:** `mlruns/` and database volumes are mapped locally so models and runs persist between sessions.
