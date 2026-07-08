# RiftRace

Illicit Bitcoin transaction detection on the real-world Elliptic dataset, combining a classical ML classifier with a Gen AI layer for retrieval-grounded, multi-agent investigation reports.

## Live Demo

**[https://riftrace.vercel.app](https://riftrace.vercel.app)**

> Hosted on free tier — first load may take ~60s while the backend wakes up.

## Screenshot

> _Add a screenshot of the app here._

## Key Features

- **Hybrid detection system**: a classical Random Forest classifier (F1 = 0.787, ROC-AUC = 0.929 on a temporal held-out test split) handles the actual illicit/licit prediction, while a Gen AI layer sits on top for retrieval, orchestration, and natural-language explanation — the model never depends on the LLM to produce a verdict.
- **Retrieval-grounded explanations**: risk assessments are grounded in a small curated knowledge base of real AML (anti-money-laundering) typologies, not free-floating LLM speculation.
- **Multi-agent investigation reports**: a LangGraph state machine chains deterministic analysis with LLM reasoning steps to produce a structured investigator-style report per transaction.
- **Transaction graph exploration**: look up any transaction's directly connected neighbors and their known class, to see the surrounding money-flow context.
- **MCP server**: the same prediction/graph/investigation capabilities are also exposed as MCP tools for use by MCP-compatible AI clients, independent of the HTTP API.

## Architecture

**Phase A — Retrieval-Augmented Explanation (RAG).** `POST /explain/{tx_id}` grounds a single LLM call in a small knowledge base (`aml_patterns.json`) of money-laundering typologies. Retrieval uses TF-IDF vectors (not a neural embedding model — see note below) indexed with FAISS (`IndexFlatIP` over normalized vectors, i.e. cosine similarity).

**Phase B — Multi-agent investigation (LangGraph).** `POST /investigate/{tx_id}` runs a `StateGraph` of four nodes sharing one state object:

```
ANALYZE -> RETRIEVE -> REASON -> RECOMMEND -> END
```

`ANALYZE` and `RETRIEVE` are deterministic (model prediction, graph neighbors, knowledge-base retrieval — no LLM call, never fail on a missing API key). `REASON` and `RECOMMEND` are LLM steps; if `GROQ_API_KEY` isn't configured, they degrade to a clear non-grounded message instead of failing the request.

**Phase C — MCP server.** `backend/mcp_server/server.py` exposes `predict_transaction`, `get_transaction_graph`, `get_dataset_stats`, and `investigate_transaction` as MCP tools over stdio, reusing the same service layer and Pydantic schemas as the HTTP API, so tool output is guaranteed to match the API's shape.

> Why TF-IDF and not a neural embedding model: the original design used `sentence-transformers/all-MiniLM-L6-v2`, but on a 17-document knowledge base that pulled in `torch` for ~390MB of RAM just to embed a corpus small enough to fit in a spreadsheet — enough to blow past Render's 512MB free-tier limit. A TF-IDF vectorizer fitted offline and committed as a small artifact gives the same retrieval behavior at effectively zero extra runtime memory, reusing scikit-learn (already a hard dependency).

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI (Python), deployed on [Render](https://render.com) |
| Frontend | React 19 + Vite, deployed on [Vercel](https://vercel.com) |
| ML model | scikit-learn Random Forest (served); GCN/GraphSAGE also trained for comparison (see Notable Finding) |
| Feature storage | SQLite, queried per-request rather than held in memory; hosted as a GitHub Release asset (~153MB) and pulled at build time, since it's too large for Git LFS on Render's free build environment |
| Retrieval | TF-IDF + FAISS (`IndexFlatIP`) over a curated AML typology knowledge base |
| Multi-agent orchestration | LangGraph `StateGraph` |
| LLM | [Groq](https://groq.com) (`llama-3.3-70b-versatile`), used only for the natural-language reasoning/recommendation steps — never for classification |
| Tool interface | FastMCP (Model Context Protocol server) |

## Dataset

The [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set): 203,769 transaction nodes, 234,355 directed edges, 49 time steps, 165 features per node (94 local + 71 aggregated from one-hop neighbors). Roughly 2% of labeled nodes are illicit, ~21% licit, and the rest unknown — evaluation and the served model's metrics are computed on a **temporal** split (train on early time steps, test on steps 35–49), not a random resample, since a random split would leak future information into training.

## Notable Finding: GNN Underperformance

Two graph neural networks (GCN and GraphSAGE) were trained and evaluated on the same temporal split alongside the classical baselines. Both underperformed the Random Forest baseline on the illicit-class F1/ROC-AUC metrics:

| Model | F1 (illicit) | ROC-AUC |
|---|---|---|
| Random Forest (served) | 0.787 | 0.929 |
| GraphSAGE | 0.599 | 0.880 |
| GCN | 0.558 | 0.864 |

This is consistent with published findings on the Elliptic dataset showing GNN performance degrading under temporal drift: the graph's structure and feature distributions shift across time steps in ways that hurt models relying heavily on neighborhood aggregation, more than they hurt a feature-based classifier. This is why the Random Forest, not a GNN, is the model actually served in production.

Full precision/recall/F1/ROC-AUC/accuracy for all five trained models (including Logistic Regression and XGBoost) are in [`results/model_comparison.md`](results/model_comparison.md), reproducible by running `python backend/models/evaluate.py`.

## Local Setup

### Prerequisites

- Python 3.11+ (developed on 3.14)
- Node.js 18+
- The raw Elliptic dataset CSVs (see `backend/data_pipeline/README.md` for the expected files) placed in `backend/data/raw/`

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Set your own Groq API key as an environment variable (optional — the API and UI work without it, only the LLM-generated explanations degrade to a fallback message without it). This is a real environment variable, not a `.env` file — nothing in the app loads one:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY = "your-own-groq-api-key"

# macOS/Linux
export GROQ_API_KEY="your-own-groq-api-key"
```

Run the data pipeline once to build the processed dataset, feature store, graph cache, and knowledge-base embeddings (see `backend/data_pipeline/README.md` and `backend/models/` for each step), then start the API:

```bash
uvicorn api.main:app --reload --port 8000
```

The API validates all required model/data files exist at startup and refuses to serve traffic if any are missing, so a broken setup fails immediately with a clear error rather than 500ing on the first real request.

### Frontend

```bash
cd frontend
npm install
```

Point the frontend at your backend. Vite bakes this in at build time, so it must be set before `npm run build` (or in a `.env` / `.env.local` file for `npm run dev`):

```
VITE_API_BASE_URL=http://localhost:8000
```

Then:

```bash
npm run dev      # local development server
npm run build    # production build to dist/
```

### MCP server

```bash
cd backend
python -m mcp_server.server
```

Runs as a stdio MCP server for use by MCP-compatible clients — a separate process from the FastAPI app, sharing the same service layer.
