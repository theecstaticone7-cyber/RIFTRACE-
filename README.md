# RiftRace

Agentic graph intelligence platform for detecting illicit Bitcoin transactions
using the Elliptic dataset.

## Structure

- `backend/` — Python/FastAPI backend, graph data pipeline, and ML models
- `frontend/` — React frontend

## Status

Phase 0: repo scaffolding and dataset inspection only. No model training, API
routes, or frontend code yet.

## Backend setup

```
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Dataset

Place the raw Elliptic dataset CSVs in `backend/data/raw/`. See
`backend/data_pipeline/README.md` for the expected file structure.

Inspect the dataset:

```
cd backend
python data_pipeline/inspect_dataset.py
```
