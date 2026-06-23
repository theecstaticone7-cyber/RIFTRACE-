# Data Pipeline

## Expected raw dataset files

Place the following files in `backend/data/raw/` (not committed to git):

### `elliptic_txs_features.csv`
- No header row.
- 203,769 rows, one per transaction node.
- 167 columns:
  - Column 0: `txId` (transaction node id)
  - Column 1: `time_step` (1–49, the local time step the transaction belongs to)
  - Columns 2–166: 165 numerical features. The first 94 are local transaction
    features; the remaining 71 are aggregated features from one-hop neighbors.

### `elliptic_txs_classes.csv`
- Header: `txId,class`
- One row per transaction node (203,769 rows).
- `class` values:
  - `"1"` — illicit
  - `"2"` — licit
  - `"unknown"` — unlabeled

### `elliptic_txs_edgelist.csv`
- Header: `txId1,txId2`
- 234,355 rows, one per directed edge (payment flow) between transaction nodes.

## Usage

```
cd backend
python data_pipeline/inspect_dataset.py
```

This loads the three CSVs above and prints summary statistics: node count,
edge count, class balance, number of time steps, and feature count per node.
