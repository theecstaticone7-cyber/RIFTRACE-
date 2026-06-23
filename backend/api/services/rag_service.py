"""Retrieval over a small, curated knowledge base of Bitcoin money-laundering
typologies (data/knowledge_base/aml_patterns.json), used to ground /explain
and /investigate's LLM output in real, citable background instead of letting
the model invent generic-sounding justification.

Embeddings: TF-IDF, not a local neural embedding model. The original version
used sentence-transformers/all-MiniLM-L6-v2, which pulls in torch -- on a
17-document corpus that added ~390MB of RAM (torch + the model itself) just
to embed a knowledge base small enough to fit in a spreadsheet, which is
what made this app blow past Render's 512MB free-tier limit. The fitted
TfidfVectorizer and the docs' embeddings are precomputed offline by
data_pipeline/build_kb_embeddings.py and committed under data/knowledge_base/
-- this module only ever loads those two small artifacts (~40KB total) and
reuses the same fitted vectorizer to embed queries at request time, so docs
and queries stay in the same vector space. scikit-learn is already a hard
dependency of model_service, so this adds no new package and ~0MB of new
runtime memory.

Index: FAISS IndexFlatIP over normalized vectors, i.e. cosine similarity --
unchanged from the original design; with only ~17 short docs an exact flat
index is plenty fast and keeps this dependency-light rather than reaching
for an approximate index meant for much larger corpora.
"""

import json
from pathlib import Path

import faiss
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BACKEND_DIR / "data" / "knowledge_base"
KB_PATH = KB_DIR / "aml_patterns.json"
VECTORIZER_PATH = KB_DIR / "tfidf_vectorizer.joblib"
EMBEDDINGS_PATH = KB_DIR / "aml_patterns_embeddings.npy"

# Checked by api/main.py's fail-fast startup validation, same as the other
# services' REQUIRED_FILES. Imports joblib lazily (see _load) so this module
# can be imported without it if ever needed -- not load-bearing, just tidy.
REQUIRED_FILES = [KB_PATH, VECTORIZER_PATH, EMBEDDINGS_PATH]

# Populated on first use by _load() (module-level cache, same pattern as
# model_service/graph_service).
_vectorizer = None
_index = None
_docs = None  # list of {"id", "title", "text"}, in the same order as _index


def _load() -> None:
    global _vectorizer, _index, _docs
    if _index is not None:
        return

    import joblib  # local import: only this function needs it

    _docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
    _vectorizer = joblib.load(VECTORIZER_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)


def warm_up() -> None:
    """Loads the vectorizer, embeddings, and FAISS index at startup -- a
    missing/broken artifact should surface immediately, not silently on the
    first /explain or /investigate request.
    """
    _load()


def _embed_query(query: str) -> np.ndarray:
    vector = _vectorizer.transform([query]).toarray().astype("float32")
    norm = np.linalg.norm(vector)
    # A query that shares zero vocabulary with the fitted corpus produces an
    # all-zero TF-IDF vector -- skip normalizing (would divide by zero) and
    # let it score 0 similarity against every doc rather than crashing.
    if norm > 0:
        vector = vector / norm
    return vector


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Top-k knowledge base docs most similar to `query`, ranked by cosine
    similarity (embeddings are normalized, so inner product == cosine).
    """
    _load()
    k = min(k, len(_docs))

    query_vector = _embed_query(query)
    scores, indices = _index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = _docs[idx]
        results.append({**doc, "score": float(score)})
    return results


def health_check() -> dict:
    """Confirms the knowledge base is actually loaded and indexed."""
    _load()
    return {"kb_loaded": True, "kb_doc_count": len(_docs)}
