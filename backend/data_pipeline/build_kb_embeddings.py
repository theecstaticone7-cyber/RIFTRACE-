"""Precomputes TF-IDF embeddings for the AML knowledge-base docs used by
rag_service, so the deployed app never has to load sentence-transformers/
torch (~390MB of RAM) just to embed a 17-document corpus.

TF-IDF (not a swapped-in neural embedding) is used for the *docs* here too,
not only at query time -- a dense semantic doc embedding and a lexical query
embedding live in different vector spaces and can't be compared via cosine
similarity. Fitting one TfidfVectorizer and reusing it for both docs (here,
offline) and queries (rag_service.py, at request time) keeps both sides in
the same space. scikit-learn is already a hard dependency of model_service,
so this adds no new runtime cost at all.

Re-run this whenever data/knowledge_base/aml_patterns.json changes:
    cd backend && venv\\Scripts\\python.exe -m data_pipeline.build_kb_embeddings
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
KB_PATH = DATA_DIR / "aml_patterns.json"
VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.joblib"
EMBEDDINGS_PATH = DATA_DIR / "aml_patterns_embeddings.npy"


def build_embeddings() -> tuple[TfidfVectorizer, np.ndarray]:
    docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
    # Title + text: titles carry concentrated keywords ("sanctioned
    # addresses", "low connectivity") that the synthetic queries built by
    # routers/explain.py and services/agent_service.py often echo directly.
    corpus = [f"{doc['title']}. {doc['text']}" for doc in docs]

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(corpus).toarray().astype("float32")
    # L2-normalize so FAISS's IndexFlatIP (inner product) computes cosine
    # similarity, matching rag_service's existing convention.
    embeddings = normalize(embeddings, norm="l2").astype("float32")
    return vectorizer, embeddings


def save(vectorizer: TfidfVectorizer, embeddings: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)


if __name__ == "__main__":
    vectorizer, embeddings = build_embeddings()
    save(vectorizer, embeddings)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
