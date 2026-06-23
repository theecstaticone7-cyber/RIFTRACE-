"""Pydantic request/response models for the RiftRace API."""

from typing import Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    tx_id: int


class PredictResponse(BaseModel):
    tx_id: int
    prediction: Literal["licit", "illicit"]
    probability_illicit: float


class Neighbor(BaseModel):
    tx_id: int
    direction: Literal["incoming", "outgoing"]
    known_class: Literal["licit", "illicit", "unknown"]


class TransactionGraphResponse(BaseModel):
    tx_id: int
    known_class: Literal["licit", "illicit", "unknown"]
    neighbors: list[Neighbor]
    num_neighbors: int


class ModelMetrics(BaseModel):
    # Illicit-class metrics from the temporal test split (steps 35-49), the
    # same numbers produced by models/evaluate.py.
    precision: float
    recall: float
    f1: float
    roc_auc: float
    accuracy: float


class DatasetStats(BaseModel):
    total_nodes: int
    total_edges: int
    num_time_steps: int
    feature_count: int
    pct_licit: float
    pct_illicit: float
    pct_unknown: float


class StatsResponse(BaseModel):
    model_name: str
    metrics: ModelMetrics
    dataset: DatasetStats


class FlaggedTransaction(BaseModel):
    tx_id: int
    probability_illicit: float
    # Ground-truth label from the dataset, not the model's prediction --
    # lets an analyst see at a glance whether a flagged transaction is a
    # true positive ("illicit") or a false positive ("licit").
    known_class: Literal["licit", "illicit"]


class FlaggedResponse(BaseModel):
    flagged: list[FlaggedTransaction]
    count: int


class ExplainResponse(BaseModel):
    tx_id: int
    prediction: Literal["licit", "illicit"]
    probability_illicit: float
    # Plain-English explanation grounded in the transaction's real
    # prediction/neighbor data plus the retrieved knowledge-base context.
    explanation: str
    # Titles of the knowledge-base docs retrieved for this explanation.
    sources: list[str]
    # False when GROQ_API_KEY isn't set or the LLM call failed -- in that
    # case `explanation` is a clear fallback message, not a fabricated one.
    grounded: bool


class InvestigationResponse(BaseModel):
    tx_id: int
    prediction: Literal["licit", "illicit"]
    probability_illicit: float
    # ANALYZE's deterministic summary of the prediction + graph signals.
    analysis: str
    # Titles of the knowledge-base docs RETRIEVE pulled in for REASON.
    retrieved_sources: list[str]
    # REASON's output: risk level + justification.
    risk_assessment: str
    # RECOMMEND's output: suggested next investigative action(s).
    recommended_actions: str
    # False when GROQ_API_KEY isn't set or either LLM step's call failed --
    # in that case risk_assessment/recommended_actions hold llm_service's
    # fallback message rather than a fabricated assessment.
    grounded: bool
