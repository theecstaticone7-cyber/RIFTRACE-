"""Phase C: exposes RiftRace's capabilities as MCP tools, so an MCP-compatible
client can call them directly instead of going through the FastAPI HTTP API.

Completely separate process from api/main.py -- this file is never imported
by the FastAPI app, and nothing here is imported by it either. It reuses the
same service modules (model_service, graph_service, rag_service,
agent_service) and the same Pydantic schemas (api/schemas.py) the API
already validates against, so a tool's output shape is guaranteed to match
its HTTP equivalent without duplicating any prediction/graph/LLM logic.

Run with: cd backend && python -m mcp_server.server
(stdio transport -- the right choice for a locally-spawned subprocess, as
opposed to sse/streamable-http for a networked server, since MCP clients
typically launch local servers this way.)
"""

from mcp.server.fastmcp import FastMCP

from api.schemas import (
    DatasetStats,
    InvestigationResponse,
    ModelMetrics,
    PredictResponse,
    StatsResponse,
    TransactionGraphResponse,
)
from api.services import agent_service, graph_service, model_service, rag_service

mcp = FastMCP("riftrace")


@mcp.tool()
def predict_transaction(tx_id: int) -> PredictResponse:
    """Predict whether a Bitcoin transaction is licit or illicit.

    Runs RiftRace's trained Random Forest model (F1=0.81, ROC-AUC=0.94 on
    the held-out temporal test split) against the transaction's feature
    vector from the Elliptic dataset.

    Args:
        tx_id: The Elliptic dataset transaction ID to score.

    Raises:
        ValueError: tx_id isn't present in the dataset's feature store.
    """
    try:
        result = model_service.predict_transaction(tx_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return PredictResponse(**result)


@mcp.tool()
def get_transaction_graph(tx_id: int) -> TransactionGraphResponse:
    """Get a transaction's known class and its directly connected neighbors.

    Looks up the transaction in RiftRace's cached transaction graph (built
    from the Elliptic edge list) and returns each neighbor's tx_id, whether
    it's an incoming or outgoing edge, and its ground-truth class
    (licit/illicit/unknown).

    Args:
        tx_id: The Elliptic dataset transaction ID to look up.

    Raises:
        ValueError: tx_id isn't present in the transaction graph.
    """
    try:
        result = graph_service.get_neighbors(tx_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return TransactionGraphResponse(**result)


@mcp.tool()
def get_dataset_stats() -> StatsResponse:
    """Get the served model's performance metrics and dataset summary.

    Metrics (precision/recall/F1/ROC-AUC/accuracy for the illicit class) are
    computed on the temporal test split (steps 35-49), never a random
    resample. Dataset stats cover the full Elliptic graph: node/edge counts
    and the licit/illicit/unknown class breakdown.
    """
    return StatsResponse(
        model_name=model_service.MODEL_NAME,
        metrics=ModelMetrics(**model_service.get_model_metrics()),
        dataset=DatasetStats(**model_service.get_dataset_stats()),
    )


@mcp.tool()
def investigate_transaction(tx_id: int) -> InvestigationResponse:
    """Run RiftRace's multi-agent investigation workflow on a transaction.

    Runs the same LangGraph pipeline as POST /investigate/{tx_id}: ANALYZE
    (model prediction + graph neighbors) -> RETRIEVE (relevant AML typology
    knowledge) -> REASON (LLM risk assessment) -> RECOMMEND (LLM next
    investigative action). Returns all four outputs plus the retrieved
    source titles.

    If GROQ_API_KEY isn't configured on the server, or the Groq call fails,
    risk_assessment and recommended_actions hold a clear fallback message
    (`grounded` is False) instead of raising -- the analysis and retrieved
    sources are still real either way, since those steps don't need the LLM.

    Args:
        tx_id: The Elliptic dataset transaction ID to investigate.

    Raises:
        ValueError: tx_id isn't present in the dataset's feature store.
    """
    try:
        report = agent_service.run_investigation(tx_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    return InvestigationResponse(tx_id=tx_id, **report)


def _warm_up_services() -> None:
    """Mirrors api/main.py's lifespan warm-up: this is a separate process
    with its own module-level caches, so loading the model/graph/knowledge
    base eagerly here means the same thing it means there -- fail fast on a
    missing/broken artifact, and no first-call latency spike for whichever
    tool an assistant happens to call first.
    """
    model_service.warm_up()
    graph_service.warm_up()
    rag_service.warm_up()


if __name__ == "__main__":
    _warm_up_services()
    mcp.run(transport="stdio")
