"""POST /explain/{tx_id} -- a plain-English explanation of why a
transaction received its prediction, grounded (RAG) in a small knowledge
base of Bitcoin money-laundering typologies plus the transaction's own
prediction and neighbor data. Reuses model_service/graph_service rather
than recomputing anything.
"""

from fastapi import APIRouter, HTTPException

from ..logging_config import logger
from ..schemas import ExplainResponse
from ..services import graph_service, llm_service, model_service, rag_service

router = APIRouter()

SYSTEM_PROMPT = (
    "You are a blockchain anti-money-laundering analyst assistant. Explain, "
    "in plain English and in at most five sentences, why a Bitcoin "
    "transaction received the given risk prediction. Base your explanation "
    "strictly on the transaction data and background patterns provided to "
    "you -- never invent specifics (amounts, dates, names) that aren't in "
    "that data. If the transaction is low risk, say so plainly rather than "
    "manufacturing concern."
)


def _build_retrieval_query(prediction: dict, neighbors: dict | None) -> str:
    """Builds the RAG query from this specific transaction's real
    characteristics (not a fixed string), so retrieval actually depends on
    the data being explained.
    """
    parts = [
        f"Bitcoin transaction predicted {prediction['prediction']} with "
        f"{prediction['probability_illicit']:.2f} probability of being illicit."
    ]

    if neighbors:
        illicit_neighbors = sum(1 for n in neighbors["neighbors"] if n["known_class"] == "illicit")
        if illicit_neighbors:
            parts.append(f"Connected to {illicit_neighbors} known illicit transactions.")
        if neighbors["num_neighbors"] > 20:
            parts.append("Very high number of connected transactions, high fan-out connectivity.")
        elif neighbors["num_neighbors"] <= 1:
            parts.append("Very few connected transactions, low connectivity.")

    return " ".join(parts)


def _build_user_prompt(tx_id: int, prediction: dict, neighbors: dict | None, sources: list[dict]) -> str:
    lines = [
        f"Transaction tx_id: {tx_id}",
        f"Model prediction: {prediction['prediction']}",
        f"Probability illicit: {prediction['probability_illicit']:.4f}",
    ]

    if neighbors:
        lines.append(f"Connected transactions in graph: {neighbors['num_neighbors']}")
        by_class: dict[str, int] = {}
        for n in neighbors["neighbors"]:
            by_class[n["known_class"]] = by_class.get(n["known_class"], 0) + 1
        if by_class:
            breakdown = ", ".join(f"{count} {cls}" for cls, count in sorted(by_class.items()))
            lines.append(f"Neighbor class breakdown: {breakdown}")
    else:
        lines.append("Connected transactions in graph: unknown (this tx_id was not found in the graph)")

    lines.append("\nRelevant background on money-laundering patterns:")
    for doc in sources:
        lines.append(f"- {doc['title']}: {doc['text']}")

    lines.append("\nExplain why this transaction received this prediction.")
    return "\n".join(lines)


@router.post("/explain/{tx_id}", response_model=ExplainResponse)
def explain_transaction(tx_id: int) -> ExplainResponse:
    logger.info(f"/explain tx_id={tx_id}")
    try:
        prediction = model_service.predict_transaction(tx_id)
    except KeyError as exc:
        logger.warning(f"/explain tx_id={tx_id} not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Neighbor context is supplementary, not required -- a tx that predicts
    # fine but has no graph entry still gets an explanation, just without
    # the connectivity-derived part of the prompt.
    try:
        neighbors = graph_service.get_neighbors(tx_id)
    except KeyError:
        neighbors = None

    query = _build_retrieval_query(prediction, neighbors)
    sources = rag_service.retrieve(query, k=3)
    user_prompt = _build_user_prompt(tx_id, prediction, neighbors, sources)

    result = llm_service.generate_explanation(SYSTEM_PROMPT, user_prompt)

    return ExplainResponse(
        tx_id=tx_id,
        prediction=prediction["prediction"],
        probability_illicit=prediction["probability_illicit"],
        explanation=result["explanation"],
        sources=[doc["title"] for doc in sources],
        grounded=result["grounded"],
    )
