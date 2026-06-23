"""Phase B: a multi-agent investigation workflow built on LangGraph, run by
POST /investigate/{tx_id}. Replaces /explain's single LLM call with four
nodes that share one state object and run in sequence:

    ANALYZE -> RETRIEVE -> REASON -> RECOMMEND -> END

ANALYZE and RETRIEVE are deterministic (reuse model_service, graph_service,
rag_service -- no LLM, no network, never fail due to a missing API key).
REASON and RECOMMEND are LLM steps that both go through llm_service, which
already degrades to a clear, non-grounded message when GROQ_API_KEY is
unset or the call fails -- this module doesn't special-case that, it just
inherits the fallback from each node's llm_service.generate_explanation()
call.

Deliberately separate from routers/explain.py and llm_service.py: nothing
here is imported by them, and nothing in them is changed, so Phase A's
behavior (and its tests) can't regress.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import graph_service, llm_service, model_service, rag_service


class InvestigationState(TypedDict, total=False):
    tx_id: int
    prediction: dict
    neighbors: dict | None
    analysis: str
    retrieved_sources: list[dict]
    risk_assessment: str
    recommended_actions: str
    grounded: bool


REASON_SYSTEM_PROMPT = (
    "You are a senior blockchain anti-money-laundering investigator. Given a "
    "transaction's model prediction, its graph connectivity analysis, and "
    "relevant AML typology background, assess its risk level and justify "
    "that assessment in plain English, in at most five sentences. Base your "
    "assessment strictly on the data provided -- never invent specifics "
    "(amounts, dates, names) that aren't in it."
)

RECOMMEND_SYSTEM_PROMPT = (
    "You are a senior blockchain anti-money-laundering investigator advising "
    "a human analyst. Given the transaction's data and your own prior risk "
    "assessment, recommend one or two concrete next investigative actions "
    "the analyst should take (for example, tracing a specific neighbor, or "
    "checking for a specific laundering pattern). Be specific and concise -- "
    "at most three sentences. If the transaction is low risk, say plainly "
    "that no further action is needed rather than manufacturing busywork."
)


def _describe_neighbors(neighbors: dict | None) -> str:
    """Mirrors routers/explain.py's _build_retrieval_query phrasing for the
    fan-out/low-connectivity thresholds (not just a style match -- RETRIEVE
    uses this same text as its RAG query, and rag_service does lexical
    TF-IDF matching now, not semantic embedding, so literally echoing the
    knowledge base's own wording ("low connectivity", "fan-out") is what
    makes retrieval actually find the relevant docs instead of returning the
    same generic top-3 regardless of this transaction's real connectivity.
    """
    if not neighbors:
        return "This transaction was not found in the connectivity graph, so no neighbor data is available."

    if neighbors["num_neighbors"] == 0:
        return "This transaction has no connected transactions in the graph."

    by_class: dict[str, int] = {}
    for n in neighbors["neighbors"]:
        by_class[n["known_class"]] = by_class.get(n["known_class"], 0) + 1
    breakdown = ", ".join(f"{count} {cls}" for cls, count in sorted(by_class.items()))

    parts = [
        f"This transaction has {neighbors['num_neighbors']} connected transactions "
        f"in the graph ({breakdown})."
    ]

    illicit_neighbors = by_class.get("illicit", 0)
    if illicit_neighbors:
        parts.append(f"Connected to {illicit_neighbors} known illicit transactions.")
    if neighbors["num_neighbors"] > 20:
        parts.append("Very high number of connected transactions, high fan-out connectivity.")
    elif neighbors["num_neighbors"] <= 1:
        parts.append("Very few connected transactions, low connectivity.")

    return " ".join(parts)


def analyze_node(state: InvestigationState) -> dict:
    """Pure data gathering: the actual prediction + neighbor lookups, then a
    deterministic plain-English summary of those raw signals. Raises
    KeyError if tx_id isn't in the dataset -- left uncaught so it propagates
    through graph.invoke() to the router, same as /explain's 404 handling.
    """
    tx_id = state["tx_id"]
    prediction = model_service.predict_transaction(tx_id)

    try:
        neighbors = graph_service.get_neighbors(tx_id)
    except KeyError:
        neighbors = None

    analysis = (
        f"Transaction {tx_id} was predicted {prediction['prediction']} with "
        f"{prediction['probability_illicit']:.4f} probability of being illicit. "
        f"{_describe_neighbors(neighbors)}"
    )

    return {"prediction": prediction, "neighbors": neighbors, "analysis": analysis}


def retrieve_node(state: InvestigationState) -> dict:
    """Grounds the upcoming LLM steps in real AML background, retrieved
    using ANALYZE's summary as the query -- so retrieval depends on this
    transaction's actual signals, not a fixed string.
    """
    sources = rag_service.retrieve(state["analysis"], k=3)
    return {"retrieved_sources": sources}


def _build_background(sources: list[dict]) -> str:
    lines = ["Relevant background on money-laundering patterns:"]
    lines += [f"- {doc['title']}: {doc['text']}" for doc in sources]
    return "\n".join(lines)


def reason_node(state: InvestigationState) -> dict:
    user_prompt = (
        f"{state['analysis']}\n\n"
        f"{_build_background(state['retrieved_sources'])}\n\n"
        "Assess this transaction's risk level and justify your assessment."
    )
    result = llm_service.generate_explanation(REASON_SYSTEM_PROMPT, user_prompt)
    return {"risk_assessment": result["explanation"], "grounded": result["grounded"]}


def recommend_node(state: InvestigationState) -> dict:
    user_prompt = (
        f"{state['analysis']}\n\n"
        f"Your prior risk assessment: {state['risk_assessment']}\n\n"
        "Recommend the next investigative action(s) for a human analyst."
    )
    result = llm_service.generate_explanation(RECOMMEND_SYSTEM_PROMPT, user_prompt)
    return {
        "recommended_actions": result["explanation"],
        # Overall report is only "grounded" if both LLM steps actually ran
        # against a real model, not just one of the two.
        "grounded": state.get("grounded", True) and result["grounded"],
    }


def _build_graph():
    graph = StateGraph(InvestigationState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("recommend", recommend_node)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


# Built once at import time: pure in-memory graph wiring, no file I/O or
# network calls, so unlike model_service/rag_service this needs no separate
# warm_up()/REQUIRED_FILES -- a wiring mistake here fails as soon as
# api.routers.investigate is imported, which happens at app startup anyway.
_COMPILED_GRAPH = _build_graph()


def run_investigation(tx_id: int) -> dict:
    """Runs the full analyze -> retrieve -> reason -> recommend workflow for
    `tx_id` and returns a flat dict ready to build an InvestigationResponse
    from. Raises KeyError if tx_id isn't in the dataset.
    """
    final_state = _COMPILED_GRAPH.invoke({"tx_id": tx_id})

    return {
        "prediction": final_state["prediction"]["prediction"],
        "probability_illicit": final_state["prediction"]["probability_illicit"],
        "analysis": final_state["analysis"],
        "retrieved_sources": [doc["title"] for doc in final_state["retrieved_sources"]],
        "risk_assessment": final_state["risk_assessment"],
        "recommended_actions": final_state["recommended_actions"],
        "grounded": final_state["grounded"],
    }
