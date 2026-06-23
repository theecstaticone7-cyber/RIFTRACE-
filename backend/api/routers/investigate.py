"""POST /investigate/{tx_id} -- runs the Phase B multi-agent LangGraph
workflow (analyze -> retrieve -> reason -> recommend, see
services/agent_service.py) and returns a structured investigation report.

Separate from /explain (routers/explain.py), which still serves its
single-step explanation unchanged -- this is an additional, richer report,
not a replacement.
"""

from fastapi import APIRouter, HTTPException

from ..logging_config import logger
from ..schemas import InvestigationResponse
from ..services import agent_service

router = APIRouter()


@router.post("/investigate/{tx_id}", response_model=InvestigationResponse)
def investigate_transaction(tx_id: int) -> InvestigationResponse:
    logger.info(f"/investigate tx_id={tx_id}")
    try:
        report = agent_service.run_investigation(tx_id)
    except KeyError as exc:
        logger.warning(f"/investigate tx_id={tx_id} not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return InvestigationResponse(tx_id=tx_id, **report)
