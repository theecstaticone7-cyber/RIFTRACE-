"""GET /transaction/{tx_id}/graph -- a transaction's immediate neighbors,
for graph visualization on the frontend.
"""

from fastapi import APIRouter, HTTPException

from ..logging_config import logger
from ..schemas import TransactionGraphResponse
from ..services import graph_service

router = APIRouter()


@router.get("/transaction/{tx_id}/graph", response_model=TransactionGraphResponse)
def get_transaction_graph(tx_id: int) -> TransactionGraphResponse:
    logger.info(f"/transaction/{tx_id}/graph")
    try:
        result = graph_service.get_neighbors(tx_id)
    except KeyError as exc:
        logger.warning(f"/transaction/{tx_id}/graph not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TransactionGraphResponse(**result)
