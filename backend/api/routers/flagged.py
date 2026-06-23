"""GET /flagged -- top N test-set transactions predicted illicit, ranked by
probability descending. Backs the dashboard's investigation entry point: an
analyst doesn't need to already know a tx_id to find something suspicious.
"""

from fastapi import APIRouter, Query

from ..logging_config import logger
from ..schemas import FlaggedResponse
from ..services import model_service

router = APIRouter()


@router.get("/flagged", response_model=FlaggedResponse)
def get_flagged(limit: int = Query(20, ge=1, le=500)) -> FlaggedResponse:
    logger.info(f"/flagged limit={limit}")
    flagged = model_service.get_flagged_transactions(limit)
    return FlaggedResponse(flagged=flagged, count=len(flagged))
