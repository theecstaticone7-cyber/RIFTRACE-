"""POST /predict -- risk prediction for a single transaction ID."""

from fastapi import APIRouter, HTTPException

from ..logging_config import logger
from ..schemas import PredictRequest, PredictResponse
from ..services import model_service

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    logger.info(f"/predict tx_id={request.tx_id}")
    try:
        result = model_service.predict_transaction(request.tx_id)
    except KeyError as exc:
        logger.warning(f"/predict tx_id={request.tx_id} not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictResponse(**result)
