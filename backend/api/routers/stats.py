"""GET /stats -- served model's performance metrics and dataset summary."""

from fastapi import APIRouter

from ..schemas import StatsResponse
from ..services import model_service

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return StatsResponse(
        model_name=model_service.MODEL_NAME,
        metrics=model_service.get_model_metrics(),
        dataset=model_service.get_dataset_stats(),
    )
