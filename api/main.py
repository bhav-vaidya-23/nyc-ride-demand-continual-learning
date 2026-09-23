"""
FastAPI application for NYC Ride Demand Prediction & Continual Learning Monitoring.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    MonitoringResponse,
    DriftResponse,
    GovernanceResponse,
)
from api.services import DemandPredictionService

app = FastAPI(
    title="NYC Ride Demand Prediction API",
    description=(
        "Production inference service and operational monitoring platform for 15-minute "
        "NYC ride demand forecasting with event-triggered continual learning governance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
def health_check() -> HealthResponse:
    """Return operational health status of the API."""
    return HealthResponse(status="healthy", service="NYC Ride Demand Prediction API", version="1.0.0")


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    tags=["Model Registry"],
    summary="Get active production model and lineage",
)
def get_model() -> ModelInfoResponse:
    """Return active production champion model, training window, and registry lineage."""
    try:
        service = DemandPredictionService.get_instance()
        return service.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model metadata: {str(e)}",
        ) from e


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict 15-minute ride demand for a pickup zone",
)
def predict_demand(request: PredictionRequest) -> PredictionResponse:
    """
    Generate a 15-minute ride demand forecast for a given NYC TLC pickup zone (1-263) and timestamp.
    Features are computed dynamically from past historical intervals with zero future leakage.
    """
    if request.zone < 1 or request.zone > 263:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid zone ID {request.zone}. Must be between 1 and 263.",
        )

    try:
        service = DemandPredictionService.get_instance()
        return service.predict(zone=request.zone, timestamp_str=request.timestamp)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        ) from ve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        ) from e


@app.get(
    "/monitoring",
    response_model=MonitoringResponse,
    tags=["Monitoring"],
    summary="Get production performance metrics and degradation status",
)
def get_monitoring() -> MonitoringResponse:
    """Return rolling 24-hour MAE, RMSE, prediction bias, and operational degradation status."""
    try:
        service = DemandPredictionService.get_instance()
        return service.get_monitoring_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing monitoring summary: {str(e)}",
        ) from e


@app.get(
    "/drift",
    response_model=DriftResponse,
    tags=["Monitoring"],
    summary="Get feature drift (PSI and KS-test) metrics",
)
def get_drift() -> DriftResponse:
    """Return Population Stability Index (PSI) and Kolmogorov-Smirnov statistics across monitored features."""
    try:
        service = DemandPredictionService.get_instance()
        return service.get_drift_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing drift summary: {str(e)}",
        ) from e


@app.get(
    "/governance",
    response_model=GovernanceResponse,
    tags=["Governance"],
    summary="Get Champion-Challenger evaluation gate results",
)
def get_governance() -> GovernanceResponse:
    """Return Champion vs Challenger holdout evaluation metrics, promotion decision, and audit lineage."""
    try:
        service = DemandPredictionService.get_instance()
        return service.get_governance_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving governance audit records: {str(e)}",
        ) from e
