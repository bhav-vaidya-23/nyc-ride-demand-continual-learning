"""
Pydantic schemas for the NYC Ride Demand Prediction API.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="API operational health status")
    service: str = Field(default="NYC Ride Demand Prediction API")
    version: str = Field(default="1.0.0")


class ModelInfoResponse(BaseModel):
    champion_version: str
    model_type: str
    status: str
    training_period: List[str]
    features: List[str]
    validation_metrics: Dict[str, float]
    registered_at: Optional[str] = None
    lineage: List[Dict[str, Any]] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    zone: int = Field(..., ge=1, le=263, description="NYC TLC Pickup Location ID (1-263)", example=138)
    timestamp: str = Field(..., description="Timestamp in 'YYYY-MM-DD HH:MM:SS' or ISO format", example="2025-02-20 18:00:00")


class PredictionResponse(BaseModel):
    zone: int
    timestamp: str
    predicted_demand: float
    model_version: str
    execution_time_ms: Optional[float] = None


class MonitoringResponse(BaseModel):
    overall_mae: float
    overall_rmse: float
    overall_bias: float
    overall_wape: float
    latest_rolling_24h_mae: float
    peak_rolling_24h_mae: float
    degradation_threshold_mae: float
    degradation_status: str
    is_degraded: bool
    latest_timestamp: str
    total_predictions_logged: int
    number_of_zones: int


class DriftFeatureItem(BaseModel):
    feature: str
    psi: float
    psi_status: str
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    drift_detected: bool


class DriftResponse(BaseModel):
    drift_status: str
    features: List[DriftFeatureItem]
    reference_period: str
    current_period: str
    scientific_note: str


class GovernanceResponse(BaseModel):
    champion_version: str
    challenger_version: Optional[str]
    champion_mae: float
    challenger_mae: Optional[float]
    observed_improvement_pct: Optional[float]
    required_improvement_pct: float
    decision: str
    reason: str
    lineage: List[Dict[str, Any]]
