"""
Continual learning model governance, versioning, and champion-challenger promotion gates.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from src.config import (
    MODELS_DIR,
    MODEL_REGISTRY_JSON,
    PROMOTION_IMPROVEMENT_THRESHOLD,
)
from src.metrics import calculate_metrics, evaluate_subgroups


class ModelRegistry:
    """
    Manages model versioning, metadata logging, and lineage tracking.
    """

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or MODEL_REGISTRY_JSON
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict] = self._load()

    def _load(self) -> Dict[str, Dict]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(
        self,
        version: str,
        model_type: str,
        training_period: Tuple[str, str],
        features: List[str],
        val_metrics: Dict[str, float],
        status: str = "candidate",
        reason: str = "Initial training",
        artifact_path: Optional[str] = None,
    ) -> Dict:
        entry = {
            "version": version,
            "model_type": model_type,
            "training_period": list(training_period),
            "features": features,
            "validation_metrics": val_metrics,
            "status": status,
            "reason": reason,
            "registered_at": datetime.now().isoformat(),
            "artifact_path": artifact_path,
        }
        self.models[version] = entry
        self._save()
        return entry

    def get_champion(self) -> Optional[Dict]:
        for m in self.models.values():
            if m.get("status") == "champion":
                return m
        return None

    def promote_to_champion(self, version: str, reason: str) -> None:
        for v, m in self.models.items():
            if m.get("status") == "champion":
                m["status"] = "archived"
        if version in self.models:
            self.models[version]["status"] = "champion"
            self.models[version]["reason"] = reason
            self.models[version]["promoted_at"] = datetime.now().isoformat()
            self._save()

    def get_lineage_table(self) -> pd.DataFrame:
        rows = []
        for v, m in self.models.items():
            vm = m.get("validation_metrics", {})
            rows.append({
                "Version": v,
                "Model Type": m.get("model_type"),
                "Status": m.get("status"),
                "Val MAE": vm.get("MAE"),
                "Val RMSE": vm.get("RMSE"),
                "Training Start": m.get("training_period", ["", ""])[0],
                "Training End": m.get("training_period", ["", ""])[1],
                "Reason": m.get("reason"),
            })
        return pd.DataFrame(rows)


def evaluate_champion_challenger(
    champion_model,
    challenger_model,
    holdout_df: pd.DataFrame,
    feature_cols: List[str],
    high_demand_zones: List[int],
    min_improvement: float = PROMOTION_IMPROVEMENT_THRESHOLD,
) -> Tuple[bool, Dict]:
    """
    Controlled Champion vs Challenger evaluation gate.

    Promotion Criteria:
    1. Challenger overall MAE must be at least min_improvement lower than Champion.
    2. Challenger high-demand zone MAE must not regress by more than 2%.
    """
    X_val = holdout_df[feature_cols]
    y_val = holdout_df["demand"].to_numpy()

    champ_preds = champion_model.predict(X_val)
    chall_preds = challenger_model.predict(X_val)

    champ_eval = evaluate_subgroups(holdout_df, champ_preds, high_demand_zones)
    chall_eval = evaluate_subgroups(holdout_df, chall_preds, high_demand_zones)

    champ_mae = champ_eval["Overall"]["MAE"]
    chall_mae = chall_eval["Overall"]["MAE"]

    pct_overall_imprv = ((champ_mae - chall_mae) / champ_mae)

    champ_high_mae = champ_eval["High-Demand Zones"]["MAE"]
    chall_high_mae = chall_eval["High-Demand Zones"]["MAE"]
    pct_high_regress = ((chall_high_mae - champ_high_mae) / champ_high_mae)

    passed_overall = pct_overall_imprv >= min_improvement
    passed_high_zone = pct_high_regress <= 0.02

    promoted = passed_overall and passed_high_zone

    summary = {
        "promoted": promoted,
        "champion_mae": champ_mae,
        "challenger_mae": chall_mae,
        "pct_overall_improvement": round(pct_overall_imprv * 100, 2),
        "champion_high_zone_mae": champ_high_mae,
        "challenger_high_zone_mae": chall_high_mae,
        "pct_high_zone_regression": round(pct_high_regress * 100, 2),
        "passed_overall_threshold": passed_overall,
        "passed_high_zone_safety": passed_high_zone,
    }
    return promoted, summary
