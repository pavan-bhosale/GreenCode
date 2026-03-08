"""
GreenCode Hybrid Predictor
============================
Main prediction pipeline combining all modules.
Equation (3): E_hybrid = E_physics + E_residual_ml

This is the primary entry point for predictions.
"""

import os
import json
from pathlib import Path
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

from core.static_analyzer import analyze_code
from core.physics_estimator import estimate_energy, estimate_power_watts
from core.carbon_estimator import estimate_carbon, compare_regions
from core.cost_estimator import estimate_cost, compare_instances

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "residual_model.pkl"
META_PATH = MODELS_DIR / "model_meta.json"


# ──────────────────────────────────────────────────────────────────────────────
# Load ML model
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_feature_keys = None


def _load_model():
    """Load the trained residual model and feature keys."""
    global _model, _feature_keys

    if _model is not None:
        return True

    if not MODEL_PATH.exists():
        print(f"⚠️  No trained model found at {MODEL_PATH}")
        print("   Run `python -m core.train_model` to train the model first.")
        return False

    if joblib is None:
        print("⚠️  joblib not installed")
        return False

    _model = joblib.load(MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
            _feature_keys = meta.get("feature_keys")

    return True


def _get_ml_residual(features: dict) -> float:
    """Predict the residual error using the ML model."""
    if _feature_keys is None:
        return 0.0

    feature_vector = []
    for key in _feature_keys:
        val = features.get(key, 0)
        if isinstance(val, str):
            val = 0
        feature_vector.append(float(val))

    residual = _model.predict(np.array([feature_vector]))[0]
    return float(residual)


# ──────────────────────────────────────────────────────────────────────────────
# Green Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_green_score(energy_joules: float, sloc: int) -> dict:
    """
    Compute a Green Score (A-D+ rating) based on energy efficiency.

    Based on energy per line of code — normalized for code size.
    """
    if sloc <= 0:
        sloc = 1

    energy_per_line = energy_joules / sloc

    # Rating thresholds (Joules per SLOC)
    if energy_per_line < 0.005:
        grade, label, color = "A+", "Exceptional", "#00C853"
    elif energy_per_line < 0.01:
        grade, label, color = "A", "Excellent", "#00E676"
    elif energy_per_line < 0.025:
        grade, label, color = "B+", "Very Good", "#69F0AE"
    elif energy_per_line < 0.05:
        grade, label, color = "B", "Good", "#76FF03"
    elif energy_per_line < 0.1:
        grade, label, color = "C+", "Fair", "#FFEB3B"
    elif energy_per_line < 0.25:
        grade, label, color = "C", "Needs Improvement", "#FFC107"
    elif energy_per_line < 0.5:
        grade, label, color = "D+", "Poor", "#FF9800"
    else:
        grade, label, color = "D", "Very Inefficient", "#FF5722"

    return {
        "grade": grade,
        "label": label,
        "color": color,
        "energy_per_sloc": round(energy_per_line, 6),
        "sloc": sloc,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    code: str,
    region: str = "India",
    instance: str = "aws-t3.medium",
    use_ml: bool = True,
) -> dict:
    """
    Full GreenCode prediction pipeline.

    Parameters
    ----------
    code : str
        Python source code to analyze.
    region : str
        Region for carbon intensity calculation.
    instance : str
        Cloud instance type for cost estimation.
    use_ml : bool
        Whether to use the hybrid ML model (if available).

    Returns
    -------
    dict
        Complete analysis results including:
        - features: static code features
        - energy: energy consumption (physics + ML hybrid)
        - carbon: CO₂ emissions for selected region
        - cost: cloud execution cost
        - green_score: efficiency rating
    """
    # 1. Static Analysis
    features = analyze_code(code)
    if "error" in features:
        return {"error": features["error"]}

    # 2. Physics-based energy estimate
    physics = estimate_energy(features)
    physics_energy = physics["energy_joules"]

    # 3. ML residual correction (Equation 3)
    ml_residual = 0.0
    model_used = False
    if use_ml and _load_model() and _model is not None:
        ml_residual = _get_ml_residual(features)
        model_used = True

    # Hybrid energy = physics + ML residual
    hybrid_energy_joules = max(0.0001, physics_energy + ml_residual)
    hybrid_energy_kwh = hybrid_energy_joules / 3_600_000

    # 4. Carbon estimation (Equation 4)
    carbon = estimate_carbon(hybrid_energy_kwh, region)

    # 5. Cost estimation (Equation 5)
    cost = estimate_cost(physics["runtime_seconds"], instance)

    # 6. Green Score
    sloc = features.get("sloc", 10)
    green_score = compute_green_score(hybrid_energy_joules, sloc)

    # 7. Component breakdown for charts
    power = physics["power"]
    total_power = power["cpu_watts"] + power["memory_watts"] + power["io_watts"] + power["network_watts"]
    if total_power > 0:
        breakdown = {
            "CPU": round(power["cpu_watts"] / total_power * 100, 1),
            "Memory": round(power["memory_watts"] / total_power * 100, 1),
            "I/O": round(power["io_watts"] / total_power * 100, 1),
            "Network": round(power["network_watts"] / total_power * 100, 1),
        }
    else:
        breakdown = {"CPU": 25, "Memory": 25, "I/O": 25, "Network": 25}

    return {
        "features": features,
        "energy": {
            "physics_joules": round(physics_energy, 6),
            "ml_residual_joules": round(ml_residual, 6),
            "hybrid_joules": round(hybrid_energy_joules, 6),
            "hybrid_kwh": round(hybrid_energy_kwh, 10),
            "hybrid_wh": round(hybrid_energy_kwh * 1000, 8),
            "runtime_seconds": physics["runtime_seconds"],
            "model_used": model_used,
        },
        "power": power,
        "component_breakdown_pct": breakdown,
        "carbon": carbon,
        "cost": cost,
        "green_score": green_score,
    }


def predict_and_compare(
    code: str,
    regions: list[str] | None = None,
    instance: str = "aws-t3.medium",
) -> dict:
    """
    Run prediction and include multi-region carbon comparison.
    """
    result = predict(code, region="India", instance=instance)
    if "error" in result:
        return result

    energy_kwh = result["energy"]["hybrid_kwh"]
    result["regional_comparison"] = compare_regions(energy_kwh, regions)
    result["instance_comparison"] = compare_instances(result["energy"]["runtime_seconds"])

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

data = [64, 34, 25, 12, 22, 11, 90]
result = bubble_sort(data)
print(result)
"""
    result = predict(sample, region="India")
    print(json.dumps(result, indent=2, default=str))
