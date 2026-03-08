"""
GreenCode Model Trainer
========================
Generates synthetic training data and trains an XGBoost residual model.
The residual model learns to correct systematic errors in the physics estimator.

E_hybrid = E_physics + E_residual (Equation 3)
"""

import os
import json
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path

# We'll use try/except for imports that might not be installed yet
try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    import joblib
except ImportError:
    print("⚠️  Required packages not installed. Run: pip install -r requirements.txt")
    raise

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────────────

# Feature ranges for synthetic code samples
FEATURE_RANGES = {
    "sloc":                   (5, 500),
    "lloc":                   (5, 500),
    "function_count":         (0, 20),
    "class_count":            (0, 5),
    "loop_count":             (0, 15),
    "max_loop_depth":         (0, 4),
    "nested_loop_count":      (0, 8),
    "conditional_count":      (0, 20),
    "try_except_count":       (0, 5),
    "list_comprehensions":    (0, 10),
    "io_operations":          (0, 10),
    "network_calls":          (0, 5),
    "heavy_math_imports":     (0, 3),
    "recursion_candidates":   (0, 3),
    "avg_complexity":         (1, 15),
    "max_complexity":         (1, 25),
    "total_complexity":       (1, 100),
    "halstead_volume":        (0, 5000),
    "halstead_difficulty":    (0, 50),
    "halstead_effort":        (0, 100000),
    "maintainability_index":  (10, 100),
    "computational_intensity":(0, 100),
}

# Feature keys used for the ML model (numeric only)
FEATURE_KEYS = list(FEATURE_RANGES.keys())


def _generate_synthetic_energy(features: dict) -> float:
    """
    Generate a realistic energy value (Joules) for a synthetic code sample.
    Uses a combination of known relationships + noise.
    """
    sloc = features["sloc"]
    loops = features["loop_count"]
    depth = features["max_loop_depth"]
    nested = features["nested_loop_count"]
    complexity = features["avg_complexity"]
    io_ops = features["io_operations"]
    net = features["network_calls"]
    math_heavy = features["heavy_math_imports"]
    intensity = features["computational_intensity"]

    # Base energy scales with code size
    base = sloc * 0.002

    # Loop contribution (exponential with depth)
    loop_energy = loops * 0.05 * (2.5 ** depth)
    nested_penalty = nested * 0.15 * (1.5 ** depth)

    # I/O and network are relatively expensive
    io_energy = io_ops * 0.08 + net * 0.5

    # Heavy math libraries use more CPU
    math_energy = math_heavy * 0.3 * (1 + intensity / 50)

    # Complexity scaling
    complexity_factor = 1.0 + (complexity - 1) * 0.05

    # Total with noise
    energy = (base + loop_energy + nested_penalty + io_energy + math_energy) * complexity_factor

    # Add realistic noise (±15%)
    noise = random.gauss(1.0, 0.15)
    energy *= max(0.5, noise)

    return round(max(0.001, energy), 6)


def generate_training_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training dataset.
    
    Each row represents a code sample with features and actual energy consumption.
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for _ in range(n_samples):
        features = {}
        for key, (lo, hi) in FEATURE_RANGES.items():
            if isinstance(lo, float) or isinstance(hi, float):
                features[key] = round(random.uniform(lo, hi), 2)
            else:
                features[key] = random.randint(lo, hi)

        # Add realistic correlations
        # More loops → higher depth is more likely
        if features["loop_count"] == 0:
            features["max_loop_depth"] = 0
            features["nested_loop_count"] = 0
        else:
            features["max_loop_depth"] = min(
                features["max_loop_depth"], features["loop_count"]
            )
            features["nested_loop_count"] = min(
                features["nested_loop_count"], max(0, features["loop_count"] - 1)
            )

        # lloc ≈ sloc (roughly)
        features["lloc"] = max(features["sloc"] - random.randint(0, 10), 1)

        # Generate "actual" energy
        actual_energy = _generate_synthetic_energy(features)
        features["actual_energy_joules"] = actual_energy

        records.append(features)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Physics estimate for training data
# ──────────────────────────────────────────────────────────────────────────────

def _physics_estimate_for_row(row: dict) -> float:
    """Compute physics-based energy estimate for a training data row."""
    from core.physics_estimator import estimate_energy

    # Determine workload type from features
    if row.get("network_calls", 0) > 0:
        workload = "network"
    elif row.get("io_operations", 0) >= 3:
        workload = "io_heavy"
    elif row.get("heavy_math_imports", 0) > 0 or row.get("computational_intensity", 0) > 50:
        workload = "cpu_heavy"
    elif row.get("sloc", 10) < 10:
        workload = "trivial"
    else:
        workload = "mixed"

    features_with_type = {**row, "workload_type": workload}
    result = estimate_energy(features_with_type)
    return result["energy_joules"]


# ──────────────────────────────────────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────────────────────────────────────

def train_residual_model(n_samples: int = 2000, seed: int = 42) -> dict:
    """
    Train XGBoost residual model:
    1. Generate synthetic data
    2. Compute physics estimates for each sample
    3. Compute residual = actual - physics
    4. Train XGBoost to predict residual from features
    5. Save model

    Returns training metrics.
    """
    print("🔧 Generating synthetic training data...")
    df = generate_training_data(n_samples, seed)

    print("⚡ Computing physics estimates...")
    physics_estimates = []
    for _, row in df.iterrows():
        pe = _physics_estimate_for_row(row.to_dict())
        physics_estimates.append(pe)
    df["physics_energy"] = physics_estimates

    # Residual = actual - physics (what the physics model gets wrong)
    df["residual"] = df["actual_energy_joules"] - df["physics_energy"]

    # Prepare features and target
    X = df[FEATURE_KEYS].values
    y = df["residual"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print("🤖 Training XGBoost residual model...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    # Hybrid predictions on test set
    test_indices = df.index[len(X_train):]
    physics_test = df.loc[test_indices, "physics_energy"].values
    actual_test = df.loc[test_indices, "actual_energy_joules"].values

    hybrid_pred = physics_test + y_pred

    # Metrics: physics-only vs hybrid
    physics_mape = mean_absolute_percentage_error(actual_test, physics_test) * 100
    hybrid_mape = mean_absolute_percentage_error(actual_test, hybrid_pred) * 100
    physics_r2 = r2_score(actual_test, physics_test)
    hybrid_r2 = r2_score(actual_test, hybrid_pred)

    metrics = {
        "n_samples": n_samples,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "physics_only": {
            "mape": round(physics_mape, 2),
            "r2": round(physics_r2, 4),
        },
        "hybrid_model": {
            "mape": round(hybrid_mape, 2),
            "r2": round(hybrid_r2, 4),
        },
        "improvement": {
            "mape_reduction_pct": round((physics_mape - hybrid_mape) / physics_mape * 100, 1),
            "r2_improvement": round(hybrid_r2 - physics_r2, 4),
        },
    }

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "residual_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

    # Save feature keys
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"feature_keys": FEATURE_KEYS, "metrics": metrics}, f, indent=2)
    print(f"📋 Metadata saved to {meta_path}")

    # Save training data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "training_data.csv", index=False)
    print(f"📊 Training data saved to {DATA_DIR / 'training_data.csv'}")

    print("\n✅ Training complete!")
    print(f"   Physics-only  MAPE: {metrics['physics_only']['mape']:.2f}%  R²: {metrics['physics_only']['r2']:.4f}")
    print(f"   Hybrid model  MAPE: {metrics['hybrid_model']['mape']:.2f}%  R²: {metrics['hybrid_model']['r2']:.4f}")
    print(f"   Improvement:  {metrics['improvement']['mape_reduction_pct']:.1f}% MAPE reduction")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    train_residual_model()
