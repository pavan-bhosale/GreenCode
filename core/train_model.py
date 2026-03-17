"""
GreenCode — Model Trainer
==========================
Generates realistic synthetic training data that matches the scale of
the physics_estimator output, then trains the XGBoost residual model.

Run from project root:
    python -m core.train_model

or directly:
    python core/train_model.py
"""

import json
import math
import random
import numpy as np
from pathlib import Path

# ── Try imports ──────────────────────────────────────────────────────────────
try:
    import joblib
except ImportError:
    raise ImportError("Run: pip install joblib")

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Run: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    raise ImportError("Run: pip install scikit-learn")


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODELS_DIR / "residual_model.pkl"
META_PATH    = MODELS_DIR / "model_meta.json"


# ── Feature keys (must match static_analyzer output) ─────────────────────────
FEATURE_KEYS = [
    "sloc", "lloc", "function_count", "class_count",
    "loop_count", "max_loop_depth", "nested_loop_count",
    "conditional_count", "try_except_count", "list_comprehensions",
    "io_operations", "network_calls", "heavy_math_imports",
    "recursion_candidates", "avg_complexity", "max_complexity",
    "total_complexity", "halstead_volume", "halstead_difficulty",
    "halstead_effort", "maintainability_index", "computational_intensity",
]


# ── Physics estimator (mirror of physics_estimator.py) ───────────────────────
# We inline a simplified version here so train_model.py is self-contained
# and doesn't import the full app stack.

CPU_IDLE = 5.0;  CPU_MAX = 25.0;  CPU_VCPUS = 2
MEM_PER_GB = 0.375; MEM_GB = 4.0
IO_IDLE = 0.5;   IO_ACTIVE = 3.0
NET_IDLE = 0.2;  NET_ACTIVE = 2.0
PUE = 1.2

WORKLOAD_PROFILES = {
    "cpu_heavy": (0.85, 0.60, 0.10, 0.05),
    "io_heavy":  (0.30, 0.40, 0.80, 0.05),
    "network":   (0.25, 0.30, 0.15, 0.85),
    "mixed":     (0.50, 0.45, 0.30, 0.15),
    "trivial":   (0.10, 0.10, 0.05, 0.02),
}

def _physics_power(workload: str) -> float:
    cu, mu, iu, nu = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["mixed"])
    p_cpu = CPU_VCPUS * (CPU_IDLE + (CPU_MAX - CPU_IDLE) * cu)
    p_mem = MEM_GB * MEM_PER_GB * (0.5 + 0.5 * mu)
    p_io  = IO_IDLE  + (IO_ACTIVE  - IO_IDLE)  * iu
    p_net = NET_IDLE + (NET_ACTIVE - NET_IDLE) * nu
    return (p_cpu + p_mem + p_io + p_net) * PUE

def _physics_runtime(feats: dict) -> float:
    sloc     = feats.get("sloc", 10)
    depth    = feats.get("max_loop_depth", 0)
    loops    = feats.get("loop_count", 0)
    io_ops   = feats.get("io_operations", 0)
    net      = feats.get("network_calls", 0)
    cmplx    = feats.get("avg_complexity", 1.0)
    intensity= feats.get("computational_intensity", 10)

    base = sloc * 0.001
    loop_f = 1.0
    if loops > 0:
        loop_f = 1.0 + (depth * depth * 2.5)
        loop_f = min(loop_f, 20.0)
    io_time  = io_ops * 0.01 + net * 0.08
    cmplx_f  = 1.0 + (cmplx - 1) * 0.05
    intensity_f = 1.0 + intensity / 200.0
    est = (base * loop_f * cmplx_f * intensity_f) + io_time
    return max(0.001, min(est, 10.0))

def physics_energy(feats: dict) -> float:
    """Returns energy in Joules — same formula as physics_estimator.py."""
    return _physics_power(feats.get("workload_type", "mixed")) * _physics_runtime(feats)


# ── Synthetic data generator ──────────────────────────────────────────────────

def _workload_type(feats: dict) -> str:
    if feats["network_calls"] > 0:
        return "network"
    if feats["io_operations"] >= 3:
        return "io_heavy"
    if feats["heavy_math_imports"] > 0 or feats["computational_intensity"] > 50:
        return "cpu_heavy"
    if feats["sloc"] < 10:
        return "trivial"
    return "mixed"


def _computational_intensity(feats: dict) -> float:
    return min(100.0, (
        feats["loop_count"]        * 10
        + feats["nested_loop_count"] * 20
        + feats["max_loop_depth"]    * 15
        + feats["heavy_math_imports"]* 15
        + feats["recursion_candidates"] * 10
        + feats["avg_complexity"]    * 3
    ))


def generate_sample(workload_hint: str | None = None) -> tuple[dict, float]:
    """
    Generate one realistic synthetic code feature sample.
    Returns (features_dict, true_energy_joules).

    Strategy:
    - Sample code-structure features from realistic distributions
    - Compute physics energy from those features (our "ground truth" baseline)
    - Add a small realistic residual (the thing our ML model should learn)
    - The residual is proportional to physics energy → keeps scales consistent
    """
    rng = random.random

    # ── Choose workload class ─────────────────────────────────────────
    workload = workload_hint or random.choice([
        "trivial", "trivial",
        "mixed", "mixed", "mixed",
        "cpu_heavy", "cpu_heavy",
        "io_heavy",
        "network",
    ])

    # ── Structural features by workload ──────────────────────────────
    if workload == "trivial":
        sloc            = random.randint(3, 20)
        loop_count      = random.randint(0, 1)
        max_loop_depth  = min(loop_count, 1)
        nested          = 0
        io_ops          = random.randint(0, 1)
        net_calls       = 0
        heavy_math      = 0
        avg_cmplx       = round(random.uniform(1.0, 2.0), 2)

    elif workload == "cpu_heavy":
        sloc            = random.randint(20, 200)
        loop_count      = random.randint(2, 6)
        max_loop_depth  = random.randint(2, min(loop_count, 3))  # capped at 3
        nested          = random.randint(1, max(1, max_loop_depth - 1))
        io_ops          = random.randint(0, 2)
        net_calls       = 0
        heavy_math      = random.randint(0, 3)
        avg_cmplx       = round(random.uniform(3.0, 10.0), 2)

    elif workload == "io_heavy":
        sloc            = random.randint(15, 150)
        loop_count      = random.randint(1, 4)
        max_loop_depth  = random.randint(1, 2)
        nested          = 0
        io_ops          = random.randint(3, 15)
        net_calls       = 0
        heavy_math      = 0
        avg_cmplx       = round(random.uniform(2.0, 6.0), 2)

    elif workload == "network":
        sloc            = random.randint(10, 120)
        loop_count      = random.randint(0, 3)
        max_loop_depth  = random.randint(0, 2)
        nested          = 0
        io_ops          = random.randint(0, 3)
        net_calls       = random.randint(1, 8)
        heavy_math      = 0
        avg_cmplx       = round(random.uniform(2.0, 7.0), 2)

    else:  # mixed
        sloc            = random.randint(10, 200)
        loop_count      = random.randint(0, 5)
        max_loop_depth  = random.randint(0, min(loop_count + 1, 4))
        nested          = random.randint(0, max(0, max_loop_depth - 1))
        io_ops          = random.randint(0, 5)
        net_calls       = random.randint(0, 2)
        heavy_math      = random.randint(0, 2)
        avg_cmplx       = round(random.uniform(1.5, 8.0), 2)

    # Derived / secondary features
    function_count   = max(1, sloc // random.randint(8, 20))
    class_count      = random.randint(0, max(1, function_count // 3))
    conditional_count= random.randint(0, loop_count * 2 + 2)
    try_except       = random.randint(0, max(1, io_ops))
    list_comp        = random.randint(0, 3)
    dict_comp        = random.randint(0, 2)
    gen_exp          = random.randint(0, 2)
    lambda_count     = random.randint(0, 2)
    recursion        = random.randint(0, min(2, function_count))
    max_cmplx        = int(avg_cmplx * random.uniform(1.2, 2.5))
    total_cmplx      = int(avg_cmplx * function_count)
    lloc             = max(sloc, sloc + random.randint(0, sloc // 3))

    # Halstead approximations (proportional to code size & complexity)
    halstead_volume  = round(sloc * avg_cmplx * random.uniform(8, 20), 2)
    halstead_diff    = round(avg_cmplx * random.uniform(2, 8), 2)
    halstead_effort  = round(halstead_volume * halstead_diff, 2)
    maintainability  = round(max(0, min(100,
        171 - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * total_cmplx
            - 16.2 * math.log(max(1, sloc))
    )), 2)

    feats = {
        "sloc": sloc, "lloc": lloc,
        "function_count": function_count, "class_count": class_count,
        "loop_count": loop_count, "max_loop_depth": max_loop_depth,
        "nested_loop_count": nested, "conditional_count": conditional_count,
        "try_except_count": try_except, "list_comprehensions": list_comp,
        "dict_comprehensions": dict_comp, "generator_expressions": gen_exp,
        "lambda_count": lambda_count, "recursion_candidates": recursion,
        "io_operations": io_ops, "network_calls": net_calls,
        "heavy_math_imports": heavy_math,
        "avg_complexity": avg_cmplx, "max_complexity": max_cmplx,
        "total_complexity": total_cmplx,
        "halstead_volume": halstead_volume, "halstead_difficulty": halstead_diff,
        "halstead_effort": halstead_effort, "halstead_bugs": round(halstead_volume / 3000, 4),
        "maintainability_index": maintainability,
        "yield_count": 0, "await_count": random.randint(0, net_calls),
        "decorator_count": random.randint(0, 2),
        "global_variables": random.randint(0, 3),
    }

    # Derived workload fields
    feats["computational_intensity"] = _computational_intensity(feats)
    feats["workload_type"] = _workload_type(feats)

    # ── Physics baseline ──────────────────────────────────────────────
    e_phys = physics_energy(feats)

    # ── Realistic residual ────────────────────────────────────────────
    # Deterministic feature-driven corrections (the learnable signal).
    # Small Gaussian noise only — so XGBoost learns real patterns,
    # not random noise.
    residual_pct = 0.0
    residual_pct += recursion        * 0.08   # recursion always costs more
    residual_pct -= heavy_math       * 0.06   # numpy is efficient per op
    residual_pct += nested           * 0.07   # deep nesting costs more
    residual_pct -= list_comp        * 0.03   # comprehensions faster than loops
    residual_pct += net_calls        * 0.10   # network latency overhead
    residual_pct += io_ops           * 0.02   # each IO op adds overhead
    residual_pct += (avg_cmplx - 1)  * 0.02  # higher complexity = more energy
    residual_pct -= (sloc / 500.0)   * 0.05  # larger files = better CPU cache

    # Small Gaussian noise only (std=5%, not uniform ±30%)
    noise = random.gauss(0, 0.05)
    residual_pct += noise

    # Clamp to ±35% of physics
    residual_pct = max(-0.35, min(0.35, residual_pct))
    e_residual = e_phys * residual_pct

    # True energy = physics + residual (what model should predict)
    e_true = max(1e-6, e_phys + e_residual)

    return feats, e_true, e_residual


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset(n_samples: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_samples synthetic (features, residual) pairs."""
    print(f"🔄 Generating {n_samples} synthetic training samples...")

    X_rows, y_rows = [], []
    workloads = ["trivial", "mixed", "cpu_heavy", "io_heavy", "network"]

    for i in range(n_samples):
        # Balanced workload distribution
        hint = workloads[i % len(workloads)]
        feats, e_true, e_residual = generate_sample(hint)

        # Build feature vector in the exact order of FEATURE_KEYS
        row = [float(feats.get(k, 0)) for k in FEATURE_KEYS]
        X_rows.append(row)
        y_rows.append(e_residual)  # ← model learns to predict the residual

        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{n_samples} samples generated...")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Residual range: [{y.min():.6f}, {y.max():.6f}] J")
    print(f"   Residual mean:  {y.mean():.6f} J")
    return X, y


# ── Train ─────────────────────────────────────────────────────────────────────

def train(n_samples: int = 10000):
    print("\n🌿 GreenCode Model Trainer")
    print("=" * 50)

    # 1. Generate data
    X, y = build_dataset(n_samples)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 3. Train XGBoost with a pipeline (scaler + model)
    print("\n🤖 Training XGBoost residual model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=1.0,
            reg_lambda=5.0,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ))
    ])
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # ── Stable metrics: SMAPE + MAE (MAPE breaks when residuals near zero) ─
    def smape(y_true, y_pred):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask  = denom > 1e-9
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    train_smape = smape(y_train, y_pred_train)
    test_smape  = smape(y_test,  y_pred_test)
    train_mae   = mae(y_train, y_pred_train)
    test_mae    = mae(y_test,  y_pred_test)
    train_r2    = float(r2_score(y_train, y_pred_train))
    test_r2     = float(r2_score(y_test,  y_pred_test))

    print(f"\n📈 Results:")
    print(f"   Train  SMAPE: {train_smape:.2f}%  |  MAE: {train_mae:.6f} J  |  R²: {train_r2:.4f}")
    print(f"   Test   SMAPE: {test_smape:.2f}%  |  MAE: {test_mae:.6f} J  |  R²: {test_r2:.4f}")

    # 5. Physics-only baseline (predicting zero residual)
    phys_smape = smape(y_test, np.zeros_like(y_test))
    phys_mae   = mae(y_test,   np.zeros_like(y_test))
    print(f"\n🔬 Physics-only baseline  SMAPE: {phys_smape:.2f}%  |  MAE: {phys_mae:.6f} J")
    print(f"   ML improvement:   SMAPE: {phys_smape - test_smape:+.2f}%  |  MAE: {phys_mae - test_mae:+.6f} J")

    # 6. Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # 7. Save metadata
    meta = {
        "feature_keys": FEATURE_KEYS,
        "n_features": len(FEATURE_KEYS),
        "metrics": {
            "n_samples": n_samples,
            "n_train": len(X_train),
            "n_test":  len(X_test),
            "physics_only": {
                "smape": round(phys_smape, 2),
                "mae_joules": round(phys_mae, 6),
            },
            "hybrid_model": {
                "train_smape": round(train_smape, 2),
                "test_smape":  round(test_smape,  2),
                "train_mae_j": round(train_mae,   6),
                "test_mae_j":  round(test_mae,    6),
                "train_r2":    round(train_r2,    4),
                "test_r2":     round(test_r2,     4),
            },
            "improvement": {
                "smape_reduction_pct": round(phys_smape - test_smape, 2),
                "mae_reduction_j":     round(phys_mae   - test_mae,   6),
            },
        },
        "model_type": "Pipeline(StandardScaler + XGBRegressor)",
        "target": "residual_joules",
        "note": "Model predicts E_residual = E_true - E_physics. Hybrid = Physics + Residual.",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"📋 Metadata saved → {META_PATH}")

    # 8. Quick sanity check
    print("\n🧪 Sanity check on 3 real-world-like snippets:")
    test_cases = [
        {"name": "Simple loop (7 lines)", "feats": {
            "sloc":7,"lloc":7,"loop_count":1,"max_loop_depth":1,"nested_loop_count":0,
            "function_count":1,"class_count":0,"conditional_count":1,"try_except_count":0,
            "list_comprehensions":0,"io_operations":0,"network_calls":0,"heavy_math_imports":0,
            "recursion_candidates":0,"avg_complexity":2.0,"max_complexity":2,"total_complexity":2,
            "halstead_volume":80,"halstead_difficulty":4,"halstead_effort":320,
            "maintainability_index":72,"computational_intensity":25,"dict_comprehensions":0,
            "generator_expressions":0,"lambda_count":0,
        }},
        {"name": "Bubble sort (15 lines)", "feats": {
            "sloc":15,"lloc":15,"loop_count":2,"max_loop_depth":2,"nested_loop_count":1,
            "function_count":1,"class_count":0,"conditional_count":2,"try_except_count":0,
            "list_comprehensions":0,"io_operations":0,"network_calls":0,"heavy_math_imports":0,
            "recursion_candidates":0,"avg_complexity":4.0,"max_complexity":4,"total_complexity":4,
            "halstead_volume":200,"halstead_difficulty":8,"halstead_effort":1600,
            "maintainability_index":60,"computational_intensity":65,"dict_comprehensions":0,
            "generator_expressions":0,"lambda_count":0,
        }},
        {"name": "API call (20 lines)", "feats": {
            "sloc":20,"lloc":20,"loop_count":0,"max_loop_depth":0,"nested_loop_count":0,
            "function_count":2,"class_count":0,"conditional_count":3,"try_except_count":1,
            "list_comprehensions":0,"io_operations":1,"network_calls":3,"heavy_math_imports":0,
            "recursion_candidates":0,"avg_complexity":3.0,"max_complexity":4,"total_complexity":6,
            "halstead_volume":150,"halstead_difficulty":6,"halstead_effort":900,
            "maintainability_index":68,"computational_intensity":9,"dict_comprehensions":0,
            "generator_expressions":0,"lambda_count":0,
        }},
    ]

    for tc in test_cases:
        feats = tc["feats"]
        feats["workload_type"] = _workload_type(feats)
        feats["computational_intensity"] = _computational_intensity(feats)

        e_phys = physics_energy(feats)
        row = np.array([[float(feats.get(k, 0)) for k in FEATURE_KEYS]], dtype=np.float32)
        e_residual = float(model.predict(row)[0])
        e_hybrid = max(0.0001, e_phys + e_residual)

        print(f"\n   [{tc['name']}]")
        print(f"     Physics:  {e_phys:.6f} J")
        print(f"     Residual: {e_residual:+.6f} J  ({e_residual/e_phys*100:+.1f}%)")
        print(f"     Hybrid:   {e_hybrid:.6f} J")
        print(f"     CO₂ (India): {e_hybrid/3_600_000 * 720 * 1000:.6f} g")

    print("\n✅ Training complete! Your model is ready.")
    print(f"   Run your Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    train(n)