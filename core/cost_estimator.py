"""
GreenCode Cloud Cost Estimator
================================
Estimates cloud execution cost based on runtime and instance pricing.

Equation (5): Cost = (runtime / 3600) × price_per_hour
"""

# ──────────────────────────────────────────────────────────────────────────────
# Cloud Instance Pricing (USD/hour) — On-Demand, Linux, US regions
# ──────────────────────────────────────────────────────────────────────────────

INSTANCE_PRICING = {
    # AWS
    "aws-t3.micro":    {"price": 0.0104, "vcpus": 2, "mem_gb": 1,  "provider": "AWS"},
    "aws-t3.small":    {"price": 0.0208, "vcpus": 2, "mem_gb": 2,  "provider": "AWS"},
    "aws-t3.medium":   {"price": 0.0416, "vcpus": 2, "mem_gb": 4,  "provider": "AWS"},
    "aws-t3.large":    {"price": 0.0832, "vcpus": 2, "mem_gb": 8,  "provider": "AWS"},
    "aws-m5.large":    {"price": 0.0960, "vcpus": 2, "mem_gb": 8,  "provider": "AWS"},
    "aws-c5.large":    {"price": 0.0850, "vcpus": 2, "mem_gb": 4,  "provider": "AWS"},
    # GCP
    "gcp-e2-medium":   {"price": 0.0335, "vcpus": 2, "mem_gb": 4,  "provider": "GCP"},
    "gcp-e2-standard-2":{"price": 0.0670, "vcpus": 2, "mem_gb": 8, "provider": "GCP"},
    "gcp-n1-standard-1":{"price": 0.0475, "vcpus": 1, "mem_gb": 3.75, "provider": "GCP"},
    # Azure
    "azure-B2s":       {"price": 0.0416, "vcpus": 2, "mem_gb": 4,  "provider": "Azure"},
    "azure-D2s-v3":    {"price": 0.0960, "vcpus": 2, "mem_gb": 8,  "provider": "Azure"},
}

DEFAULT_INSTANCE = "aws-t3.medium"


def get_available_instances() -> list[str]:
    """Return list of available instance type keys."""
    return list(INSTANCE_PRICING.keys())


def get_instance_info(instance: str) -> dict:
    """Return pricing and spec info for an instance type."""
    return INSTANCE_PRICING.get(instance, INSTANCE_PRICING[DEFAULT_INSTANCE])


def estimate_cost(runtime_seconds: float, instance: str = DEFAULT_INSTANCE) -> dict:
    """
    Equation (5): Cost = (runtime_seconds / 3600) × price_per_hour

    Parameters
    ----------
    runtime_seconds : float
        Estimated runtime in seconds.
    instance : str
        Cloud instance type key.

    Returns
    -------
    dict
        Cost breakdown with per-second, per-minute, per-hour rates.
    """
    info = get_instance_info(instance)
    price_per_hour = info["price"]
    price_per_second = price_per_hour / 3600
    price_per_minute = price_per_hour / 60

    cost = (runtime_seconds / 3600) * price_per_hour

    # Monthly cost if running continuously
    monthly_hours = 730  # avg hours/month
    monthly_cost = price_per_hour * monthly_hours

    # Annual cost estimate (if this task ran once per minute 24/7)
    runs_per_year = 525_960  # minutes in a year
    annual_cost_continuous = cost * runs_per_year

    return {
        "instance_type": instance,
        "provider": info["provider"],
        "vcpus": info["vcpus"],
        "memory_gb": info["mem_gb"],
        "price_per_hour": price_per_hour,
        "runtime_seconds": round(runtime_seconds, 6),
        "estimated_cost_usd": round(cost, 10),
        "cost_per_1000_runs": round(cost * 1000, 6),
        "cost_per_million_runs": round(cost * 1_000_000, 4),
        "monthly_instance_cost": round(monthly_cost, 2),
    }


def compare_instances(runtime_seconds: float, instances: list[str] | None = None) -> list[dict]:
    """
    Compare cost across multiple cloud instances for the same workload.
    """
    if instances is None:
        instances = [
            "aws-t3.micro", "aws-t3.medium", "aws-t3.large",
            "gcp-e2-medium", "azure-B2s",
        ]

    results = []
    for inst in instances:
        if inst in INSTANCE_PRICING:
            results.append(estimate_cost(runtime_seconds, inst))

    results.sort(key=lambda x: x["estimated_cost_usd"])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    runtime = 2.5  # seconds
    print("=== Default instance ===")
    print(json.dumps(estimate_cost(runtime), indent=2))

    print("\n=== Instance comparison ===")
    for r in compare_instances(runtime):
        print(f"  {r['instance_type']:25s} → ${r['estimated_cost_usd']:.10f}")
