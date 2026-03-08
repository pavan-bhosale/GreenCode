"""
GreenCode Physics-Based Energy Estimator
=========================================
Estimates energy consumption using physics-based power models.
Equation references from the GreenCode research paper.
"""

import math


# ──────────────────────────────────────────────────────────────────────────────
# Hardware Power Constants (Watts) — based on typical cloud instance (t3.medium)
# ──────────────────────────────────────────────────────────────────────────────

# Intel Xeon (typical cloud vCPU)
CPU_IDLE_POWER = 5.0        # W — idle power per vCPU
CPU_MAX_POWER = 25.0        # W — max power per vCPU
CPU_VCPUS = 2               # t3.medium has 2 vCPUs

# Memory (4 GB for t3.medium)
MEM_POWER_PER_GB = 0.375    # W per GB (DDR4 typical)
MEM_GB = 4.0

# Storage I/O
IO_IDLE_POWER = 0.5         # W
IO_ACTIVE_POWER = 3.0       # W per active I/O stream

# Network
NET_IDLE_POWER = 0.2        # W
NET_ACTIVE_POWER = 2.0      # W per active connection

# Power Usage Effectiveness (data center overhead — cooling, etc.)
PUE = 1.2  # typical cloud PUE


# ──────────────────────────────────────────────────────────────────────────────
# Workload profiles — mapping workload_type → utilization factors
# ──────────────────────────────────────────────────────────────────────────────

WORKLOAD_PROFILES = {
    "cpu_heavy": {
        "cpu_utilization": 0.85,
        "mem_utilization": 0.60,
        "io_utilization": 0.10,
        "net_utilization": 0.05,
    },
    "io_heavy": {
        "cpu_utilization": 0.30,
        "mem_utilization": 0.40,
        "io_utilization": 0.80,
        "net_utilization": 0.05,
    },
    "network": {
        "cpu_utilization": 0.25,
        "mem_utilization": 0.30,
        "io_utilization": 0.15,
        "net_utilization": 0.85,
    },
    "mixed": {
        "cpu_utilization": 0.50,
        "mem_utilization": 0.45,
        "io_utilization": 0.30,
        "net_utilization": 0.15,
    },
    "trivial": {
        "cpu_utilization": 0.10,
        "mem_utilization": 0.10,
        "io_utilization": 0.05,
        "net_utilization": 0.02,
    },
}


def estimate_runtime_seconds(features: dict) -> float:
    """
    Estimate code runtime in seconds from static features.
    
    Uses a heuristic model based on code complexity, loop depth, 
    and I/O patterns. This is a rough estimate — the ML residual 
    model corrects for systematic errors.
    """
    sloc = features.get("sloc", 10)
    loop_count = features.get("loop_count", 0)
    max_depth = features.get("max_loop_depth", 0)
    nested = features.get("nested_loop_count", 0)
    complexity = features.get("avg_complexity", 1)
    io_ops = features.get("io_operations", 0)
    net_calls = features.get("network_calls", 0)
    intensity = features.get("computational_intensity", 10)

    # Base time: assume ~0.001s per SLOC for simple sequential code
    base_time = sloc * 0.001

    # Loop multiplier: nested loops are exponentially more expensive
    loop_factor = 1.0
    if loop_count > 0:
        # Assume ~100 iterations per loop level
        loop_factor = max(1.0, (100 ** max_depth) * 0.00001)
        loop_factor = min(loop_factor, 300.0)  # cap at 5 minutes

    # I/O factor: each I/O op adds ~10ms, network adds ~100ms
    io_time = io_ops * 0.01 + net_calls * 0.1

    # Complexity scaling
    complexity_factor = 1.0 + (complexity - 1) * 0.1

    estimated = (base_time * loop_factor * complexity_factor) + io_time

    # Scale by intensity
    estimated *= (1.0 + intensity / 100.0)

    return max(0.001, round(estimated, 6))


def estimate_power_watts(features: dict) -> dict:
    """
    Equation (1): P_total = P_cpu + P_mem + P_io + P_net
    
    Each component computed as:
        P_component = P_idle + (P_max - P_idle) × utilization
    
    Returns power breakdown in Watts for each component.
    """
    workload = features.get("workload_type", "mixed")
    profile = WORKLOAD_PROFILES.get(workload, WORKLOAD_PROFILES["mixed"])

    # CPU power (equation 1a)
    cpu_util = profile["cpu_utilization"]
    p_cpu = CPU_VCPUS * (CPU_IDLE_POWER + (CPU_MAX_POWER - CPU_IDLE_POWER) * cpu_util)

    # Memory power (equation 1b)
    mem_util = profile["mem_utilization"]
    p_mem = MEM_GB * MEM_POWER_PER_GB * (0.5 + 0.5 * mem_util)

    # I/O power (equation 1c)
    io_util = profile["io_utilization"]
    p_io = IO_IDLE_POWER + (IO_ACTIVE_POWER - IO_IDLE_POWER) * io_util

    # Network power (equation 1d)
    net_util = profile["net_utilization"]
    p_net = NET_IDLE_POWER + (NET_ACTIVE_POWER - NET_IDLE_POWER) * net_util

    # Total with PUE overhead
    p_total = (p_cpu + p_mem + p_io + p_net) * PUE

    return {
        "cpu_watts": round(p_cpu, 4),
        "memory_watts": round(p_mem, 4),
        "io_watts": round(p_io, 4),
        "network_watts": round(p_net, 4),
        "total_watts": round(p_total, 4),
        "pue": PUE,
        "workload_profile": workload,
    }


def estimate_energy(features: dict) -> dict:
    """
    Equation (2): E = P_total × t
    
    Combines power estimation with runtime estimation.
    Returns energy in Joules and kWh.
    """
    power = estimate_power_watts(features)
    runtime_s = estimate_runtime_seconds(features)

    energy_joules = power["total_watts"] * runtime_s
    energy_kwh = energy_joules / 3_600_000  # 1 kWh = 3.6 MJ

    return {
        "runtime_seconds": round(runtime_s, 6),
        "power": power,
        "energy_joules": round(energy_joules, 6),
        "energy_kwh": round(energy_kwh, 10),
        "energy_wh": round(energy_kwh * 1000, 8),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample_features = {
        "sloc": 25,
        "loop_count": 2,
        "max_loop_depth": 2,
        "nested_loop_count": 1,
        "avg_complexity": 3.5,
        "io_operations": 2,
        "network_calls": 0,
        "computational_intensity": 55,
        "workload_type": "cpu_heavy",
    }
    result = estimate_energy(sample_features)
    print(json.dumps(result, indent=2))
