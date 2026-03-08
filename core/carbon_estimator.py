"""
GreenCode Carbon Estimator
===========================
Converts energy consumption to CO₂ emissions using region-specific
carbon intensity factors.

Equation (4): CO₂(g) = E(kWh) × CI(gCO₂/kWh)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Regional Carbon Intensity Data (gCO₂ per kWh)
# Sources: IEA 2023, Ember Climate, electricitymaps.com
# ──────────────────────────────────────────────────────────────────────────────

CARBON_INTENSITY = {
    # Asia
    "India":        709,
    "China":        555,
    "Japan":        471,
    "South Korea":  415,
    "Singapore":    408,
    # Americas
    "US":           386,
    "US-California":220,
    "US-Texas":     350,
    "Canada":       120,
    "Brazil":       75,
    # Europe
    "EU-Average":   231,
    "Germany":      338,
    "UK":           233,
    "France":       56,
    "Sweden":       13,
    "Norway":       8,
    # Oceania
    "Australia":    540,
    "New Zealand":  105,
}

# Friendly display with flag emojis
REGION_DISPLAY = {
    "India":        "🇮🇳 India",
    "China":        "🇨🇳 China",
    "Japan":        "🇯🇵 Japan",
    "South Korea":  "🇰🇷 South Korea",
    "Singapore":    "🇸🇬 Singapore",
    "US":           "🇺🇸 United States",
    "US-California":"🇺🇸 US (California)",
    "US-Texas":     "🇺🇸 US (Texas)",
    "Canada":       "🇨🇦 Canada",
    "Brazil":       "🇧🇷 Brazil",
    "EU-Average":   "🇪🇺 EU Average",
    "Germany":      "🇩🇪 Germany",
    "UK":           "🇬🇧 United Kingdom",
    "France":       "🇫🇷 France",
    "Sweden":       "🇸🇪 Sweden",
    "Norway":       "🇳🇴 Norway",
    "Australia":    "🇦🇺 Australia",
    "New Zealand":  "🇳🇿 New Zealand",
}


def get_available_regions() -> list[str]:
    """Return list of available region keys."""
    return list(CARBON_INTENSITY.keys())


def get_region_display(region: str) -> str:
    """Return display string with flag emoji for a region."""
    return REGION_DISPLAY.get(region, region)


def get_carbon_intensity(region: str) -> float:
    """
    Get carbon intensity for a region in gCO₂/kWh.
    Falls back to global average if region not found.
    """
    return CARBON_INTENSITY.get(region, 442)  # 442 = world average


def estimate_carbon(energy_kwh: float, region: str = "India") -> dict:
    """
    Equation (4): CO₂(g) = E(kWh) × CI(gCO₂/kWh)

    Parameters
    ----------
    energy_kwh : float
        Energy consumption in kilowatt-hours.
    region : str
        Region key (e.g., "India", "US", "France").

    Returns
    -------
    dict
        Carbon emissions breakdown.
    """
    ci = get_carbon_intensity(region)
    co2_grams = energy_kwh * ci
    co2_kg = co2_grams / 1000

    # Context: equivalent everyday activities
    # Average tree absorbs ~22kg CO₂/year ≈ 60g/day
    trees_equivalent = co2_grams / 60 if co2_grams > 0 else 0

    # A Google search ≈ 0.2g CO₂
    google_searches = co2_grams / 0.2 if co2_grams > 0 else 0

    # Smartphone charge ≈ 8g CO₂
    phone_charges = co2_grams / 8 if co2_grams > 0 else 0

    return {
        "region": region,
        "region_display": get_region_display(region),
        "carbon_intensity_gco2_kwh": ci,
        "co2_grams": round(co2_grams, 6),
        "co2_kg": round(co2_kg, 9),
        "equivalents": {
            "google_searches": round(google_searches, 2),
            "phone_charges": round(phone_charges, 4),
            "tree_days_to_offset": round(trees_equivalent, 4),
        },
    }


def compare_regions(energy_kwh: float, regions: list[str] | None = None) -> list[dict]:
    """
    Compare carbon emissions across multiple regions for the same energy.
    Useful for showing how location matters.
    """
    if regions is None:
        regions = ["India", "US", "EU-Average", "Canada", "France", "Sweden"]

    results = []
    for region in regions:
        result = estimate_carbon(energy_kwh, region)
        results.append(result)

    # Sort by CO₂ descending
    results.sort(key=lambda x: x["co2_grams"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    energy = 0.001  # 1 Wh = 0.001 kWh
    print("=== Single region ===")
    print(json.dumps(estimate_carbon(energy, "India"), indent=2))

    print("\n=== Regional comparison ===")
    for r in compare_regions(energy):
        print(f"  {r['region_display']:25s} → {r['co2_grams']:.4f}g CO₂")
