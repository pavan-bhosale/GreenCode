"""
GreenCode Streamlit Dashboard
==============================
Interactive UI for predicting energy, carbon, and cost of Python code.
"""

import streamlit as st
import pandas as pd
import json

from core.hybrid_predictor import predict, predict_and_compare
from core.carbon_estimator import get_available_regions, get_region_display
from core.cost_estimator import get_available_instances
from utils.visualizer import (
    plot_energy_breakdown,
    plot_carbon_pie,
    plot_cost_projection,
    render_green_score_badge,
)

# ─── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GreenCode Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Example Snippets ─────────────────────────────────────────────────────────

EXAMPLES = {
    "Simple Loop": """def count_to_n(n):
    total = 0
    for i in range(n):
        total += i
    return total

print(count_to_n(1000000))""",

    "Matrix Mult (CPU Heavy)": """import numpy as np

def matrix_multiply(size=100):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    result = np.dot(a, b)
    
    # Unnecessary nested loops for demonstration
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = round(result[i][j], 2)
            
    return result

matrix_multiply(200)""",

    "File I/O (Disk Heavy)": """import os

def process_file_lines():
    with open("temp.txt", "w") as f:
        for i in range(5000):
            f.write(f"Line number {i}\\n")
            
    count = 0
    with open("temp.txt", "r") as f:
        for line in f:
            if "5" in line:
                count += 1
                
    os.remove("temp.txt")
    return count

process_file_lines()""",
}


# ─── UI Helper Functions ──────────────────────────────────────────────────────

def render_metric_card(title: str, value: str, subvalue: str, icon: str, color_class: str = ""):
    """Render a nice looking metric card using HTML/CSS."""
    st.markdown(f"""
        <div class="metric-card {color_class}">
            <div class="metric-header">
                <span class="metric-icon">{icon}</span>
                <span class="metric-title">{title}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-subvalue">{subvalue}</div>
        </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 GreenCode")
    st.markdown("*Predict energy, carbon, and cost before deployment.*")
    st.markdown("---")
    
    analysis_mode = st.radio(
        "Mode",
        ["Single File Analysis", "Compare Two Versions"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    
    # Region selector
    regions = get_available_regions()
    region_options = {r: get_region_display(r) for r in regions}
    selected_region = st.selectbox(
        "Deployment Region",
        options=regions,
        format_func=lambda x: region_options[x],
        index=regions.index("India") if "India" in regions else 0,
        help="Carbon intensity varies heavily by region."
    )
    
    # Instance selector
    instances = get_available_instances()
    selected_instance = st.selectbox(
        "Cloud Instance Target",
        options=instances,
        index=instances.index("aws-t3.medium") if "aws-t3.medium" in instances else 0,
    )
    
    use_ml = st.checkbox("Enable ML Residual Correction", value=True, 
                         help="Uses XGBoost model to correct physics-based estimation errors.")

    st.markdown("---")
    st.markdown("Built by **Pavan** & VCET Team")


# ─── Main Content: Single File Analysis ───────────────────────────────────────

if analysis_mode == "Single File Analysis":
    st.markdown("""
    <div class="hero-badge">
        <div class="badge-dot"></div>
        INDIACom-2026 · Track 4 Sustainability · VCET
    </div>
    <h1>Code <em>Greener,</em><br>Ship Smarter</h1>
    <p class="hero-sub">
        GreenCode predicts your code's energy consumption, carbon footprint, and cloud cost
        <em>before</em> you deploy — using a hybrid physics + ML model.
    </p>
    """, unsafe_allow_html=True)
    
    # Code Input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Source Code")
        example = st.selectbox("Load an example template:", ["— Custom Code —"] + list(EXAMPLES.keys()))
        
        default_code = EXAMPLES[example] if example != "— Custom Code —" else EXAMPLES["Simple Loop"]
        
        code_input = st.text_area(
            "Paste Python code here:",
            value=default_code,
            height=400,
            key="code_input"
        )
        
        analyze_btn = st.button("🚀 Analyze Code", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("### Analysis Results")
        
        if analyze_btn or 'last_result' in st.session_state:
            # Show spinner while calculating
            if analyze_btn:
                with st.spinner("Analyzing AST and estimating energy..."):
                    result = predict_and_compare(code_input, regions=[selected_region], instance=selected_instance)
                    st.session_state.last_result = result
            else:
                result = st.session_state.last_result
            
            if "error" in result:
                st.error(f"Failed to parse code: {result['error']}")
            else:
                # 1. Top Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_metric_card(
                        "Energy", 
                        f"{result['energy']['hybrid_joules']:.4f} J", 
                        f"{result['energy']['hybrid_kwh']*1000:.6f} Wh", 
                        "⚡", "energy-card"
                    )
                with m2:
                    render_metric_card(
                        "Carbon footprint", 
                        f"{result['carbon']['co2_grams']:.4f} g", 
                        f"Region: {result['carbon']['region_display']}", 
                        "☁️", "carbon-card"
                    )
                with m3:
                    render_metric_card(
                        "Est. Cost (1M runs)", 
                        f"${result['cost']['cost_per_million_runs']:.4f}", 
                        f"Target: {result['cost']['instance_type']}", 
                        "💸", "cost-card"
                    )
                
                # 2. Green Score Badge
                st.markdown("<br>", unsafe_allow_html=True)
                score_html = render_green_score_badge(result["green_score"])
                st.markdown(score_html, unsafe_allow_html=True)
                
                # tabs for details
                tab_charts, tab_regions, tab_features = st.tabs(["📊 Charts", "🌍 Regional Comparison", "🔍 Diagnostics"])
                
                with tab_charts:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(plot_energy_breakdown(result["component_breakdown_pct"]), use_container_width=True)
                    with c2:
                        st.plotly_chart(plot_carbon_pie(result["component_breakdown_pct"], result["carbon"]["co2_grams"]), use_container_width=True)
                        
                    st.plotly_chart(plot_cost_projection(result["cost"]["estimated_cost_usd"]), use_container_width=True)
                
                with tab_regions:
                    st.markdown("### Carbon Emissions by Region")
                    st.markdown("See how deploying the exact same code to different regions changes its carbon footprint.")
                    
                    # Convert to dataframe for nice table display
                    reg_data = []
                    for r in result["regional_comparison"]:
                        reg_data.append({
                            "Region": r["region_display"],
                            "Carbon Intensity (gCO₂/kWh)": r["carbon_intensity_gco2_kwh"],
                            "Estimated Emissions (g CO₂)": round(r["co2_grams"], 6)
                        })
                    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)
                    
                with tab_features:
                    f1, f2 = st.columns(2)
                    with f1:
                        st.markdown("**Detected Workload Profile:**")
                        st.info(result["features"]["workload_type"].replace("_", " ").title())
                        st.markdown("**Core Metrics:**")
                        st.json({k: v for k,v in result["features"].items() if k in ["sloc", "loop_count", "max_loop_depth", "avg_complexity", "computational_intensity"]})
                    with f2:
                        st.markdown("**Energy Model Details:**")
                        st.json({
                            "physics_estimate_J": result["energy"]["physics_joules"],
                            "ml_correction_J": result["energy"]["ml_residual_joules"],
                            "final_hybrid_J": result["energy"]["hybrid_joules"],
                            "estimated_runtime_s": result["energy"]["runtime_seconds"]
                        })


# ─── Main Content: Comparison Mode ────────────────────────────────────────────

elif analysis_mode == "Compare Two Versions":
    st.markdown("""
    <div class="hero-badge">
        <div class="badge-dot"></div>
        Comparison Mode
    </div>
    <h1>Same Code,<br><em>Different Planet.</em></h1>
    <p class="hero-sub">Paste two implementations of the same feature to see which one is more sustainable and cost-effective.</p>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Version A")
        code_a = st.text_area("Implementation A:", value=EXAMPLES["Simple Loop"], height=300, key="code_a")
    with c2:
        st.markdown("### Version B")
        code_b = st.text_area("Implementation B:", value=EXAMPLES["Matrix Mult (CPU Heavy)"], height=300, key="code_b")
        
    if st.button("⚖️ Compare Now", type="primary", use_container_width=True):
        with st.spinner("Analyzing both versions..."):
            res_a = predict(code_a, region=selected_region, instance=selected_instance)
            res_b = predict(code_b, region=selected_region, instance=selected_instance)
            
            if "error" in res_a or "error" in res_b:
                st.error("Error parsing one of the code snippets. Check for syntax errors.")
            else:
                st.markdown("---")
                st.markdown("### Results")
                
                # Determine winner
                winner = "A" if res_a["energy"]["hybrid_joules"] < res_b["energy"]["hybrid_joules"] else "B"
                st.success(f"🏆 Version **{winner}** is greener!")
                
                # Comparison Table
                comp_data = {
                    "Metric": [
                        "Energy (Joules)", 
                        "Carbon Footprint (g)", 
                        "Cost per 1M runs", 
                        "Estimated Runtime (s)",
                        "Complexity Score",
                        "Green Score"
                    ],
                    "Version A": [
                        f"{res_a['energy']['hybrid_joules']:.6f}",
                        f"{res_a['carbon']['co2_grams']:.6f}",
                        f"${res_a['cost']['cost_per_million_runs']:.4f}",
                        f"{res_a['energy']['runtime_seconds']:.6f}",
                        f"{res_a['features']['computational_intensity']:.1f}",
                        res_a["green_score"]["grade"]
                    ],
                    "Version B": [
                        f"{res_b['energy']['hybrid_joules']:.6f}",
                        f"{res_b['carbon']['co2_grams']:.6f}",
                        f"${res_b['cost']['cost_per_million_runs']:.4f}",
                        f"{res_b['energy']['runtime_seconds']:.6f}",
                        f"{res_b['features']['computational_intensity']:.1f}",
                        res_b["green_score"]["grade"]
                    ]
                }
                
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
