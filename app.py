"""
GreenCode Streamlit Dashboard
==============================
Interactive UI for predicting energy, carbon, and cost of Python code.
"""

import streamlit as st
import pandas as pd
import json
import zipfile
import io
import os
import time
import threading

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

        input_mode = st.radio(
            "Input method:",
            ["✏️ Text", "📄 File", "📦 Folder (ZIP)"],
            horizontal=True,
            key="input_mode"
        )

        code_input = ""
        upload_warning = ""

        if input_mode == "✏️ Text":
            example = st.selectbox("Load an example template:", ["— Custom Code —"] + list(EXAMPLES.keys()))
            default_code = EXAMPLES[example] if example != "— Custom Code —" else EXAMPLES["Simple Loop"]
            code_input = st.text_area(
                "Paste Python code here:",
                value=default_code,
                height=360,
                key="code_input_text"
            )

        elif input_mode == "📄 File":
            uploaded_file = st.file_uploader(
                "Upload any source file:",
                key="code_file"
            )
            if uploaded_file is not None:
                raw = uploaded_file.read()
                for enc in ("utf-8", "latin-1"):
                    try:
                        code_input = raw.decode(enc)
                        break
                    except (UnicodeDecodeError, ValueError):
                        code_input = ""
                # Null bytes mean it decoded as latin-1 but is actually binary
                if code_input and '\x00' in code_input:
                    code_input = ""
                if code_input:
                    st.success(f"\u2705 Loaded: `{uploaded_file.name}` ({len(code_input.splitlines())} lines)")
                    st.code(code_input[:500] + ("..." if len(code_input) > 500 else ""), language="python")
                else:
                    upload_warning = f"\u26a0\ufe0f `{uploaded_file.name}` appears to be a binary file and cannot be analysed as source code."
            else:
                st.info("Upload any source file — `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.cpp`, `.html`, `.css`, and more.")

        elif input_mode == "📦 Folder (ZIP)":
            uploaded_zip = st.file_uploader(
                "Upload a project ZIP archive:",
                key="code_zip"
            )
            if uploaded_zip is not None:
                try:
                    zip_bytes = uploaded_zip.read()
                    # Validate it's actually a ZIP
                    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
                        upload_warning = "⚠️ The uploaded file is not a valid ZIP archive."
                    else:
                        # Always-skip patterns: compiled/binary file extensions
                        BINARY_EXTS = {
                            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp", ".tiff",
                            ".woff", ".woff2", ".ttf", ".otf", ".eot",
                            ".mp4", ".mp3", ".wav", ".ogg", ".webm",
                            ".exe", ".dll", ".so", ".dylib", ".bin", ".obj",
                            ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
                            ".pdf", ".doc", ".docx", ".xls", ".xlsx",
                            ".pyc", ".pyo", ".class",
                            ".DS_Store", ".map",
                        }
                        SKIP_DIRS = ("__MACOSX", "node_modules/", ".git/", ".venv/", "venv/", ".tox/")

                        parts = []
                        included = []
                        skipped = []
                        stage_times = {}

                        def _trunc(name, n=30):
                            return name if len(name) <= n else name[:n-3] + "..."

                        # ── Stage 1: Extract files from ZIP ──────────────────
                        t_extract_start = time.perf_counter()
                        with st.status("🗂️ Extracting files from ZIP...", expanded=True) as s1:
                            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                                all_names = sorted(
                                    n for n in zf.namelist()
                                    if not n.endswith("/")
                                    and not any(n.startswith(d) for d in SKIP_DIRS)
                                )
                                total_files = len(all_names)
                                counter = st.empty()

                                for idx, fname in enumerate(all_names, 1):
                                    counter.markdown(f"📄 Extracting **{idx}** / **{total_files}** — `{_trunc(fname)}`")
                                    _, ext = os.path.splitext(fname.lower())
                                    if ext in BINARY_EXTS:
                                        skipped.append((fname, "binary extension"))
                                        continue
                                    try:
                                        raw = zf.read(fname)
                                        for enc in ("utf-8", "latin-1"):
                                            try:
                                                content = raw.decode(enc)
                                                break
                                            except (UnicodeDecodeError, ValueError):
                                                content = None
                                        if content is None or '\x00' in content:
                                            skipped.append((fname, "binary content"))
                                            continue
                                        parts.append(f"# --- {fname} ---\n{content}")
                                        included.append(fname)
                                    except Exception:
                                        skipped.append((fname, "read error"))

                                counter.empty()

                            stage_times["extraction"] = time.perf_counter() - t_extract_start
                            if not parts:
                                s1.update(label="⚠️ No readable files found", state="error")
                                upload_warning = "⚠️ No readable text files found inside the ZIP archive."
                            else:
                                s1.update(
                                    label=f"✅ Extracted {total_files} files ({stage_times['extraction']:.3f}s)",
                                    state="complete", expanded=False
                                )

                        if parts:
                            # ── Stage 2: Analyse included files ──────────────
                            t_analysis_start = time.perf_counter()
                            with st.status("🔍 Analysing files...", expanded=True) as s2:
                                counter2 = st.empty()
                                for idx, fname in enumerate(included, 1):
                                    counter2.markdown(f"⚙️ Analysing file **{idx}** / **{len(included)}** — `{_trunc(os.path.basename(fname))}`")
                                counter2.empty()
                                stage_times["analysis"] = time.perf_counter() - t_analysis_start
                                s2.update(
                                    label=f"✅ Successfully included {len(included)} / {total_files} files ({stage_times['analysis']:.3f}s)",
                                    state="complete", expanded=False
                                )

                            # ── Stage 3: Catalogue skipped files ─────────────
                            if skipped:
                                t_skip_start = time.perf_counter()
                                with st.status("🗃️ Cataloguing skipped files...", expanded=True) as s3:
                                    counter3 = st.empty()
                                    for idx, (fname, reason) in enumerate(skipped, 1):
                                        counter3.markdown(f"⏭️ Cataloguing skipped file **{idx}** / **{len(skipped)}** — `{_trunc(os.path.basename(fname))}`")
                                    counter3.empty()
                                    stage_times["skipped"] = time.perf_counter() - t_skip_start
                                    s3.update(
                                        label=f"⚠️ {len(skipped)} files were skipped (binary or unreadable)",
                                        state="complete", expanded=False
                                    )
                                with st.expander(f"⏭️ Show skipped files ({len(skipped)})"):
                                    for fname, reason in skipped:
                                        st.markdown(f"• `{os.path.basename(fname)}` — *{reason}*")

                            code_input = "\n\n".join(parts)
                            st.session_state['_zip_stage_times'] = stage_times

                            with st.expander(f"📂 Files included in analysis ({len(included)})"):
                                for f in included:
                                    st.markdown(f"- `{f}`")
                            st.code(code_input[:600] + ("..." if len(code_input) > 600 else ""), language="python")

                except Exception as e:
                    upload_warning = f"⚠️ Could not read ZIP archive: {e}"
            else:
                st.info("Upload a `.zip` of your project. All readable text files (`.py`, `.js`, `.ts`, `.html`, `.css`, `.md`, etc.) will be extracted and analysed. Binary files are skipped automatically.")

        if upload_warning:
            st.warning(upload_warning)

        analyze_btn = st.button(
            "🚀 Analyze Code",
            type="primary",
            width="stretch",
            disabled=(not code_input.strip())
        )
        
    with col2:
        st.markdown("### Analysis Results")

        if analyze_btn or 'last_result' in st.session_state:
            if analyze_btn:
                # Grab pre-existing extraction stage times (from ZIP mode)
                zip_times = st.session_state.pop('_zip_stage_times', {})
                timing = {}
                timing['extraction'] = zip_times.get('extraction', 0.0)
                timing['analysis'] = zip_times.get('analysis', 0.0)

                # ── Stage 4: ML Prediction with live timer ───────────
                result_container = {}

                def _run_prediction():
                    result_container['result'] = predict_and_compare(
                        code_input, regions=[selected_region], instance=selected_instance
                    )

                t_pred_start = time.perf_counter()
                pred_thread = threading.Thread(target=_run_prediction)
                pred_thread.start()

                timer_display = st.empty()
                while pred_thread.is_alive():
                    elapsed = time.perf_counter() - t_pred_start
                    timer_display.markdown(f"⏱️ **Predicting...** `{elapsed:.1f}s`")
                    time.sleep(0.1)
                pred_thread.join()
                timing['inference'] = time.perf_counter() - t_pred_start
                timing['total'] = timing['extraction'] + timing['analysis'] + timing['inference']
                timer_display.empty()

                result = result_container['result']
                result['_timing'] = timing
                st.session_state.last_result = result
            else:
                result = st.session_state.last_result

            if "error" in result:
                st.error(f"Failed to parse code: {result['error']}")
            else:
                # Show fallback analyser notice if applicable
                if result.get("features", {}).get("_fallback_used"):
                    st.warning("⚠️ Some code could not be parsed as Python AST (e.g. non-Python files, emojis, or encoding issues). "
                               "A fallback regex-based analyser was used — results are approximate.")

                # ── Timing breakdown badge ───────────────────────────
                if '_timing' in result:
                    t = result['_timing']
                    st.markdown(f"""
                    <div style="border:1px solid #1e2e20; border-radius:4px; padding:10px 14px; background:#111a13; margin-bottom:16px; font-family:'DM Mono',monospace; font-size:0.78rem; color:#6b8870;">
                        <span style="color:#3dffa0; font-weight:bold;">⚡ Prediction completed in {t['total']:.3f}s</span><br>
                        <span style="opacity:0.7;">📂 Extraction: &nbsp;&nbsp;{t['extraction']:.3f}s</span><br>
                        <span style="opacity:0.7;">🔍 Analysis: &nbsp;&nbsp;&nbsp;&nbsp;{t['analysis']:.3f}s</span><br>
                        <span style="opacity:0.7;">🤖 ML Inference: {t['inference']:.3f}s</span>
                    </div>
                    """, unsafe_allow_html=True)
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
                        st.plotly_chart(
                            plot_energy_breakdown(result["component_breakdown_pct"]),
                            width="stretch"
                        )
                    with c2:
                        carbon_g = result["carbon"]["co2_grams"]
                        if carbon_g is not None and carbon_g > 0:
                            st.plotly_chart(
                                plot_carbon_pie(result["component_breakdown_pct"], carbon_g),
                                width="stretch"
                            )
                        else:
                            st.info("☁️ No measurable carbon emissions for this snippet in the selected region.")

                    st.plotly_chart(
                        plot_cost_projection(result["cost"]["estimated_cost_usd"]),
                        width="stretch"
                    )
                
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
                    st.dataframe(pd.DataFrame(reg_data), width="stretch", hide_index=True)
                    
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
        
    if st.button("⚖️ Compare Now", type="primary", width="stretch"):
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
                
                st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)
