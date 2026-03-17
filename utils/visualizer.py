"""
GreenCode Visualizer Utilities
==============================
Helper functions for generating Plotly charts and rendering HTML badges
in the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px

# ─── Chart Colors ─────────────────────────────────────────────────────────────

COLORS = {
    "CPU": "#3dffa0",      # var(--green)
    "Memory": "#1a7a4a",   # var(--green-dim)
    "I/O": "#f5c842",      # var(--amber)
    "Network": "#6b8870",  # var(--muted)
    "Cost": "#ff5e5e",     # var(--red)
    "Carbon": "#e8f0ea",   # var(--text)
}

# ─── Plotly Charts ────────────────────────────────────────────────────────────

def plot_energy_breakdown(breakdown_pct: dict) -> go.Figure:
    """Create a bar chart showing power breakdown by component."""
    labels = list(breakdown_pct.keys())
    values = list(breakdown_pct.values())
    colors = [COLORS.get(label, "#999999") for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels, 
            y=values,
            marker_color=colors,
            text=[f"{v}%" for v in values],
            textposition='auto',
            textfont=dict(color="#090e0a", family="DM Mono", weight="bold"),
        )
    ])
    
    fig.update_layout(
        title=dict(text="Hardware Power Breakdown", font=dict(color="#e8f0ea", family="Syne")),
        yaxis=dict(title="Percentage (%)", color="#6b8870"),
        xaxis=dict(color="#6b8870"),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    
    return fig


def plot_carbon_pie(breakdown_pct: dict, total_carbon_g: float) -> go.Figure:
    """Create a pie chart showing carbon emissions by component."""
    labels = list(breakdown_pct.keys())
    values = [(v / 100) * total_carbon_g for v in breakdown_pct.values()]
    colors = [COLORS.get(label, "#999999") for label in labels]

    # Guard: if all values are effectively zero, return a fallback figure
    if sum(values) < 1e-12:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Carbon Contribution", font=dict(color="#e8f0ea", family="Syne")),
            annotations=[dict(
                text="No measurable carbon<br>emissions for this snippet",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#6b8870", family="DM Mono", size=14),
            )],
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        return fig

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=colors, line=dict(color="#090e0a", width=2)),
            textinfo='label+percent',
            textfont=dict(color="#e8f0ea", family="DM Mono"),
        )
    ])

    fig.update_layout(
        title=dict(text="Carbon Contribution", font=dict(color="#e8f0ea", family="Syne")),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        showlegend=False
    )

    return fig


def plot_cost_projection(cost_per_run: float) -> go.Figure:
    """Create a line chart showing cloud cost projecting over runs."""
    runs = [1, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    costs = [r * cost_per_run for r in runs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=runs, 
        y=costs,
        mode='lines+markers',
        name='Cost ($)',
        line=dict(color=COLORS["Cost"], width=3),
        marker=dict(size=8, color=COLORS["Cost"], line=dict(color="#090e0a", width=1))
    ))
    
    fig.update_layout(
        title=dict(text="Cloud Cost Projection", font=dict(color="#e8f0ea", family="Syne")),
        xaxis=dict(title="Number of Executions", type="log", color="#6b8870"),
        yaxis=dict(title="Estimated Cost (USD)", color="#6b8870"),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
    )
    
    return fig


# ─── HTML/CSS Components ──────────────────────────────────────────────────────

def render_green_score_badge(score_data: dict) -> str:
    """Render the Green Score as an HTML badge."""
    grade = score_data["grade"]
    label = score_data["label"]
    color = score_data["color"]
    
    # Map colors to new theme if needed:
    if "A" in grade: 
        color = "#3dffa0"
    elif "B" in grade:
        color = "#f5c842"
    elif "C" in grade:
        color = "#ffbd2e"
    else:
        color = "#ff5e5e"

    return f"""
    <div style="border: 1px solid #1e2e20; border-radius: 4px; padding: 1.5rem; background: #111a13; display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; position: relative; overflow: hidden;">
        <div style="position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: {color};"></div>
        <div style="padding-left: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: {color}; animation: blink 1.5s ease infinite;"></div>
                <h4 style="margin: 0; color: #e8f0ea; font-family: 'Syne', sans-serif; font-size: 1.2rem; letter-spacing: 0.05em; text-transform: uppercase;">Green Score</h4>
            </div>
            <div style="color: #6b8870; font-family: 'DM Mono', monospace; font-size: 0.8rem;">Energy Efficiency Rating</div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-family: 'DM Mono', monospace; font-weight: 600; font-size: 1.1rem; color: {color}; text-transform: uppercase; letter-spacing: 0.1em;">{label}</div>
            <div style="background: {color}; color: #090e0a; width: 56px; height: 56px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 800; font-family: 'Syne', sans-serif; clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);">
                {grade}
            </div>
        </div>
        <style>@keyframes blink {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.2;}} }}</style>
    </div>
    """
