import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from src.utils import generate_sample_data, get_yield_category, generate_parameter_alerts
from src.preprocessing import get_preprocessing_pipeline, prepare_data
from src.model_training import train_models, get_feature_importance
from src.evaluation import compare_models, evaluate_model
from agent.graph import build_graph
from utils.pdf_generator import create_pdf

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="HaveCrops Analytics | AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Premium Theme with Architecture animations and spacing adjustments
st.markdown(
    """
    <style>
        /* Base Animation Keyframes */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.7); }
            100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.4); }
        }
        @keyframes underlineGrow {
            from { width: 0; }
            to { width: 80px; }
        }

        .stApp, [data-testid="stAppViewContainer"], .main {
            background-color: #0B0F19 !important;
            color: #FFFFFF !important;
            font-family: 'Inter', system-ui, sans-serif !important;
        }

        h1, h2, h3, h4, h5 {
            color: #FFFFFF !important;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        div[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1E293B !important;
        }
        div[data-testid="stSidebar"] * { color: #FFFFFF !important; }

        /* Buttons: Premium Blue */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1px solid #1E293B !important;
            background: #111827 !important;
            color: #FFFFFF !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
            border-color: #3B82F6 !important;
        }
        button[kind="primary"] {
            background-color: #3B82F6 !important;
            color: #FFFFFF !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3) !important;
        }
        button[kind="primary"]:hover {
            background-color: #2563EB !important;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.6) !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 1rem; background: transparent; border-bottom: 2px solid #1E293B; padding-bottom: 5px; 
        }
        .stTabs [data-baseweb="tab"] { color: #9CA3AF !important; font-weight: 500; transition: color 0.2s ease; }
        .stTabs [data-baseweb="tab"]:hover { color: #FFFFFF !important; }
        .stTabs [aria-selected="true"] {
            color: #3B82F6 !important;
            border-bottom-color: #3B82F6 !important;
        }

        /* Cards and Elements with subtle lift */
        .metric-card {
            background: #111827;
            padding: 1.75rem;
            border-radius: 14px;
            text-align: center;
            border: 1px solid #1E293B;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            animation: fadeIn 0.4s ease-out;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
            border-color: #3B82F6;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF;
            margin-top: 0.75rem;
        }

        .insight-card {
            background: #111827;
            border: 1px solid #1E293B;
            border-left: 4px solid #22C55E;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            animation: fadeIn 0.5s ease-out;
            color: #E2E8F0;
        }

        .timeline-step {
            display: flex;
            align-items: flex-start;
            gap: 1.2rem;
            margin-bottom: 1rem;
            padding: 1.5rem;
            background: #111827;
            border-radius: 12px;
            border: 1px solid #1E293B;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.25s ease;
            animation: fadeIn 0.6s ease-out forwards;
        }
        .timeline-step:hover {
            transform: scale(1.01) translateX(4px);
            border-color: #3B82F6;
            background: #1E293B;
        }
        .timeline-icon {
            font-size: 1.5rem;
            min-width: 48px; height: 48px;
            display: flex; align-items: center; justify-content: center;
            background: rgba(59, 130, 246, 0.1);
            border-radius: 50%;
            border: 1px solid #3B82F6;
            color: #3B82F6;
        }

        .disclaimer-box {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid #F59E0B;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-top: 2rem;
            color: #FCD34D;
            animation: fadeIn 0.7s ease-out;
        }

        .final-footer {
            text-align: center;
            padding: 2.5rem 1rem;
            margin-top: 4rem;
            border-top: 1px solid #1E293B;
            color: #9CA3AF;
            font-size: 0.95rem;
            background: linear-gradient(180deg, transparent, #0B0F19);
        }
        
        /* Architecture Graph Flow CSS */
        .arch-node {
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            position: relative;
            z-index: 2;
            width: 260px;
            margin: 0 auto;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .arch-node:hover {
            transform: translateY(-3px) scale(1.02);
            border-color: #3B82F6;
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
        }
        .arch-node-active {
            border-color: #3B82F6;
            animation: pulseGlow 2s infinite;
        }
        .arch-title {
            color: #FFFFFF; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.25rem;
        }
        .arch-desc {
            color: #9CA3AF; font-size: 0.85rem; line-height: 1.4;
        }
        .arch-arrow {
            text-align: center; color: #3B82F6; font-size: 1.5rem; margin: 0.5rem 0;
            animation: fadeIn 1s ease-in;
        }
        .arch-split {
            display: flex; justify-content: center; gap: 2rem; margin: 1rem 0; relative;
        }
        
        /* Overview Grid */
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        .grid-card {
            background: #111827;
            border: 1px solid #1E293B;
            padding: 1.5rem;
            border-radius: 12px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            animation: fadeIn 0.5s ease-out forwards;
        }
        .grid-card:hover {
            transform: translateY(-5px);
            border-color: #22C55E;
            box-shadow: 0 8px 25px rgba(34, 197, 94, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_FONT = dict(family="system-ui, -apple-system, sans-serif", color="#FFFFFF", size=12)
PLOTLY_PAPER = "rgba(0,0,0,0)"
PLOTLY_PLOT = "rgba(0,0,0,0)"

def plotly_dark_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_PLOT,
        font=PLOTLY_FONT,
        margin=dict(l=30, r=30, t=50, b=30),
        hovermode="closest"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B', zerolinecolor='#334155')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E293B', zerolinecolor='#334155')
    return fig

def risk_meter_value(pred: float, max_yield: float) -> float:
    risk = 100 - (pred / max_yield * 100)
    return round(max(0.0, min(100.0, float(risk))), 1)

def confidence_from_model(pipeline, X_test, y_test) -> int:
    m = evaluate_model(pipeline, X_test, y_test)
    if not m or "R2 Score" not in m:
        return 85
    r2 = float(m["R2 Score"])
    return max(0, min(100, int(round(r2 * 100))))

@st.cache_data
def load_and_prep_data():
    data_path = "data/sample_farm_data.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        generate_sample_data(data_path)
    return pd.read_csv(data_path)

@st.cache_resource
def get_trained_models(_df, num_feat, cat_feat):
    X_train, X_test, y_train, y_test = prepare_data(_df)
    preprocessor = get_preprocessing_pipeline(num_feat, cat_feat)
    models = train_models(preprocessor, X_train, y_train)
    return models, X_test, y_test

df = load_and_prep_data()
numeric_features = ["Rainfall", "Fertilizer_Used", "Soil_pH"]
categorical_features = ["Soil_Type", "Crop_Type"]
trained_models, X_test, y_test = get_trained_models(df, numeric_features, categorical_features)

# ==============================================================================
# MAIN SIDEBAR
# ==============================================================================
st.sidebar.markdown("<h3 style='color:#FFFFFF;margin-top:0;'>Parameters</h3>", unsafe_allow_html=True)

s_rain = st.sidebar.number_input("Average Rainfall (mm)", 200, 1200, 600)
s_ph = st.sidebar.slider("Soil pH Level", 4.0, 9.5, 6.5, 0.1)
s_fert = st.sidebar.number_input("Fertilizer Amount (kg/ha)", 0, 300, 120)
s_soil = st.sidebar.selectbox("Soil Type", df["Soil_Type"].unique())
s_crop = st.sidebar.selectbox("Crop Type", df["Crop_Type"].unique())
s_model = st.sidebar.selectbox("Select ML Model", list(trained_models.keys()))

input_df = pd.DataFrame(
    {
        "Rainfall": [s_rain],
        "Soil_Type": [s_soil],
        "Fertilizer_Used": [s_fert],
        "Soil_pH": [s_ph],
        "Crop_Type": [s_crop],
    }
)

pred = trained_models[s_model].predict(input_df)[0]
cat = get_yield_category(pred, df)
alerts = generate_parameter_alerts(s_rain, s_ph, s_fert)
conf_pct = confidence_from_model(trained_models[s_model], X_test, y_test)
report_date = datetime.now().strftime("%B %d, %Y")

tier_color = "#22C55E" if cat == "High" else "#F59E0B" if cat == "Medium" else "#EF4444"

# ==============================================================================
# TABS NAVIGATION
# ==============================================================================
tab_overview, tab_report, tab_predict, tab_analytics, tab_perf, tab_data, tab_arch = st.tabs(
    ["Overview", "AI Advisory Report", "Yield Predictor", "Analytics", "Model Evaluation", "Data Source", "Architecture"]
)

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3.5rem; margin-top: 2rem; animation: fadeIn 0.4s ease-out;">
        <h1 style="color: #FFFFFF !important; font-size: 3.2rem; margin-bottom: 0.5rem; letter-spacing: -0.02em;">HaveCrops Analytics</h1>
        <h3 style="color: #3B82F6 !important; font-weight: 500; font-size: 1.4rem;">Next-Gen AI Advisory System</h3>
        <div style="height: 4px; background: #3B82F6; margin: 1.5rem auto; border-radius: 2px; animation: underlineGrow 1s ease-out forwards;"></div>
        <p style="color: #9CA3AF; font-size: 1.15rem; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            A premier agronomy forecasting engine fusing foundational Machine Learning pipelines with cutting-edge LangGraph-Agentic automated advisories via RAG architectures.
        </p>
    </div>

    <h2 style="font-size: 1.8rem; margin-bottom: 0.5rem;">Core Features Dashboard</h2>
    <div class="overview-grid">
        <div class="grid-card" style="animation-delay: 0.1s;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
            <h4 style="margin:0 0 0.5rem 0; color:#FFFFFF;">ML Pipeline Prediction</h4>
            <p style="color:#9CA3AF; font-size:0.95rem; margin:0;">Utilizes Ridge, Lasso, and Tree-based models to intelligently forecast yield variance based on soil and weather parameters.</p>
        </div>
        <div class="grid-card" style="animation-delay: 0.2s;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">🧠</div>
            <h4 style="margin:0 0 0.5rem 0; color:#FFFFFF;">LLM Advisory Node</h4>
            <p style="color:#9CA3AF; font-size:0.95rem; margin:0;">Synthesizes user constraints via LangGraph workflows to output hyper-detailed programmatic mitigation plans.</p>
        </div>
        <div class="grid-card" style="animation-delay: 0.3s;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📈</div>
            <h4 style="margin:0 0 0.5rem 0; color:#FFFFFF;">Visual Analytics</h4>
            <p style="color:#9CA3AF; font-size:0.95rem; margin:0;">Cross-examine target yields securely securely against global historically validated distribution layers instantly.</p>
        </div>
        <div class="grid-card" style="animation-delay: 0.4s;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">⚖️</div>
            <h4 style="margin:0 0 0.5rem 0; color:#FFFFFF;">Risk Analysis Vector</h4>
            <p style="color:#9CA3AF; font-size:0.95rem; margin:0;">Analyzes inputs dynamically to tag risk metrics via isolated agronomic thresholds mathematically.</p>
        </div>
    </div>

    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 2.5rem; margin-bottom: 2.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); animation: fadeIn 0.6s ease-out;">
        <h3 style="margin-top: 0; color: #FFFFFF !important; margin-bottom: 1.5rem; font-size: 1.5rem;">Interaction Guide</h3>
        <ol style="margin-bottom: 0; color: #D1D5DB; font-size: 1.1rem; line-height: 1.9; padding-left: 1.2rem;">
            <li><b>Configure Inputs:</b> Modulate rainfall, soil types, and fertilizer parameters sequentially via the Sidebar.</li>
            <li><b>Select Model Topology:</b> Switch between Linear Regression, Decision Trees, or Ensembles effortlessly.</li>
            <li><b>Review Analytics:</b> Shift into the Yield Predictor or Analytics tabs to interact with plotted outputs.</li>
            <li><b>Generate Advisory:</b> Connect to the LLM agent via the AI Advisory tab to retrieve PDF exportable diagnostics.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: AI ADVISORY REPORT ---
with tab_report:
    st.markdown('<br/>', unsafe_allow_html=True)
    st.info("Formulate your specific advisory requirement constraint before generating the node execution.", icon="ℹ️")
    
    st.markdown("### 📋 Primary Diagnostic Summary", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3B82F6 !important;">{pred:.1f}</div><div class="metric-label">Yield (q/ha)</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{tier_color} !important;">{cat}</div><div class="metric-label">Threshold Risk</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{conf_pct}%</div><div class="metric-label">System Confidence</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem; padding-top: 0.4rem;">{s_crop}</div><div class="metric-label">Target Flora</div></div>', unsafe_allow_html=True)

    # Mini Visual Additions underneath summary
    c_mini1, c_mini2 = st.columns([1, 1])
    with c_mini1:
        fig_r = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_meter_value(pred, df["Yield"].max()),
            title = {'text': "Agronomic Risk Index", 'font': {'color': '#FFFFFF'}},
            number = {'suffix': "%", 'font': {'color': '#3B82F6'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "#334155"},
                'bar': {'color': tier_color},
                'bgcolor': "#1E293B",
                'borderwidth': 2,
                'bordercolor': "#334155",
            }
        ))
        fig_r.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"))
        st.plotly_chart(fig_r, use_container_width=True)

    with c_mini2:
        fig_radar = go.Figure(go.Scatterpolar(
          r=[s_rain/1200, s_ph/9.5, s_fert/300],
          theta=['Rainfall', 'Soil pH', 'Fertilizer'],
          fill='toself',
          fillcolor='rgba(59, 130, 246, 0.4)',
          line_color='#3B82F6'
        ))
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"), bgcolor="#0B0F19"),
          showlegend=False, height=280, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"), title="Input Intensity Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    advisory_query = st.text_input("Injection Context for RAG Agent", placeholder=f"e.g. Provide specific irrigation strategies for maintaining {s_crop} soil health.")
    final_query = advisory_query if advisory_query else f"Give precise numbered action steps to maximize yield explicitly for {s_crop} given the {s_soil} soil conditions."

    gen = st.button("Initialize LangGraph Agent & Generate Report", type="primary", use_container_width=True)

    if gen:
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Missing **GROQ_API_KEY**. LLM Generation physically disabled.")
        else:
            with st.spinner("Connecting to Serverless Agent Engine... synthesizing inputs."):
                try:
                    graph = build_graph()
                    res = graph.invoke(
                        {
                            "crop_data": {
                                "crop": s_crop,
                                "soil": s_soil,
                                "rainfall": s_rain,
                                "temperature": 25,
                                "ph": s_ph,
                                "fertilizer": s_fert,
                            },
                            "prediction": pred,
                            "risk_level": cat,
                            "query": final_query,
                        }
                    )

                    st.success("Target Action Plan generated.")

                    st.markdown("### 🚀 Algorithmic Action Plan", unsafe_allow_html=True)
                    lines = [ln for ln in res["report"].split("\\n") if len(ln.strip()) > 10 and not ln.strip().startswith("#")]
                    labels = ["System Initialization", "Resource Modulations", "Geochemical Corrections", "Tracking Metrics"]
                    
                    for i, step in enumerate(lines[:4]):
                        icon = "⚙️" if i==0 else "💧" if i==1 else "🧪" if i==2 else "📊"
                        lab = labels[i] if i < len(labels) else f"Step {i+1}"
                        st.markdown(
                            f'''
                            <div class="timeline-step">
                                <div class="timeline-icon">{icon}</div>
                                <div>
                                    <div style="font-weight:600;color:#FFFFFF;font-size:1.15rem;margin:0 0 8px 0;">{lab}</div>
                                    <div style="margin:0;color:#D1D5DB;font-size:0.98rem;line-height:1.6;">{step}</div>
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True,
                        )

                    st.markdown("### 💡 Logic Insights", unsafe_allow_html=True)
                    st.markdown(
                        f'''
                        <div class="insight-card">
                            <strong style="font-size:1.1rem;color:#FFFFFF;">Diagnostic Takeaway</strong><br/><br/>
                            <span style="line-height:1.6; color:#D1D5DB;">Current operational models verify systemic integrity matching {conf_pct}% confidence towards standard historical thresholds. Prioritize localized physical soil verification alongside these LLM-generated operational suggestions for absolute yield security surrounding {s_crop}.</span>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

                    with st.expander("Explore Raw Agent Output / Export"):
                        st.markdown(f"<div style='background:#111827;padding:1.5rem;border-radius:8px;border:1px solid #1E293B;font-family:monospace;white-space:pre-wrap;color:#A7F3D0;'>{res['report']}</div>", unsafe_allow_html=True)

                    pdf_path = create_pdf(res["report"], s_crop, datetime.now().strftime("%B %d, %Y at %H:%M"))
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button("Export as PDF File", pdf_file, file_name=f"Analytics_Report_{s_crop.replace(' ', '_')}.pdf", type="primary")
                except Exception as e:
                    st.error(f"Inference failure: {str(e)}")

# --- TAB 3: PREDICTOR ---
with tab_predict:
    st.markdown("### Yield Input Details", unsafe_allow_html=True)
    st.info(f"Model Architecture Selected: **{s_model}**")
    for alert in alerts:
        st.warning(alert)
    with st.expander("JSON Serialized Request Payload", expanded=True):
        st.json({"Rainfall": s_rain, "pH": s_ph, "Fertilizer": s_fert, "Soil": s_soil, "Crop": s_crop, "Model Threshold Flag": cat})

# --- TAB 4: ANALYTICS ---
with tab_analytics:
    st.markdown("### Global Parameter Heatmaps & Graphs", unsafe_allow_html=True)
    st.markdown("<p style='color: #9CA3AF; margin-bottom: 2rem;'>Interactive visualizations scaling current constraints against historical matrix records.</p>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Chart 1: Yield vs Rainfall
        fig_rain = px.scatter(df, x="Rainfall", y="Yield", color="Crop_Type", 
                              title="Rainfall Correlation Matrix", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rain.add_trace(go.Scatter(x=[s_rain], y=[pred], mode="markers", 
                                      marker=dict(size=18, color="#FFFFFF", symbol="star", line=dict(color='#EF4444', width=2)), name="Active Pred"))
        fig_rain = plotly_dark_layout(fig_rain)
        st.plotly_chart(fig_rain, use_container_width=True)

        # Chart 3: Model Comparison
        model_names = list(trained_models.keys())
        model_preds = [trained_models[m].predict(input_df)[0] for m in model_names]
        colors = ["#3B82F6" if m == s_model else "#1E3A8A" for m in model_names]
        
        fig_comp = go.Figure(data=[go.Bar(x=model_names, y=model_preds, marker_color=colors, text=[f"{p:.1f}" for p in model_preds], textposition='auto')])
        fig_comp.update_layout(title="Prediction Latency across Deployments")
        fig_comp = plotly_dark_layout(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

    with chart_col2:
        # Chart 2: Soil pH vs Yield
        fig_ph = px.scatter(df, x="Soil_pH", y="Yield", color="Crop_Type", 
                            title="Geochemical (pH) vs Yield", color_discrete_sequence=px.colors.qualitative.Set3)
        fig_ph.add_trace(go.Scatter(x=[s_ph], y=[pred], mode="markers", 
                                    marker=dict(size=18, color="#FFFFFF", symbol="star", line=dict(color='#EF4444', width=2)), name="Active Pred"))
        fig_ph = plotly_dark_layout(fig_ph)
        st.plotly_chart(fig_ph, use_container_width=True)

        # Chart 4: Yield Distribution Histogram
        fig_hist = px.histogram(df, x="Yield", nbins=30, title="Yield Record Frequencies Database",
                                marginal="box", color_discrete_sequence=["#22C55E"])
        fig_hist.add_vline(x=pred, line_dash="dash", line_color="#3B82F6", annotation_text="Active Pred")
        fig_hist = plotly_dark_layout(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 5: EVALUATION ---
with tab_perf:
    st.markdown("### Performance Evaluation Grids", unsafe_allow_html=True)
    st.dataframe(compare_models(trained_models, X_test, y_test), use_container_width=True)

# --- TAB 6: DATA ---
with tab_data:
    st.markdown("### Active Source Dataset", unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Database Vectors", f"{len(df)}")
    m_col2.metric("Mean Global Yield", f"{df['Yield'].mean():.2f} q/ha")
    st.dataframe(df.head(100), use_container_width=True)

# --- TAB 7: ARCHITECTURE ---
with tab_arch:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; margin-top: 1rem; animation: fadeIn 0.5s ease-out;">
        <h1 style="color: #FFFFFF !important; font-size: 2.6rem;">System Architecture</h1>
        <h3 style="color: #3B82F6 !important; font-weight: 500;">Milestone 2: ML + RAG + LLM Pipeline Integration</h3>
    </div>

    <!-- MAIN INFERENCE PIPELINE -->
    <h3 style="text-align: center; margin-top: 2rem; color: #E2E8F0;">Layer 1: Inference & Generation Flow</h3>
    <div style="padding: 2rem 0;">
        <div class="arch-node arch-node-active">
            <div class="arch-title">1. User Parameter Injection</div>
            <div class="arch-desc">Real-time payload (Rainfall, pH, Soil Type, Constraints)</div>
        </div>
        <div class="arch-arrow">↓</div>
        
        <div class="arch-node">
            <div class="arch-title">2. Matrix Preprocessing</div>
            <div class="arch-desc">Numeric Scaling & Categorical OHE Encodings</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            <div class="arch-title">3. Predictive ML Sublayer</div>
            <div class="arch-desc">Trained Regression Trees map exact numeric Yield forecast</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-split" style="position:relative;">
            <div class="arch-node" style="width:220px; border-color: #EF4444;">
                <div class="arch-title" style="color:#EF4444;">High Risk Branch</div>
                <div class="arch-desc">Triggers deep analysis</div>
            </div>
            <div class="arch-node" style="width:220px; border-color: #22C55E;">
                <div class="arch-title" style="color:#22C55E;">Low Risk Branch</div>
                <div class="arch-desc">Standard output route</div>
            </div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-node arch-node-active">
            <div class="arch-title">4. Agentic RAG Filter (FAISS)</div>
            <div class="arch-desc">Contextual matching against agricultural documentation chunks</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            <div class="arch-title">5. LangGraph LLM Engine</div>
            <div class="arch-desc">Synthesizes data constraints and generates precise textual solutions</div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-node" style="border-color: #10B981;">
            <div class="arch-title" style="color: #10B981;">6. Formatted Output Block</div>
            <div class="arch-desc">Dashboard Rendering & Automated PDF Export</div>
        </div>
    </div>
    
    <hr style="border-color: #1E293B; margin: 3rem 0;"/>

    <!-- TRAINING PIPELINE -->
    <h3 style="text-align: center; margin-top: 1rem; color: #E2E8F0;">Layer 2: Model Training Topology</h3>
    <div style="padding: 2rem 0; display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:10px;">
        <div class="arch-node" style="width:160px;"><div class="arch-title">Raw CSV</div></div>
        <div style="color:#3B82F6;">→</div>
        <div class="arch-node" style="width:160px;"><div class="arch-title">Cleaning</div></div>
        <div style="color:#3B82F6;">→</div>
        <div class="arch-node" style="width:160px;"><div class="arch-title">Feat. Eng</div></div>
        <div style="color:#3B82F6;">→</div>
        <div class="arch-node arch-node-active" style="width:160px;"><div class="arch-title">Model Fits</div></div>
        <div style="color:#3B82F6;">→</div>
        <div class="arch-node" style="width:160px; border-color:#22C55E;"><div class="arch-title" style="color:#22C55E;">Deploy</div></div>
    </div>
    """, unsafe_allow_html=True)


# --- FOOTER ---
st.markdown(
    """
    <div class="final-footer">
        <strong style="color: #FFFFFF; font-size: 1.05rem;">HaveCrops Analytics</strong><br/>
        <span style="color: #9CA3AF; display:inline-block; margin-top: 6px;">End-Sem Project Submission</span><br/>
        <span style="color: #3B82F6; font-weight: 500;">Vedant Satbhai</span><br/>
    </div>
    """,
    unsafe_allow_html=True,
)
