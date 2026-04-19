import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# --- Project presentation (edit for your submission) ---
PROJECT_AUTHOR = "Vedant Satbhai"
ACADEMIC_YEAR = "2025–26"

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="HaveCrops Analytics | AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light, premium SaaS theme + motion (primary UI is LIGHT — not dark)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --primary: #00C853;
            --secondary: #2979FF;
            --accent: #7C4DFF;
            --bg: #F8F9FA;
            --surface: #FFFFFF;
            --border: #E0E0E0;
            --text: #1A1A1A;
            --muted: #5C6370;
            --shadow: rgba(15, 23, 42, 0.06);
            --shadow-hover: rgba(15, 23, 42, 0.12);
        }

        .stApp, [data-testid="stAppViewContainer"], .main {
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }
        section.main > div { padding-top: 1.25rem; }

        * { font-family: 'Inter', sans-serif !important; }
        h1, h2, h3, h4, h5, .hero-title, .space-font {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text) !important;
        }
        p, span, label, .stMarkdown { color: var(--muted); }

        div[data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        div[data-testid="stSidebar"] * { color: var(--text) !important; }

        /* Page enter */
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-up {
            from { opacity: 0; transform: translateY(24px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes float-soft {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        @keyframes blink-caret {
            from, to { border-color: transparent; }
            50% { border-color: var(--primary); }
        }
        @keyframes underline-grow {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes ripple {
            to { transform: scale(4); opacity: 0; }
        }

        .page-fade { animation: fade-in 0.7s ease-out both; }
        .section-up { animation: slide-up 0.65s ease-out both; }
        .delay-1 { animation-delay: 0.08s; }
        .delay-2 { animation-delay: 0.16s; }
        .delay-3 { animation-delay: 0.24s; }

        /* Hero — soft light gradient */
        .hero-soft {
            background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 45%, #F3E5F5 100%);
            padding: 4.5rem 2rem 4rem;
            border-radius: 20px;
            text-align: center;
            border: 1px solid var(--border);
            box-shadow: 0 12px 40px var(--shadow);
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
        }
        .hero-soft::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 20% 20%, rgba(0,200,83,0.08), transparent 45%),
                        radial-gradient(circle at 80% 80%, rgba(124,77,255,0.07), transparent 40%);
            pointer-events: none;
        }
        .hero-title {
            font-size: clamp(2.2rem, 4vw, 3.25rem);
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: var(--text) !important;
            letter-spacing: -0.03em;
            position: relative;
            z-index: 1;
        }
        .typewriter-wrap {
            display: inline-block;
            max-width: 100%;
            position: relative;
            z-index: 1;
        }
        .typewriter-wrap h3 {
            display: inline-block;
            overflow: hidden;
            white-space: nowrap;
            border-right: 3px solid var(--primary);
            margin: 0 auto;
            font-size: clamp(1rem, 2vw, 1.35rem);
            font-weight: 500;
            color: var(--muted) !important;
            max-width: 0;
            animation: typing-line 3.4s steps(52, end) 0.2s forwards, blink-caret 0.75s step-end infinite;
        }
        @keyframes typing-line {
            from { max-width: 0; }
            to { max-width: min(100%, 52rem); }
        }

        /* Feature cards */
        .feature-card {
            background: var(--surface);
            padding: 1.75rem 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px var(--shadow);
            transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.35s ease;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 16px 36px var(--shadow-hover);
            border-color: rgba(41, 121, 255, 0.35);
        }
        .feature-icon { font-size: 2.25rem; margin-bottom: 0.75rem; animation: float-soft 5s ease-in-out infinite; }
        .feature-card h4 { color: var(--text) !important; margin: 0 0 0.5rem 0; font-size: 1.1rem; }

        /* Report */
        .report-hero {
            background: var(--surface);
            padding: 1.75rem 2rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 28px var(--shadow);
            margin-bottom: 1.5rem;
        }
        .report-hero h1 { margin: 0; font-size: 1.75rem; color: var(--text) !important; }
        .report-hero .sub { margin: 0.35rem 0 0 0; color: var(--muted) !important; font-size: 0.95rem; }
        .title-line {
            height: 4px;
            width: 120px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
            border-radius: 4px;
            margin-top: 1rem;
            transform-origin: left;
            animation: underline-grow 0.9s ease-out 0.2s both;
        }

        .metric-card {
            background: var(--surface);
            padding: 1.25rem 1rem;
            border-radius: 14px;
            text-align: center;
            border: 1px solid var(--border);
            box-shadow: 0 4px 16px var(--shadow);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 28px var(--shadow-hover); }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text) !important;
            line-height: 1.2;
        }
        .metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important; margin-top: 0.35rem; }

        .insight-card {
            background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E9 100%);
            border: 1px solid #C8E6C9;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            color: var(--text) !important;
        }
        .disclaimer-box {
            background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
            border: 1px solid #FDBA74;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-top: 1rem;
            color: #9A3412 !important;
        }

        /* Timeline */
        .timeline-step {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
            padding: 1.1rem 1.25rem;
            background: var(--surface);
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 14px var(--shadow);
            animation: slide-up 0.5s ease backwards;
        }
        .timeline-icon {
            font-size: 1.35rem;
            min-width: 44px; height: 44px;
            display: flex; align-items: center; justify-content: center;
            background: #E8F5E9;
            border-radius: 50%;
            border: 1px solid #C8E6C9;
        }

        /* Pipeline — light */
        .pipeline-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1.5rem 0;
            padding: 1.25rem 1rem;
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px var(--shadow);
        }
        .pipeline-node { text-align: center; flex: 1; min-width: 72px; position: relative; z-index: 2; }
        .pipeline-node .lbl { font-size: 0.78rem; color: var(--muted) !important; margin-top: 0.35rem; }
        .pipeline-circle {
            width: 52px; height: 52px;
            border-radius: 50%;
            background: #F1F5F9;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.35rem;
            margin: 0 auto;
            border: 2px solid var(--border);
            transition: all 0.45s ease;
        }
        .pipeline-node.active .pipeline-circle {
            border-color: var(--primary);
            background: rgba(0, 200, 83, 0.12);
            box-shadow: 0 0 0 4px rgba(0, 200, 83, 0.15);
        }
        .pipeline-line {
            flex: 1;
            height: 3px;
            background: #E2E8F0;
            min-width: 20px;
            align-self: center;
            margin-bottom: 1.5rem;
            border-radius: 2px;
            transition: background 0.45s ease;
        }
        .pipeline-line.active { background: linear-gradient(90deg, var(--primary), var(--secondary)); }

        /* Buttons: ripple + glow */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            position: relative;
            overflow: hidden;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(41, 121, 255, 0.2) !important;
        }
        div[data-testid="stTabs"] { margin-top: 0.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 1rem; background: transparent; border-bottom: 1px solid var(--border); }
        .stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom-color: var(--primary) !important;
        }

        /* Shimmer placeholder */
        .shimmer-bar {
            height: 10px;
            border-radius: 6px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e8e8e8 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.2s linear infinite;
        }

        .final-footer {
            text-align: center;
            padding: 2.5rem 1rem 2rem;
            margin-top: 3rem;
            border-top: 1px solid var(--border);
            color: var(--muted) !important;
            font-size: 0.9rem;
            background: linear-gradient(180deg, transparent, rgba(0,0,0,0.02));
        }
        .final-footer .foot-line {
            height: 3px;
            width: 100px;
            margin: 14px auto 0;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
            animation: fade-in 1s ease both;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_FONT = dict(family="Inter, sans-serif", color="#1A1A1A", size=12)
PLOTLY_PAPER = "#FFFFFF"
PLOTLY_PLOT = "#F8F9FA"


def plotly_light_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_PLOT,
        font=PLOTLY_FONT,
        margin=dict(l=24, r=24, t=48, b=24),
    )
    return fig


def risk_meter_value(yield_tier: str) -> float:
    """Map yield tier to a 0–100 'pressure' scale (lower yield tier → higher number)."""
    return {"Low": 78, "Medium": 46, "High": 18}.get(yield_tier, 45)


def confidence_from_model(pipeline, X_test, y_test) -> int:
    m = evaluate_model(pipeline, X_test, y_test)
    if not m or "R2 Score" not in m:
        return 85
    r2 = float(m["R2 Score"])
    return max(0, min(100, int(round(r2 * 100))))


# --- SESSION STATE ---
if "started" not in st.session_state:
    st.session_state.started = False
if "dashboard_hint" not in st.session_state:
    st.session_state.dashboard_hint = None


@st.cache_data
def load_and_prep_data():
    data_path = "data/sample_farm_data.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        generate_sample_data(data_path)
    df = pd.read_csv(data_path)
    return df


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
# HOME / LANDING
# ==============================================================================
if not st.session_state.started:
    st.markdown('<div class="page-fade">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-soft section-up">
            <h1 class="hero-title">HaveCrops Analytics</h1>
            <div class="typewriter-wrap">
                <h3>AI-Powered Crop Intelligence & Agentic Farming Advisor</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Start Analysis", type="primary", use_container_width=True, key="btn_start"):
            st.session_state.started = True
            st.session_state.dashboard_hint = "report"
            st.rerun()
    with c3:
        if st.button("Explore Dashboard", use_container_width=True, key="btn_explore"):
            st.session_state.started = True
            st.session_state.dashboard_hint = "explore"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    cards = [
        ("🌾", "Yield Prediction", "Classical ML regressors with preprocessing and validation."),
        ("🤖", "AI Advisor", "LangGraph agent with RAG and Groq (Llama 3.3)."),
        ("📊", "Data Insights", "Distributions, correlations, and scenario overlays."),
        ("📄", "Smart Reports", "Interactive advisory view and exportable PDF."),
    ]
    cols = [fc1, fc2, fc3, fc4]
    for i, col in enumerate(cols):
        icon, title, desc = cards[i]
        with col:
            st.markdown(
                f"""
                <div class="feature-card section-up delay-{i+1}">
                    <div class="feature-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p style="margin:0;font-size:0.9rem;color:#5C6370;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<br/><h2 class='space-font' style='text-align:center;margin-bottom:0.5rem;'>How it works</h2>"
        "<p style='text-align:center;color:#5C6370;'>Input → ML → RAG → LLM → Report</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="pipeline-container section-up">
            <div class="pipeline-node active"><div class="pipeline-circle">📥</div><div class="lbl">Input</div></div>
            <div class="pipeline-line active"></div>
            <div class="pipeline-node active"><div class="pipeline-circle">🧮</div><div class="lbl">ML</div></div>
            <div class="pipeline-line active"></div>
            <div class="pipeline-node active"><div class="pipeline-circle">📚</div><div class="lbl">RAG</div></div>
            <div class="pipeline-line active"></div>
            <div class="pipeline-node active"><div class="pipeline-circle">🧠</div><div class="lbl">LLM</div></div>
            <div class="pipeline-line active"></div>
            <div class="pipeline-node active"><div class="pipeline-circle">📄</div><div class="lbl">Report</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# DASHBOARD
# ==============================================================================
else:
    st.markdown('<div class="page-fade">', unsafe_allow_html=True)
    hint = st.session_state.pop("dashboard_hint", None)
    if hint == "explore":
        st.info("Welcome — use the tabs to explore **Yield Predictor**, **Analytics**, and **Model Evaluation**.")
    elif hint == "report":
        st.success("You’re in the dashboard — open **AI Advisory Report** to generate your agent report.")

    st.sidebar.markdown(
        f"<h3 style='color:#00C853;margin-top:0;' class='space-font'>Parameters</h3>",
        unsafe_allow_html=True,
    )

    s_rain = st.sidebar.number_input("Average Rainfall (mm)", 200, 1200, 600)
    s_ph = st.sidebar.slider("Soil pH Level", 4.0, 9.5, 6.5, 0.1)
    s_fert = st.sidebar.number_input("Fertilizer Amount (kg/ha)", 0, 300, 120)
    s_soil = st.sidebar.selectbox("Soil Type", df["Soil_Type"].unique())
    s_crop = st.sidebar.selectbox("Crop Type", df["Crop_Type"].unique())
    s_model = st.sidebar.selectbox("Select ML Model", list(trained_models.keys()))

    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.started = False
        st.rerun()

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
    risk_val = risk_meter_value(cat)

    tier_color = "#00C853" if cat == "High" else "#FFB300" if cat == "Medium" else "#D32F2F"

    tab_report, tab_predict, tab_analytics, tab_perf, tab_data = st.tabs(
        ["AI Advisory Report", "Yield Predictor", "Analytics", "Model Evaluation", "Data Source"]
    )

    # --- TAB 1: AI ADVISORY REPORT ---
    with tab_report:
        st.markdown(
            f"""
            <div class="report-hero section-up">
                <h1>AI Advisory Report</h1>
                <p class="sub"><b>{s_crop}</b> · {report_date}</p>
                <div class="title-line"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{pred:.1f}</div><div class="metric-label">Yield (q/ha)</div></div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value" style="color:{tier_color} !important;">{cat}</div><div class="metric-label">Risk level</div></div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{conf_pct}%</div><div class="metric-label">Confidence (R²)</div></div>""",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value" style="font-size:1.25rem;">{s_crop}</div><div class="metric-label">Crop type</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("#### Visual analytics", unsafe_allow_html=True)
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(pred),
                    number={"suffix": " q/ha", "font": {"size": 28}},
                    title={"text": "Yield gauge", "font": {"size": 14, "color": "#1A1A1A"}},
                    gauge={
                        "axis": {"range": [0, float(df["Yield"].max()) + 10], "tickcolor": "#94A3B8"},
                        "bar": {"color": "#00C853"},
                        "bgcolor": "#FFFFFF",
                        "steps": [
                            {"range": [0, float(df["Yield"].quantile(0.33))], "color": "#FFEBEE"},
                            {
                                "range": [
                                    float(df["Yield"].quantile(0.33)),
                                    float(df["Yield"].quantile(0.66)),
                                ],
                                "color": "#FFF8E1",
                            },
                            {
                                "range": [float(df["Yield"].quantile(0.66)), float(df["Yield"].max()) + 10],
                                "color": "#E8F5E9",
                            },
                        ],
                    },
                )
            )
            plotly_light_layout(fig_gauge, height=340)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with row1_c2:
            fig_risk = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_val,
                    number={"suffix": " / 100", "font": {"size": 26}},
                    title={"text": "Agronomic pressure (tier-based)", "font": {"size": 14, "color": "#1A1A1A"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#FF9800"},
                        "steps": [
                            {"range": [0, 35], "color": "#E8F5E9"},
                            {"range": [35, 65], "color": "#FFF8E1"},
                            {"range": [65, 100], "color": "#FFEBEE"},
                        ],
                    },
                )
            )
            plotly_light_layout(fig_risk, height=340)
            st.plotly_chart(fig_risk, use_container_width=True)

        row2_c1, row2_c2 = st.columns(2)
        with row2_c1:
            fi_df = None
            if "Decision Tree" in trained_models:
                fi_df = get_feature_importance(
                    trained_models["Decision Tree"], numeric_features, categorical_features
                )
            if fi_df is not None and len(fi_df) > 0:
                fig_fi = px.bar(
                    fi_df.head(12),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#E3F2FD", "#2979FF"],
                )
                fig_fi.update_yaxes(categoryorder="total ascending")
                plotly_light_layout(fig_fi, height=360)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.caption("Feature importance appears when the Decision Tree model is available.")

        with row2_c2:
            avg_rain = df["Rainfall"].mean()
            avg_ph = df["Soil_pH"].mean()
            avg_fert = df["Fertilizer_Used"].mean()
            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[s_rain / 1200, s_ph / 9.5, s_fert / 300],
                    theta=["Rainfall", "Soil pH", "Fertilizer"],
                    fill="toself",
                    name="Your input",
                    line_color="#2979FF",
                    fillcolor="rgba(41,121,255,0.25)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[avg_rain / 1200, avg_ph / 9.5, avg_fert / 300],
                    theta=["Rainfall", "Soil pH", "Fertilizer"],
                    fill="toself",
                    name="Dataset average",
                    line_color="#00C853",
                    fillcolor="rgba(0,200,83,0.2)",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E0E0E0")),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            plotly_light_layout(fig_radar, height=360)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Agentic advisory", unsafe_allow_html=True)
        advisory_query = st.text_input(
            "Optional focus for the advisor",
            placeholder="e.g. How can I improve soil pH sustainably?",
        )
        final_query = advisory_query if advisory_query else "Give precise numbered action steps to maximize yield."

        gen = st.button("Generate AI Advisory Report", type="primary", key="gen_report")

        if gen:
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Set **GROQ_API_KEY** in your environment or `.env` file.")
            else:
                progress_container = st.empty()
                progress_container.markdown(
                    """
                    <div class="pipeline-container">
                        <div class="pipeline-node active"><div class="pipeline-circle">📥</div><div class="lbl">Input</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">🧮</div><div class="lbl">ML</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">📚</div><div class="lbl">RAG</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">🧠</div><div class="lbl">LLM</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">📄</div><div class="lbl">Report</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.6)
                progress_container.markdown(
                    """
                    <div class="pipeline-container">
                        <div class="pipeline-node active"><div class="pipeline-circle">📥</div><div class="lbl">Input</div></div>
                        <div class="pipeline-line active"></div>
                        <div class="pipeline-node active"><div class="pipeline-circle">🧮</div><div class="lbl">ML</div></div>
                        <div class="pipeline-line active"></div>
                        <div class="pipeline-node active"><div class="pipeline-circle">📚</div><div class="lbl">RAG</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">🧠</div><div class="lbl">LLM</div></div>
                        <div class="pipeline-line"></div>
                        <div class="pipeline-node"><div class="pipeline-circle">📄</div><div class="lbl">Report</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

                    time.sleep(0.5)
                    progress_container.markdown(
                        """
                        <div class="pipeline-container">
                            <div class="pipeline-node active"><div class="pipeline-circle">📥</div><div class="lbl">Input</div></div>
                            <div class="pipeline-line active"></div>
                            <div class="pipeline-node active"><div class="pipeline-circle">🧮</div><div class="lbl">ML</div></div>
                            <div class="pipeline-line active"></div>
                            <div class="pipeline-node active"><div class="pipeline-circle">📚</div><div class="lbl">RAG</div></div>
                            <div class="pipeline-line active"></div>
                            <div class="pipeline-node active"><div class="pipeline-circle">🧠</div><div class="lbl">LLM</div></div>
                            <div class="pipeline-line active"></div>
                            <div class="pipeline-node active"><div class="pipeline-circle">📄</div><div class="lbl">Report</div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.toast("Report generated successfully.", icon="✅")

                    st.markdown("#### Action plan", unsafe_allow_html=True)
                    lines = [
                        ln
                        for ln in res["report"].split("\n")
                        if len(ln.strip()) > 10 and not ln.strip().startswith("#")
                    ]
                    labels = ["Soil & inputs", "Irrigation & water", "Nutrition & timing", "Monitoring"]
                    for i, step in enumerate(lines[:4]):
                        icon = "🧪" if "soil" in step.lower() or "ph" in step.lower() else "💧" if "water" in step.lower() or "irrig" in step.lower() else "🌱"
                        lab = labels[i] if i < len(labels) else f"Step {i+1}"
                        st.markdown(
                            f"""
                            <div class="timeline-step" style="animation-delay:{i*0.07}s">
                                <div class="timeline-icon">{icon}</div>
                                <div>
                                    <div style="font-weight:600;color:#1A1A1A;margin:0 0 4px 0;">{lab}</div>
                                    <div style="margin:0;color:#5C6370;font-size:0.92rem;">{step}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### AI insights", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div class="insight-card">
                            <strong>Key takeaway</strong><br/>
                            Recommendations combine your scenario, model context, and retrieved agronomy notes. Prioritize
                            measurable checks (soil tests, water balance) before large input changes.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Sources & full report text", expanded=False):
                        st.markdown(
                            f"<div style='background:#FAFAFA;padding:14px;border-radius:10px;border:1px solid #E0E0E0;'>{res['report']}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        """
                        <div class="disclaimer-box">
                            ⚠️ <strong>Disclaimer:</strong> Educational / preliminary guidance only. Confirm interventions with
                            local agronomists and certified testing where required.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    pdf_path = create_pdf(res["report"], s_crop, datetime.now().strftime("%B %d, %Y at %H:%M"))
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            "Download PDF report",
                            pdf_file,
                            file_name=f"HaveCrops_Report_{s_crop.replace(' ', '_')}.pdf",
                            type="primary",
                        )
                except Exception as e:
                    st.error(f"Agent execution failed: {str(e)}")

    # --- TAB 2: PREDICTOR ---
    with tab_predict:
        st.markdown("### Yield predictor", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-card' style='max-width:420px;'><div class='metric-value'>{pred:.2f}</div>"
            f"<div class='metric-label'>Predicted yield ({s_model})</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        for alert in alerts:
            st.warning(alert)
        with st.expander("Current scenario (JSON)", expanded=False):
            st.json(
                {
                    "Rainfall": s_rain,
                    "pH": s_ph,
                    "Fertilizer": s_fert,
                    "Soil": s_soil,
                    "Crop": s_crop,
                }
            )

    # --- TAB 3: ANALYTICS ---
    with tab_analytics:
        st.markdown("### Analytics", unsafe_allow_html=True)
        sns.set_theme(style="whitegrid", palette="pastel")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8F9FA")
            sns.scatterplot(data=df, x="Rainfall", y="Yield", hue="Crop_Type", alpha=0.35, ax=ax, legend="brief")
            ax.scatter(s_rain, pred, color="#D32F2F", s=180, marker="*", edgecolors="#FFFFFF", linewidths=1.5, zorder=5, label="Your point")
            ax.set_title("Rainfall vs yield", color="#1A1A1A")
            ax.legend(frameon=True, fancybox=True, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with v_col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8F9FA")
            sns.regplot(data=df, x="Soil_pH", y="Yield", scatter_kws={"alpha": 0.2}, line_kws={"color": "#00C853"}, ax=ax)
            ax.scatter(s_ph, pred, color="#D32F2F", s=180, marker="*", edgecolors="#FFFFFF", linewidths=1.5, zorder=5)
            ax.set_title("Soil pH vs yield", color="#1A1A1A")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- TAB 4: EVALUATION ---
    with tab_perf:
        st.markdown("### Model evaluation", unsafe_allow_html=True)
        perf_df = compare_models(trained_models, X_test, y_test)
        st.dataframe(perf_df, use_container_width=True)
        st.markdown("### Feature importance (Decision Tree)", unsafe_allow_html=True)
        if "Decision Tree" in trained_models:
            fi_df = get_feature_importance(
                trained_models["Decision Tree"], numeric_features, categorical_features
            )
            if fi_df is not None:
                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#E8EAF6", "#7C4DFF"],
                )
                fig_fi.update_yaxes(categoryorder="total ascending")
                plotly_light_layout(fig_fi, height=420)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance could not be extracted.")
        else:
            st.caption("Train Decision Tree to enable importance charts.")

    # --- TAB 5: DATA ---
    with tab_data:
        st.markdown("### Data source", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Records", f"{len(df)}")
        m_col2.metric("Mean yield", f"{df['Yield'].mean():.2f}", "q/ha")
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(
    f"""
    <div class="final-footer page-fade">
        <div class="space-font" style="font-size:1rem;color:#1A1A1A !important;margin-bottom:6px;">
            End-Sem AI Project Submission – HaveCrops Analytics
        </div>
        <div style="color:#5C6370;">{PROJECT_AUTHOR} · {ACADEMIC_YEAR}</div>
        <div class="foot-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
