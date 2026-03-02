import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import generate_sample_data, get_yield_category, generate_parameter_alerts
from src.preprocessing import get_preprocessing_pipeline, prepare_data
from src.model_training import train_models, get_feature_importance
from src.evaluation import compare_models

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="HaveCrops | Statistical Yield Platform",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bespoke Technical Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;500;700&display=swap');

        /* Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes softPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
        .animate-fade-in-up { animation: fadeInUp 0.5s ease-out forwards; }
        .animate-fade-in { animation: fadeIn 0.4s ease-out forwards; }
        .animate-pulse { animation: softPulse 2.5s ease-in-out infinite; }
        [data-testid="stVerticalBlock"] > div { transition: transform 0.2s ease, box-shadow 0.2s ease; }
        [data-testid="stVerticalBlock"] > div:hover { transform: translateY(-2px); }

        /* Global Styling */
        * { font-family: 'Space Grotesk', sans-serif; }
        code, .stMarkdown code { font-family: 'IBM Plex Mono', monospace !important; }
        .main { background-color: #f4f4f4; }
        
        /* Industrial Header */
        .header-container { 
            background-color: #1a1c1e;
            padding: 3.5rem 2rem;
            border-bottom: 6px solid #d35400;
            color: #ffffff;
            text-align: left;
            margin-bottom: 2.5rem;
            animation: fadeInUp 0.6s ease-out;
        }
        
        .header-title { 
            font-size: 3.2rem; 
            font-weight: 700; 
            margin: 0; 
            letter-spacing: -2px;
            color: #ffffff;
        }
        .header-subtitle { 
            font-size: 1.1rem; 
            opacity: 0.8; 
            font-weight: 400; 
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        /* Technical Cards */
        .stat-card {
            background: #ffffff;
            padding: 1.5rem;
            border: 1px solid #d1d1d1;
            border-radius: 2px;
            box-shadow: 4px 4px 0px 0px #1a1c1e;
            animation: fadeInUp 0.5s ease-out;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 1rem; 
            padding: 10px 0;
            border-bottom: 2px solid #d1d1d1; 
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 500;
            color: #1a1c1e;
            background-color: #e0e0e0;
            padding: 8px 25px;
            border-radius: 0px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] { 
            color: #ffffff !important; 
            background-color: #1a1c1e !important;
            border-bottom: none !important;
        }
        
        /* Prediction Result Box (Solid Industrial) */
        .predict-box {
            background: #ffffff;
            padding: 2.5rem;
            border: 2px solid #1a1c1e;
            border-radius: 4px;
            text-align: center;
            box-shadow: 10px 10px 0px 0px #d35400;
        }
        
        /* Footer */
        .footer {
            text-align: left;
            padding: 2rem;
            color: #1a1c1e;
            font-size: 0.85rem;
            margin-top: 4rem;
            border-top: 2px solid #1a1c1e;
            background: #e0e0e0;
            animation: fadeIn 0.5s ease-out;
        }
        /* Grid consistency: equal column gaps */
        [data-testid="column"] { min-width: 0; }
    </style>
""", unsafe_allow_html=True)

# --- DATA INITIALIZATION ---
@st.cache_data
def load_and_prep_data():
    data_path = "data/sample_farm_data.csv"
    if not os.path.exists(data_path):
        generate_sample_data(data_path)
    df = pd.read_csv(data_path)
    return df

df = load_and_prep_data()
numeric_features = ['Rainfall', 'Fertilizer_Used', 'Soil_pH']
categorical_features = ['Soil_Type', 'Crop_Type']

# --- ML CACHING ---
@st.cache_resource
def get_trained_models(_df, num_feat, cat_feat):
    X_train, X_test, y_train, y_test = prepare_data(_df)
    preprocessor = get_preprocessing_pipeline(num_feat, cat_feat)
    models = train_models(preprocessor, X_train, y_train)
    return models, X_test, y_test

trained_models, X_test, y_test = get_trained_models(df, numeric_features, categorical_features)

# --- PAGE HEADER ---
st.markdown("""
    <div class="header-container">
        <div class="header-title">🚜 HaveCrops Analytics</div>
        <div class="header-subtitle">Empirical Yield Modeling & Agronomic Data System</div>
    </div>
""", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab_home, tab_predict, tab_analytics, tab_perf, tab_about = st.tabs([
    "📁 Data Summary", "🧮 Yield Modeler", "📉 Data Visuals", "📋 Model Evaluation", "ℹ️ System Info"
])

# --- SIDEBAR INPUTS (REACTIVE) ---
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.markdown("Modify system inputs to trigger dynamic modeling calculations.")

s_rain = st.sidebar.number_input("Average Rainfall (mm)", 200, 1200, 600)
s_ph = st.sidebar.slider("Soil pH Level", 4.0, 9.5, 6.5, 0.1)
s_fert = st.sidebar.number_input("Fertilizer Amount (kg/ha)", 0, 300, 120)
s_soil = st.sidebar.selectbox("Soil Type", df['Soil_Type'].unique())
s_crop = st.sidebar.selectbox("Crop Type", df['Crop_Type'].unique())
s_model = st.sidebar.selectbox("Select ML Model", list(trained_models.keys()))

# --- CORE PREDICTION LOGIC (RUNS EVERY RERENDER) ---
input_df = pd.DataFrame({
    'Rainfall': [s_rain],
    'Soil_Type': [s_soil],
    'Fertilizer_Used': [s_fert],
    'Soil_pH': [s_ph],
    'Crop_Type': [s_crop]
})

pred = trained_models[s_model].predict(input_df)[0]
cat = get_yield_category(pred, df)
alerts = generate_parameter_alerts(s_rain, s_ph, s_fert)

# --- TAB 1: HOME ---
with tab_home:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.write("")
        st.markdown("### Empirical Agricultural Analysis")
        st.markdown("""
        HaveCrops Analytics utilizes historical multivariate datasets to generate yield estimates based on environmental regression models. 
        By processing rainfall records, soil characteristic data, and local crop performance, the system provides 
        statistical projections for resource planning.
        """)
        # Agriculture / data analytics visual (replaces generic illustration)
        st.markdown("""
        <div class="animate-fade-in-up" style="
            background: linear-gradient(135deg, #1a3d2e 0%, #2d5a45 50%, #1a1c1e 100%);
            border-radius: 12px; padding: 2rem; border: 2px solid #d35400;
            text-align: center; color: #fff; min-height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">🌾📊</div>
            <div style="font-size: 1.1rem; font-weight: 600; letter-spacing: 1px;">Yield × Environment × Soil</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">Data-driven projections for resource planning</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Dataset Overview")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Total Records", f"{len(df)}", "Rows")
        m_col2.metric("Mean Yield", f"{df['Yield'].mean():.2f}", "Q/ha")
        
        st.write("#### 📝 Sample Data (First 10 Rows)")
        st.dataframe(df.head(10).style.background_gradient(cmap='Greens', subset=['Yield']))

# --- TAB 2: PREDICTION ---
with tab_predict:
    st.markdown("### 🧮 Statistical Yield Predictor")
    
    p_col1, p_col2 = st.columns([1, 1], gap="medium")
    
    with p_col1:
        st.markdown(f"""
            <div class="predict-box animate-fade-in-up" style="animation-delay: 0.1s;">
                <h2 style='color:#1a1c1e;'>Estimated Yield</h2>
                <h1 class="animate-pulse" style='color:#d35400; font-size:5rem; margin:0;'>{pred:.2f}</h1>
                <p style='font-size:1.2rem; color:#1a1c1e;'>Standard Q/ha</p>
                <hr style='border-color: #1a1c1e'>
                <div style='background:#e0e0e0; padding:15px; border-radius:0px; border: 1px solid #1a1c1e;'>
                    <strong style='color:#1a1c1e;'>CLASSIFICATION:</strong> <span style='color:#d35400;'>{cat} Range</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with p_col2:
        st.markdown("### 🛡️ Parameter Threshold Warnings")
        for alert in alerts:
            st.warning(alert)
        
        st.markdown("#### 🔍 Input Summary")
        st.json({
            "Rainfall": s_rain,
            "pH": s_ph,
            "Fertilizer": s_fert,
            "Soil": s_soil,
            "Crop": s_crop
        })

# --- TAB 3: ANALYTICS (DYNAMIC) ---
with tab_analytics:
    st.markdown("### 📉 Statistical Distribution & Current Estimate")
    st.write("The red marker corresponds to the current selected parameters in the regression model.")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.write("#### 🌧 Yield vs Rainfall")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='Rainfall', y='Yield', hue='Crop_Type', palette='viridis', alpha=0.3, ax=ax)
        # Dynamic marker for current input
        ax.scatter(s_rain, pred, color='red', s=200, marker='*', label='Your Prediction', edgecolors='white', linewidth=2)
        ax.legend()
        st.pyplot(fig)
        
    with v_col2:
        st.write("#### 🧪 pH Impact Analysis")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(data=df, x='Soil_pH', y='Yield', scatter_kws={'alpha':0.2}, line_kws={'color':'#10b981'}, ax=ax)
        # Dynamic marker
        ax.scatter(s_ph, pred, color='red', s=200, marker='*', edgecolors='white', linewidth=2)
        st.pyplot(fig)
        
    st.write("#### 🚜 Yield Distribution (Your Selection Highlighted)")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Filter for the selected crop/soil to show context
    filtered_df = df[(df['Crop_Type'] == s_crop) | (df['Soil_Type'] == s_soil)]
    sns.boxplot(data=df, x='Soil_Type', y='Yield', hue='Crop_Type', palette='Greens', ax=ax)
    plt.axhline(pred, color='red', linestyle='--', alpha=0.6, label='Predicted Yield')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- TAB 4: PERFORMANCE ---
with tab_perf:
    st.markdown("### 🛠 Model Benchmarking")
    
    perf_df = compare_models(trained_models, X_test, y_test)
    
    st.table(perf_df.style.highlight_max(axis=0, subset=['R2 Score'], color='#a5d6a7').highlight_min(axis=0, subset=['MAE', 'RMSE'], color='#a5d6a7'))
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("#### 📊 Metric Comparison")
        m_fig, m_ax = plt.subplots()
        perf_df[['MAE', 'RMSE']].plot(kind='bar', ax=m_ax, color=['#2e7d32', '#81c784'])
        plt.title("Error Metrics (Lower is better)")
        st.pyplot(m_fig)
        
    with c2:
        st.write("#### 💡 Decision Tree Feature Importance")
        importance_df = get_feature_importance(trained_models['Decision Tree'], numeric_features, categorical_features)
        if importance_df is not None:
            fig, ax = plt.subplots()
            sns.barplot(data=importance_df.head(8), x='Importance', y='Feature', palette='Greens_r', ax=ax)
            st.pyplot(fig)

    st.markdown("""
        > **Academic Summary:** The Decision Tree Regressor currently shows superior performance (Higher R²) compared to Linear Regression because it can capture non-linear interactions between Soil Types and Fertilizer amounts that the linear model misses.
    """)

# --- TAB 5: ABOUT ---
with tab_about:
    st.markdown("### 📜 Final Project Submission")
    st.markdown("""
    **Project Title:** SmartCrop Yield Prediction Platform  
    **Academic Year:** 2024-25 (Mid-Semester Submission)  
    **Course:** Machine Learning in Agriculture  
    
    **Abstract:**  
    This project demonstrates an end-to-end ML pipeline for precision agriculture. Starting from data synthetic generation (simulating real farm variability), the system performs robust preprocessing, multi-model training, and interactive visualization.
    
    **Technical Stack:**
    - **Back-end:** Scikit-learn, Pandas, NumPy
    - **Visuals:** Matplotlib, Seaborn
    - **UI Framework:** Streamlit (Custom Themed)
    - **Models:** Linear Regression, CART (Decision Trees)
    """)
    
    st.info("Developed for demonstrated research in crop yield optimization via data-driven methodologies.")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        Created by Vedant Satbhai | Mid-Sem Project Submission | Agri-Analytics v2.0
    </div>
""", unsafe_allow_html=True)
