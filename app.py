import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import generate_sample_data, get_yield_category, get_actionable_insights
from src.preprocessing import get_preprocessing_pipeline, prepare_data
from src.model_training import train_models, get_feature_importance
from src.evaluation import compare_models

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="SmartCrop | Agri-Analytics Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern Agri-Theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        /* Global Styling */
        * { font-family: 'Outfit', sans-serif; }
        .main { background: linear-gradient(135deg, #f8faf8 0%, #edf2ed 100%); }
        
        /* Glassmorphism Header */
        .header-container { 
            background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
            padding: 4rem 2rem;
            border-radius: 0 0 3rem 3rem;
            color: white;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(5, 150, 105, 0.15);
            position: relative;
            overflow: hidden;
        }
        .header-container::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }
        @keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        .header-title { font-size: 4rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -1px; }
        .header-subtitle { font-size: 1.4rem; opacity: 0.9; font-weight: 300; }
        
        /* Premium Cards */
        .stat-card {
            background: white;
            padding: 2rem;
            border-radius: 1.5rem;
            border-bottom: 4px solid #10b981;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 2.5rem; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            font-weight: 600;
            color: #64748b;
            padding: 10px 20px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] { 
            color: #059669 !important; 
            background-color: #ecfdf5;
            border-bottom-color: #059669 !important; 
        }
        
        /* Prediction Result Box (Glassmorphic) */
        .predict-box {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            padding: 3rem;
            border-radius: 2rem;
            border: 1px solid rgba(16, 185, 129, 0.2);
            text-align: center;
            box-shadow: 0 25px 50px -12px rgba(16, 185, 129, 0.15);
            animation: fadeIn 0.8s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 3rem;
            color: #64748b;
            font-size: 1rem;
            margin-top: 5rem;
            border-top: 1px solid #e2e8f0;
            letter-spacing: 0.5px;
        }
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
        <div class="header-title">🌾 SmartCrop AI</div>
        <div class="header-subtitle">Advanced Crop Yield Analytics & Decision Support System</div>
    </div>
""", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab_home, tab_predict, tab_analytics, tab_perf, tab_about = st.tabs([
    "🏠 Home", "🎯 Prediction", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"
])

# --- TAB 1: HOME ---
with tab_home:
    col1, col2 = st.columns([1, 1], gap="large")
# --- SIDEBAR INPUTS (REACTIVE) ---
st.sidebar.header("🚜 Field Conditions")
st.sidebar.markdown("Adjust these values to see real-time updates across all analytics.")

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
insights = get_actionable_insights(s_rain, s_ph, s_fert)

# --- TAB 1: HOME ---
with tab_home:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.write("")
        st.markdown("### Welcome to the Future of Farming")
        st.markdown("""
        SmartCrop AI uses high-dimensional historical data to empower farmers with precision yield forecasting. 
        By analyzing rainfall patterns, soil chemistries, and regional crop performance, our models reduce agricultural 
        uncertainty and maximize ROI.
        """)
        st.image("https://img.freepik.com/free-vector/modern-agriculture-concept_23-2148197711.jpg", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Dataset Overview")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Total Records", f"{len(df)}", "Rows")
        m_col2.metric("Mean Yield", f"{df['Yield'].mean():.2f}", "Q/ha")
        
        st.write("#### 📝 Sample Data (First 10 Rows)")
        st.dataframe(df.head(10).style.background_gradient(cmap='Greens', subset=['Yield']))

# --- TAB 2: PREDICTION ---
with tab_predict:
    st.markdown("### 🎯 Real-Time Yield Calculator")
    
    p_col1, p_col2 = st.columns([1, 1], gap="medium")
    
    with p_col1:
        st.markdown(f"""
            <div class="predict-box">
                <h2 style='color:#064e3b;'>Forecasted Yield</h2>
                <h1 style='color:#059669; font-size:5rem; margin:0;'>{pred:.2f}</h1>
                <p style='font-size:1.2rem; color:#64748b;'>Quintals per Hectare</p>
                <hr style='border-color: rgba(16,185,129,0.1)'>
                <div style='background:#f0fdf4; padding:15px; border-radius:15px; border: 1px solid #dcfce7;'>
                    <strong style='color:#166534;'>PERFORMANCE:</strong> <span style='color:#10b981;'>{cat} Yield</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with p_col2:
        st.markdown("### 💡 AI-Powered Insights")
        for insight in insights:
            st.info(insight)
        
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
    st.markdown("### 📈 Live Analytics: Where your farm stands")
    st.write("The red dot indicates your current simulated inputs relative to historical data.")
    
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
    **Project Title:** SmartCrop AI Yield Prediction Platform  
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
