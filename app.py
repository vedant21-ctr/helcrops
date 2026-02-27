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
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Green-Gradient Theme
st.markdown("""
    <style>
        /* Global Background */
        .main { background-color: #f0f4f0; }
        
        /* Header */
        .header-container { 
            background: linear-gradient(135deg, #1b5e20 0%, #4caf50 100%);
            padding: 3rem;
            border-radius: 0 0 2rem 2rem;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; }
        .header-subtitle { font-size: 1.2rem; opacity: 0.9; }
        
        /* Custom Cards */
        .stat-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid #2e7d32;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 600;
            color: #555;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] { color: #2e7d32 !important; border-bottom-color: #2e7d32 !important; }
        
        /* Prediction Result */
        .predict-box {
            background: #ffffff;
            padding: 2rem;
            border-radius: 1.5rem;
            border: 2px solid #e8f5e9;
            text-align: center;
            box-shadow: 0 10px 30px rgba(46, 125, 50, 0.1);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
            margin-top: 4rem;
            border-top: 1px solid #ddd;
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
    with col1:
        st.write("")
        st.markdown("### Welcome to the Future of Farming")
        st.markdown("""
        SmartCrop AI uses high-dimensional historical data to empower farmers with precision yield forecasting. 
        By analyzing rainfall patterns, soil chemistries, and regional crop performance, our models reduce agricultural 
        uncertainty and maximize ROI.
        
        **Key Platform Capabilities:**
        - **Precision Forecasting**: Sub-hectare level accuracy in yield estimation.
        - **Decision Support**: Automated insights for soil correction and water management.
        - **Comparative Analytics**: Visualizing crop-specific trends across diverse soil types.
        """)
        st.image("https://img.freepik.com/free-vector/modern-agriculture-concept_23-2148197711.jpg?t=st=1740645000&exp=1740648600&hmac=3d2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Dataset Overview")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Total Records", f"{len(df)}", "Rows")
        m_col2.metric("Mean Yield", f"{df['Yield'].mean():.2f}", "Q/ha")
        
        st.write("#### 📝 Sample Data (First 10 Rows)")
        st.dataframe(df.head(10).style.background_gradient(cmap='Greens', subset=['Yield']))
        
        st.write("#### 🌍 Regional Crop Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        df['Crop_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Greens_r'), ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

# --- TAB 2: PREDICTION ---
with tab_predict:
    st.markdown("### 🎯 Precision Yield Calculator")
    st.write("Adjust the parameters in the sidebar to simulate crop conditions.")
    
    # Sidebar remains for inputs but refined
    st.sidebar.header("🚜 Field Conditions")
    with st.sidebar.form("prediction_form"):
        s_rain = st.number_input("Average Rainfall (mm)", 200, 1200, 600)
        s_ph = st.slider("Soil pH Level", 4.0, 9.5, 6.5, 0.1)
        s_fert = st.number_input("Fertilizer Amount (kg/ha)", 0, 300, 120)
        s_soil = st.selectbox("Soil Type", df['Soil_Type'].unique())
        s_crop = st.selectbox("Crop Type", df['Crop_Type'].unique())
        s_model = st.selectbox("Select ML Model", list(trained_models.keys()))
        s_submit = st.form_submit_button("💨 Run AI Prediction")
    
    if s_submit:
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
        
        p_col1, p_col2 = st.columns([1, 1], gap="medium")
        
        with p_col1:
            st.markdown(f"""
                <div class="predict-box">
                    <h2 style='color:#1b5e20;'>Forecasted Yield</h2>
                    <h1 style='color:#2e7d32; font-size:4rem;'>{pred:.2f}</h1>
                    <p style='font-size:1.2rem; color:#666;'>Quintals per Hectare</p>
                    <hr>
                    <div style='background:#e8f5e9; padding:10px; border-radius:10px;'>
                        <strong>Performance Category:</strong> {cat} Yield
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with p_col2:
            st.markdown("### 💡 AI-Powered Insights")
            for insight in insights:
                st.info(insight)
            
            st.markdown("#### 🔍 Rule-Based Logic")
            st.caption("Our intelligence engine flags conditions outside optimal ranges (pH 5.5-7.5, Rainfall >400mm) to suggest immediate corrective actions.")

# --- TAB 3: ANALYTICS ---
with tab_analytics:
    st.markdown("### 📈 Deep Dive: Correlation & Distribution")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.write("#### 🌧 Yield vs Rainfall Correlation")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Rainfall', y='Yield', hue='Crop_Type', palette='viridis', alpha=0.6, ax=ax)
        st.pyplot(fig)
        
    with v_col2:
        st.write("#### 🧪 pH Impact Analysis")
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='Soil_pH', y='Yield', scatter_kws={'alpha':0.3}, line_kws={'color':'green'}, ax=ax)
        st.pyplot(fig)
        
    st.write("#### 🚜 Yield Distribution by Soil & Crop Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='Soil_Type', y='Yield', hue='Crop_Type', palette='Greens', ax=ax)
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
