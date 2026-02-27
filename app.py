import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import generate_sample_data
from src.preprocessing import get_preprocessing_pipeline, prepare_data
from src.model_training import train_models, get_feature_importance
from src.evaluation import compare_models
import os

# Streamlit App Configuration
st.set_page_config(page_title="Crop Yield Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom Styling (Minimal, Agricultural-themed - Green)
st.markdown("""
    <style>
        .main { background-color: #f7f9f7; }
        .stButton>button { background-color: #2e7d32; color: white; border-radius: 8px; font-weight: bold; width: 100%; height: 50px; }
        .stButton>button:hover { background-color: #1b5e20; }
        .stSelectbox, .stNumberInput { border-radius: 8px; }
        .card { background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .title { color: #1b5e20; font-size: 2.5rem; font-weight: 800; margin-bottom: 20px; text-align: center; }
        .subtitle { color: #388e3c; font-size: 1.2rem; margin-bottom: 35px; text-align: center; }
        .prediction-box { background-color: #e8f5e9; border-left: 8px solid #2e7d32; padding: 25px; margin-top: 25px; border-radius: 8px; }
        .prediction-text { color: #1b5e20; font-size: 1.8rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Main Title Area
st.markdown('<div class="title">🌾 SmartCrop Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-driven agricultural insights for data-backed yield estimation</div>', unsafe_allow_html=True)

# Data Initialization
data_path = os.path.join("data", "sample_farm_data.csv")
if not os.path.exists(data_path):
    os.makedirs("data", exist_ok=True)
    generate_sample_data(data_path)

df = pd.read_csv(data_path)
numeric_features = ['Rainfall', 'Fertilizer_Used', 'Soil_pH']
categorical_features = ['Soil_Type', 'Crop_Type']

# Prepare Sidebar Inputs
st.sidebar.markdown("### 🌱 Farm Input Parameters")
st.sidebar.write("Configure your farm data for analysis")

with st.sidebar.form("input_form"):
    rainfall = st.number_input("Average Rainfall (mm)", min_value=0.0, max_value=2000.0, value=float(df['Rainfall'].mean()))
    fertilizer = st.number_input("Fertilizer Amount (kg/ha)", min_value=0.0, max_value=1000.0, value=float(df['Fertilizer_Used'].mean()))
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=float(df['Soil_pH'].mean()))
    soil = st.selectbox("Soil Type", options=df['Soil_Type'].unique())
    crop = st.selectbox("Current Crop Type", options=df['Crop_Type'].unique())
    model_choice = st.selectbox("Choice of Model", ["Linear Regression", "Decision Tree"])
    submit = st.form_submit_button("💨 Run ML Prediction")

# Data Preprocessing & Training
X_train, X_test, y_train, y_test = prepare_data(df)
preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
trained_models = train_models(preprocessor, X_train, y_train)

# Main Section Layout - Multi Column
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance Evaluation")
    
    comp_df = compare_models(trained_models, X_test, y_test)
    st.table(comp_df.style.highlight_max(axis=0, subset=['R2 Score'], color='#a5d6a7'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Importance (only for Decision Tree)
    if model_choice == "Decision Tree":
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 Important Yield Drivers")
        importance = get_feature_importance(trained_models["Decision Tree"], numeric_features, categorical_features)
        
        if importance:
            imp_df = pd.DataFrame(importance[:8], columns=['Feature', 'Importance'])
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax)
            ax.set_title("Top 8 Predictors")
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Yield Forecasting Results")
    
    if submit:
        # Create input df for prediction
        input_data = pd.DataFrame({
            'Rainfall': [rainfall],
            'Soil_Type': [soil],
            'Fertilizer_Used': [fertilizer],
            'Soil_pH': [ph],
            'Crop_Type': [crop]
        })
        
        prediction = trained_models[model_choice].predict(input_data)[0]
        
        st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1rem; color: #555;">Predicted Yield Estimation:</div>
                <div class="prediction-text">{prediction:.2f} Quintals/Hectare</div>
                <div style="font-size: 0.85rem; color: #777; margin-top: 10px;">
                    Model Used: {model_choice} | Basis: Historical Data Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter farm parameters in the sidebar and click 'Run ML Prediction' to see results.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple Raw Data sample
    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Dataset Snapshot")
    st.dataframe(df.head(5))
    st.markdown('</div>', unsafe_allow_html=True)
