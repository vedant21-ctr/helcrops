# 🌾 SmartCrop AI: Crop Yield Prediction System
### *Advanced Machine Learning Pipeline for Precision Agriculture*

---

## 📖 Project Overview
This project is a professional-grade **Mid-Semester Case Study** on predicting crop yields using high-dimensional historical agricultural data. The platform provides a comprehensive end-to-end Machine Learning pipeline that transforms raw environmental inputs into actionable harvesting forecasts.

### **Core Problem Statement**
Farmers often struggle with yield volatility due to complex interactions between soil chemistry, weather patterns, and crop varieties. This system aims to:
1. **Reduce Uncertainty**: Provide statistical yield estimates.
2. **Optimize Resources**: Suggest soil and water corrections based on inputs.
3. **Compare Models**: Benchmark different ML architectures for accuracy.

---

## 🏗 System Architecture
The application follows a **Modular Monolith Architecture** to ensure maintainability and scalability.

```mermaid
graph TD
    User([User Input via UI]) --> Logic[Streamlit Core]
    Logic --> Data[Data Loader / Synthetic Gen]
    Data --> Clean[Preprocessing Pipeline]
    Clean --> Train[Model Suite: Linear Reg | Decision Tree]
    Train --> Eval[Performance Benchmarking]
    Eval --> Dashboard[Visual Analytics Dashboard]
    Dashboard --> Insights[Actionable Agricultural Advice]
```

### **1. Data Preprocessing Pipeline**
Using Scikit-learn's `ColumnTransformer`, the system implements:
- **Numerical Features**: Median Imputation + Z-Score Standardization (StandardScaler).
- **Categorical Features**: Most-Frequent Imputation + One-Hot Encoding (OHE) for high-dimensional feature mapping.

### **2. Machine Learning Suite**
- **Linear Regression**: A parametric approach to model linear relationship between rainfall/fertilizer and yield.
- **Decision Tree Regressor**: A non-parametric model (CART) capable of capturing hierarchical and non-linear interactions between categorical soil types and environmental factors.

---

## 📊 Performance Benchmarking
The system evaluates models using standard regression metrics:
- **MAE (Mean Absolute Error)**: Average deviation of predictions.
- **RMSE (Root Mean Squared Error)**: Critical for penalizing large forecast errors.
- **R² Score**: Quantifying the variance explained by the model characteristics.
- **MAPE**: Percentage error analysis for relative accuracy assessment.

---

## 🚀 Deployment & Installation

### **Prerequisites**
- Python 3.8+
- Git

### **Installation Steps**
1. **Clone the project**:
   ```bash
   git clone https://github.com/vedant21-ctr/helcrops.git
   cd crop_yield_project
   ```

2. **Initialize Environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Platform**:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure
```text
crop_yield_project/
├── data/
│   └── sample_farm_data.csv    # Dynamic dataset generation
├── src/
│   ├── preprocessing.py         # Sklearn feature engineering
│   ├── model_training.py        # ML induction & feature importance
│   ├── evaluation.py            # Comparative analytics logic
│   └── utils.py                 # Rule-based inference & data scripts
├── app.py                       # Modern multi-tab dashboard
├── requirements.txt             # Academic dependency list
└── README.md                    # Submission documentation
```

---

## 📜 Academic Submission Details
- **Subject**: Machine Learning in Agriculture
- **Submission Type**: Mid-Semester Project
- **Version**: 2.0 (Professional Analytics Upgrade)
- **Developer**: Vedant Satbhai

> **Note**: This system is designed for demonstration purposes using synthetic logic that represents real-world agricultural trends.

---
© 2024 | SmartCrop Agri-Analytics Platform
