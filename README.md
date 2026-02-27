# SmartCrop: Crop Yield Prediction System

## 🌾 Problem Understanding
The agricultural sector faces significant challenges in predicting crop yields due to unpredictable environmental factors. Accurate yield prediction helps farmers and stakeholders make informed decisions regarding resource allocation, budgeting, and harvesting strategies. This system leverages historical agricultural data including rainfall, soil properties, and fertilizer usage to provide data-driven yield estimations.

## 🚜 Agro-context Explanation
Crop yield is a complex outcome influenced by:
- **Rainfall**: Essential moisture for plant growth.
- **Soil pH**: Affects nutrient availability; most crops thrive in slightly acidic to neutral soil.
- **Fertilizer Usage**: Provides critical nutrients like Nitrogen, Phosphorus, and Potassium.
- **Soil Type**: Different soils (Clay, Sandy, Loamy) have varying water retention and aeration properties.
- **Crop Variety**: Each crop has unique genetic yield potential and environmental requirements.

## 📊 Input–Output Specification
- **Inputs**: 
  - Rainfall (mm)
  - Fertilizer Used (kg/ha)
  - Soil pH (0-14)
  - Soil Type (Categorical: Clay, Sandy, Loamy, Silt, Peaty)
  - Crop Type (Categorical: Wheat, Rice, Maize, Cotton, Soybean)
- **Output**: 
  - Predicted Yield (Quintals/Hectare)

## 🏗 System Architecture
```mermaid
graph LR
    A[User Input] --> B[Preprocessing Pipeline]
    B --> C[ML Model Training]
    C --> D[Evaluation & Comparison]
    D --> E[Interactive UI Output]
```
1. **User Input**: Collected via a modern Streamlit sidebar.
2. **Preprocessing**: Handled using Scikit-learn's `ColumnTransformer`.
   - **Numerical**: Mean Imputation + Standard Scaling.
   - **Categorical**: Most Frequent Imputation + One-Hot Encoding.
3. **ML Model**: Parallel training of Linear Regression and Decision Tree Regressor.
4. **Evaluation**: Metrics (MAE, RMSE, R²) are computed on a 20% test split.
5. **UI Output**: Professional green-themed dashboard with clear predictions and visualizations.

## 🚀 How to Run Locally
1. **Navigate to project directory**:
   ```bash
   cd crop_yield_project
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

## 🤖 Model Explanation & Evaluation
- **Linear Regression**: Ideal for establishing a baseline and understanding linear relationships between inputs and yield.
- **Decision Tree Regressor**: Capable of capturing non-linear patterns and complex interactions between soil types and climate data.
- **Metrics Summary**: 
  - **MAE (Mean Absolute Error)**: Average magnitude of prediction errors.
  - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily.
  - **R² Score**: Proportion of variance in yield explained by the model.

## 📸 Screenshots (Placeholders)
![Dashboard Overview](https://via.placeholder.com/800x450.png?text=SmartCrop+Dashboard+UI)
![Feature Importance Chart](https://via.placeholder.com/800x400.png?text=Feature+Importance+Analysis)
