import pandas as pd
import numpy as np
import os

def generate_sample_data(output_path: str):
    """
    Generates a sample agricultural dataset for crop yield prediction.
    Introduces synthetic logic for rainfall, fertilizer, and pH impacts on yield.
    
    Args:
        output_path (str): The file path where the CSV should be saved.
    """
    try:
        np.random.seed(42)
        n_samples = 1000
        
        crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Soybean']
        soil_types = ['Clay', 'Sandy', 'Loamy', 'Silt', 'Peaty']
        
        data = {
            'Rainfall': np.random.normal(600, 200, n_samples).clip(200, 1200),
            'Soil_Type': np.random.choice(soil_types, n_samples),
            'Fertilizer_Used': np.random.normal(120, 40, n_samples).clip(0, 300),
            'Soil_pH': np.random.normal(6.5, 0.8, n_samples).clip(4, 9),
            'Crop_Type': np.random.choice(crops, n_samples)
        }
        
        # Calculate Yield with logical weights
        yield_val = (
            0.08 * data['Rainfall'] +
            0.15 * data['Fertilizer_Used'] +
            15.0 * (8.0 - np.abs(data['Soil_pH'] - 6.5)) +
            np.array([20 if c == 'Wheat' else 35 if c == 'Rice' else 25 for c in data['Crop_Type']]) +
            np.random.normal(0, 10, n_samples)
        )
        
        data['Yield'] = yield_val.clip(10, 200) # Ensure physical realism
        
        df = pd.DataFrame(data)
        
        # Introduce missing values (5%)
        for col in ['Rainfall', 'Fertilizer_Used', 'Soil_pH']:
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = np.nan
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Professional dataset created at {output_path}")
    except Exception as e:
        print(f"Error generating data: {e}")

def get_yield_category(yield_value, df):
    """
    Categorizes the yield based on historical distribution.
    
    Args:
        yield_value (float): Predicted yield.
        df (pd.DataFrame): Training dataset for reference.
    Returns:
        str: Low, Medium, or High
    """
    q1 = df['Yield'].quantile(0.33)
    q2 = df['Yield'].quantile(0.66)
    
    if yield_value <= q1:
        return "Low"
    elif yield_value <= q2:
        return "Medium"
    else:
        return "High"

def get_actionable_insights(rainfall, ph, fertilizer):
    """
    Provides rule-based agricultural advice.
    """
    insights = []
    
    if rainfall < 400:
        insights.append("💧 **Water Stress**: Low rainfall detected. Consider drip irrigation systems.")
    if ph < 5.5:
        insights.append("🧪 **Acidity Alert**: Soil is too acidic. Adding agricultural lime is recommended.")
    elif ph > 7.5:
        insights.append("🧪 **Alkalinity Alert**: Soil is alkaline. Consider adding sulfur or organic matter.")
    if fertilizer < 50:
        insights.append("🌱 **Nutrient Deficiency**: Low fertilizer usage. Conduct a soil test for N-P-K levels.")
        
    if not insights:
        insights.append("✅ **Balanced Conditions**: Environmental factors are stable for optimal growth.")
        
    return insights

if __name__ == "__main__":
    generate_sample_data("data/sample_farm_data.csv")
