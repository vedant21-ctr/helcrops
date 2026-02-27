import pandas as pd
import numpy as np
import os

def generate_sample_data(output_path):
    """Generates a sample agricultural dataset for crop yield prediction."""
    np.random.seed(42)
    n_samples = 1000
    
    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Soybean']
    soil_types = ['Clay', 'Sandy', 'Loamy', 'Silt', 'Peaty']
    
    data = {
        'Rainfall': np.random.normal(500, 150, n_samples),
        'Soil_Type': np.random.choice(soil_types, n_samples),
        'Fertilizer_Used': np.random.normal(100, 30, n_samples),
        'Soil_pH': np.random.normal(6.5, 0.5, n_samples),
        'Crop_Type': np.random.choice(crops, n_samples)
    }
    
    # Calculate Yield with some logic + noise
    # Base yield + factors
    yield_val = (
        0.05 * data['Rainfall'] +
        0.2 * data['Fertilizer_Used'] +
        5.0 * (7.0 - np.abs(data['Soil_pH'] - 6.5)) +
        np.array([10 if c == 'Wheat' else 15 if c == 'Rice' else 12 for c in data['Crop_Type']]) +
        np.random.normal(0, 5, n_samples)
    )
    
    data['Yield'] = yield_val
    
    df = pd.DataFrame(data)
    
    # Introduce missing values for demonstration
    for col in ['Rainfall', 'Fertilizer_Used', 'Soil_pH']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset created at {output_path}")

if __name__ == "__main__":
    generate_sample_data("crop_yield_project/data/sample_farm_data.csv")
