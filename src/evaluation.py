import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Computes professional-grade regression metrics.
    
    Args:
        pipeline: Trained prediction pipeline.
        X_test: Test features.
        y_test: Ground truth.
        
    Returns:
        dict: Metric mappings.
    """
    try:
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return {
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'R2 Score': round(float(r2), 4),
            'MAPE (%)': round(float(mape) * 100, 2)
        }
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}

def compare_models(trained_pipelines, X_test, y_test):
    """
    Constructs a comparative dataframe for multiple models.
    """
    results = {}
    
    for name, pipeline in trained_pipelines.items():
        metrics = evaluate_model(pipeline, X_test, y_test)
        if metrics:
            results[name] = metrics
        
    return pd.DataFrame(results).T
