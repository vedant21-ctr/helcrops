import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(pipeline, X_test, y_test):
    """
    Computes regression metrics for a given model.
    """
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 2)
    }

def compare_models(trained_pipelines, X_test, y_test):
    """
    Returns a dataframe comparing multiple models.
    """
    import pandas as pd
    
    results = {}
    
    for name, pipeline in trained_pipelines.items():
        results[name] = evaluate_model(pipeline, X_test, y_test)
        
    return pd.DataFrame(results).T
