from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import pandas as pd

def train_models(preprocessor, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a suite of regression models for academic comparison.
    
    Args:
        preprocessor: The fitted or unfitted ColumnTransformer.
        X_train: Feature set.
        y_train: Target set.
    """
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
    }
    
    trained_pipelines = {}
    
    for name, model in models.items():
        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            trained_pipelines[name] = pipeline
        except Exception as e:
            print(f"Error training {name}: {e}")
        
    return trained_pipelines

def get_feature_importance(pipeline, numeric_features: list, categorical_features: list):
    """
    Safely extracts and formats feature importance for interpreting model decisions.
    """
    try:
        model = pipeline.named_steps['regressor']
        if not hasattr(model, 'feature_importances_'):
            return None
            
        # Get names from OneHotEncoder in the pipeline
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_features_transformed = ohe.get_feature_names_out(categorical_features).tolist()
        
        all_features = numeric_features + cat_features_transformed
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return importance_df
            
    except (AttributeError, KeyError, ValueError) as e:
        print(f"Feature importance extraction failed: {e}")
        return None
