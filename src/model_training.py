from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

def train_models(preprocessor, X_train, y_train):
    """
    Trains multiple ML models and returns a dictionary of trained pipelines.
    """
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
    return trained_pipelines

def get_feature_importance(pipeline, numeric_features, categorical_features):
    """
    Extracts feature importance for models that support it (e.g., Decision Tree).
    """
    try:
        model = pipeline.named_steps['regressor']
        if hasattr(model, 'feature_importances_'):
            # Handling transformed feature names
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_features_transformed = ohe.get_feature_names_out(categorical_features).tolist()
            
            # Combine all feature names
            all_features = numeric_features + cat_features_transformed
            
            # Create feature importance map
            importances = model.feature_importances_
            importance_map = dict(zip(all_features, importances))
            
            return sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
            
    except (AttributeError, KeyError):
        return None
    
    return None
