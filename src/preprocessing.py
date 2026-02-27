import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Constructs a sophisticated Scikit-learn preprocessing pipeline.
    
    Args:
        numeric_features (list): Numerical columns.
        categorical_features (list): Categorical columns.
        
    Returns:
        ColumnTransformer: Preprocessing object.
    """
    if not numeric_features or not categorical_features:
        raise ValueError("Feature lists cannot be empty.")
        
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def prepare_data(df: pd.DataFrame, target_col: str = 'Yield'):
    """
    Standard data preparation: X/y split and Train-Test partition.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
