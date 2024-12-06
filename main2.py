import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def prepare_data(csv_path):
    """Prepare data for linear regression"""
    df = pd.read_csv(csv_path)
    
    # Select features and target
    features = ['age', 'bmi', 'children', 'smoker', 'sex', 'region']
    target = 'charges'
    
    X = df[features]
    y = df[target]
    
    return X, y

def create_regression_model():
    """Create preprocessing pipeline and linear regression model"""
    # Preprocessing for categorical columns
    categorical_features = ['smoker', 'sex', 'region']
    numeric_features = ['age', 'bmi', 'children']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def train_and_evaluate_model(csv_path):
    """Train linear regression and evaluate performance"""
    # Prepare data
    X, y = prepare_data(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_regression_model()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return model

# Run model
if __name__ == "__main__":
    csv_path = 'insurance.csv'
    trained_model = train_and_evaluate_model(csv_path)