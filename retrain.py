"""Retrain the XGBoost pipeline with current sklearn/xgboost versions."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

print(f"sklearn:  {__import__('sklearn').__version__}")
print(f"xgboost:  {__import__('xgboost').__version__}")
print(f"joblib:   {joblib.__version__}")
print()

# Load data
df = pd.read_csv('cleaned_data.csv')
print(f"Loaded cleaned_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")

# Target = price; drop price, year, price_per_mile
target = 'price'
drop_cols = ['price', 'year', 'price_per_mile']
X = df.drop(columns=drop_cols)
y = df[target]

# Features used by app.py: model, transmission, fuelType, mileage, tax, mpg, engineSize, car_age
numeric_features = ['mileage', 'tax', 'mpg', 'engineSize', 'car_age']
categorical_features = ['model', 'transmission', 'fuelType']

for col in categorical_features:
    X[col] = X[col].astype(str)

print(f"Target:      {target}")
print(f"Numeric:     {numeric_features}")
print(f"Categorical: {categorical_features}")
print()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# Build fresh pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=200, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    ))
])

# Train
print("\nTraining...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.4f}")
print(f"RMSE:     {rmse:,.0f}")

# Delete old file first to be sure
if os.path.exists('best_pipeline.pkl'):
    os.remove('best_pipeline.pkl')
    print("\nDeleted old best_pipeline.pkl")

# Save fresh
joblib.dump(pipeline, 'best_pipeline.pkl')
print(f"Saved new best_pipeline.pkl")

# Verify by loading fresh and predicting
print("\n--- Verification ---")
loaded = joblib.load('best_pipeline.pkl')
test_row = pd.DataFrame([{
    'model': '3 series', 'transmission': 'automatic', 'fuelType': 'diesel',
    'mileage': 25000.0, 'tax': 145.0, 'mpg': 53.0, 'engineSize': 2.0, 'car_age': 8
}])
result = loaded.predict(test_row)
print(f"Test prediction: {result[0]:,.0f}")
print("Model loads and predicts WITHOUT errors!")
