"""
Machine Learning Analysis Script
Implements various ML algorithms for predicting Fuel Consumption and CO2 Emissions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load and prepare data"""
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FuelConsumption.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """Preprocess data for ML"""
    # Encode categorical variables
    le_make = LabelEncoder()
    le_class = LabelEncoder()
    le_transmission = LabelEncoder()
    le_fuel = LabelEncoder()
    
    df_ml = df.copy()
    df_ml['MAKE_encoded'] = le_make.fit_transform(df_ml['MAKE'])
    df_ml['VEHICLE CLASS_encoded'] = le_class.fit_transform(df_ml['VEHICLE CLASS'])
    df_ml['TRANSMISSION_encoded'] = le_transmission.fit_transform(df_ml['TRANSMISSION'])
    df_ml['FUEL_encoded'] = le_fuel.fit_transform(df_ml['FUEL'])
    
    # Select features
    features = ['Year', 'ENGINE SIZE', 'CYLINDERS', 'MAKE_encoded', 
                'VEHICLE CLASS_encoded', 'TRANSMISSION_encoded', 'FUEL_encoded']
    X = df_ml[features]
    y_fuel = df_ml['FUEL CONSUMPTION']
    y_co2 = df_ml['COEMISSIONS']
    
    return X, y_fuel, y_co2, features

def train_models(X, y_fuel, y_co2, features):
    """Train ML models"""
    print("="*50)
    print("MACHINE LEARNING ANALYSIS")
    print("="*50)
    
    # Split data for fuel consumption
    X_train, X_test, y_fuel_train, y_fuel_test = train_test_split(
        X, y_fuel, test_size=0.2, random_state=42
    )
    
    # Split data for CO2 emissions
    X_train2, X_test2, y_co2_train, y_co2_test = train_test_split(
        X, y_co2, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models for Fuel Consumption
    print("\n--- Fuel Consumption Prediction ---")
    
    # Linear Regression
    lr_fuel = LinearRegression()
    lr_fuel.fit(X_train_scaled, y_fuel_train)
    y_fuel_pred_lr = lr_fuel.predict(X_test_scaled)
    print(f"\nLinear Regression:")
    print(f"  R2 Score: {r2_score(y_fuel_test, y_fuel_pred_lr):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_fuel_test, y_fuel_pred_lr)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_fuel_test, y_fuel_pred_lr):.4f}")
    
    # Random Forest
    rf_fuel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_fuel.fit(X_train, y_fuel_train)
    y_fuel_pred_rf = rf_fuel.predict(X_test)
    print(f"\nRandom Forest:")
    print(f"  R2 Score: {r2_score(y_fuel_test, y_fuel_pred_rf):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_fuel_test, y_fuel_pred_rf)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_fuel_test, y_fuel_pred_rf):.4f}")
    
    # Gradient Boosting
    gb_fuel = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_fuel.fit(X_train, y_fuel_train)
    y_fuel_pred_gb = gb_fuel.predict(X_test)
    print(f"\nGradient Boosting:")
    print(f"  R2 Score: {r2_score(y_fuel_test, y_fuel_pred_gb):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_fuel_test, y_fuel_pred_gb)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_fuel_test, y_fuel_pred_gb):.4f}")
    
    # Train models for CO2 Emissions
    print("\n--- CO2 Emissions Prediction ---")
    
    rf_co2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_co2.fit(X_train2, y_co2_train)
    y_co2_pred = rf_co2.predict(X_test2)
    print(f"\nRandom Forest:")
    print(f"  R2 Score: {r2_score(y_co2_test, y_co2_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_co2_test, y_co2_pred)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_co2_test, y_co2_pred):.4f}")
    
    # Save models
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(rf_fuel, os.path.join(model_dir, 'random_forest_fuel.pkl'))
    joblib.dump(rf_co2, os.path.join(model_dir, 'random_forest_co2.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print("\n✓ Models saved!")
    
    return rf_fuel, rf_co2

def main():
    """Main function"""
    df = load_data()
    X, y_fuel, y_co2, features = preprocess_data(df)
    train_models(X, y_fuel, y_co2, features)
    print("\n" + "="*50)
    print("ML ANALYSIS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()


