"""
Machine Learning Analysis for Employee Dataset
Predictive modeling using various algorithms for salary and bonus prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

# Create results directories
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

print("="*80)
print("MACHINE LEARNING ANALYSIS - EMPLOYEE DATASET")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('data/processed/employees_cleaned.csv')

# =============================================================================
# 1. DATA PREPROCESSING FOR ML
# =============================================================================
print("\n1. DATA PREPROCESSING FOR ML")
print("="*80)

# Create a copy for ML
df_ml = df.copy()

# Feature engineering
df_ml['Start_Year'] = pd.to_datetime(df_ml['Start_Date']).dt.year
df_ml['Start_Month'] = pd.to_datetime(df_ml['Start_Date']).dt.month
df_ml['Years_of_Service'] = (pd.to_datetime('today') - pd.to_datetime(df_ml['Start_Date'])).dt.days / 365.25

# Handle missing values
df_ml['Gender'] = df_ml['Gender'].fillna('Unknown')
df_ml['Team'] = df_ml['Team'].fillna('Unknown')
df_ml['Senior_Management'] = df_ml['Senior_Management'].fillna(False)

# Encode categorical variables
le_gender = LabelEncoder()
le_team = LabelEncoder()
df_ml['Gender_encoded'] = le_gender.fit_transform(df_ml['Gender'])
df_ml['Team_encoded'] = le_team.fit_transform(df_ml['Team'])
df_ml['Senior_Management_encoded'] = df_ml['Senior_Management'].astype(int)

# Select features for modeling
feature_columns = ['Gender_encoded', 'Team_encoded', 'Senior_Management_encoded', 
                   'Bonus_pct', 'Years_of_Service', 'Start_Year', 'Start_Month']

# Remove rows with missing target or features
df_ml = df_ml.dropna(subset=['Salary'] + feature_columns)

print(f"Dataset shape for ML: {df_ml.shape}")
print(f"Features used: {feature_columns}")

# =============================================================================
# 2. SALARY PREDICTION
# =============================================================================
print("\n2. SALARY PREDICTION")
print("="*80)

# Prepare data for salary prediction
X_salary = df_ml[feature_columns]
y_salary = df_ml['Salary']

# Split data
X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

# Scale features
scaler_salary = StandardScaler()
X_train_salary_scaled = scaler_salary.fit_transform(X_train_salary)
X_test_salary_scaled = scaler_salary.transform(X_test_salary)

print(f"Training set size: {X_train_salary.shape[0]}")
print(f"Test set size: {X_test_salary.shape[0]}")

# =============================================================================
# 2.1 LINEAR REGRESSION
# =============================================================================
print("\n2.1. Linear Regression")

lr_model = LinearRegression()
lr_model.fit(X_train_salary_scaled, y_train_salary)
y_pred_lr = lr_model.predict(X_test_salary_scaled)

lr_r2 = r2_score(y_test_salary, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_lr))
lr_mae = mean_absolute_error(y_test_salary, y_pred_lr)

print(f"R² Score: {lr_r2:.4f}")
print(f"RMSE: ${lr_rmse:,.2f}")
print(f"MAE: ${lr_mae:,.2f}")

# Cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train_salary_scaled, y_train_salary, cv=5, scoring='r2')
print(f"Cross-validation R²: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")

# =============================================================================
# 2.2 RIDGE REGRESSION
# =============================================================================
print("\n2.2. Ridge Regression")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_salary_scaled, y_train_salary)
y_pred_ridge = ridge_model.predict(X_test_salary_scaled)

ridge_r2 = r2_score(y_test_salary, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_ridge))
ridge_mae = mean_absolute_error(y_test_salary, y_pred_ridge)

print(f"R² Score: {ridge_r2:.4f}")
print(f"RMSE: ${ridge_rmse:,.2f}")
print(f"MAE: ${ridge_mae:,.2f}")

# Hyperparameter tuning
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train_salary_scaled, y_train_salary)
print(f"Best alpha: {ridge_grid.best_params_['alpha']}")

# =============================================================================
# 2.3 RANDOM FOREST
# =============================================================================
print("\n2.3. Random Forest")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_salary, y_train_salary)
y_pred_rf = rf_model.predict(X_test_salary)

rf_r2 = r2_score(y_test_salary, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_rf))
rf_mae = mean_absolute_error(y_test_salary, y_pred_rf)

print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: ${rf_rmse:,.2f}")
print(f"MAE: ${rf_mae:,.2f}")

# Feature importance
rf_feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(rf_feature_importance)

# =============================================================================
# 2.4 GRADIENT BOOSTING
# =============================================================================
print("\n2.4. Gradient Boosting")

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_model.fit(X_train_salary, y_train_salary)
y_pred_gb = gb_model.predict(X_test_salary)

gb_r2 = r2_score(y_test_salary, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_gb))
gb_mae = mean_absolute_error(y_test_salary, y_pred_gb)

print(f"R² Score: {gb_r2:.4f}")
print(f"RMSE: ${gb_rmse:,.2f}")
print(f"MAE: ${gb_mae:,.2f}")

# Feature importance
gb_feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Gradient Boosting):")
print(gb_feature_importance)

# =============================================================================
# 2.5 XGBOOST
# =============================================================================
print("\n2.5. XGBoost")

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_salary, y_train_salary)
y_pred_xgb = xgb_model.predict(X_test_salary)

xgb_r2 = r2_score(y_test_salary, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test_salary, y_pred_xgb)

print(f"R² Score: {xgb_r2:.4f}")
print(f"RMSE: ${xgb_rmse:,.2f}")
print(f"MAE: ${xgb_mae:,.2f}")

# Feature importance
xgb_feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (XGBoost):")
print(xgb_feature_importance)

# =============================================================================
# 2.6 LIGHTGBM
# =============================================================================
print("\n2.6. LightGBM")

lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train_salary, y_train_salary)
y_pred_lgb = lgb_model.predict(X_test_salary)

lgb_r2 = r2_score(y_test_salary, y_pred_lgb)
lgb_rmse = np.sqrt(mean_squared_error(y_test_salary, y_pred_lgb))
lgb_mae = mean_absolute_error(y_test_salary, y_pred_lgb)

print(f"R² Score: {lgb_r2:.4f}")
print(f"RMSE: ${lgb_rmse:,.2f}")
print(f"MAE: ${lgb_mae:,.2f}")

# Feature importance
lgb_feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (LightGBM):")
print(lgb_feature_importance)

# =============================================================================
# 3. MODEL COMPARISON
# =============================================================================
print("\n3. MODEL COMPARISON")
print("="*80)

# Create comparison DataFrame
model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 
              'Gradient Boosting', 'XGBoost', 'LightGBM'],
    'R² Score': [lr_r2, ridge_r2, rf_r2, gb_r2, xgb_r2, lgb_r2],
    'RMSE': [lr_rmse, ridge_rmse, rf_rmse, gb_rmse, xgb_rmse, lgb_rmse],
    'MAE': [lr_mae, ridge_mae, rf_mae, gb_mae, xgb_mae, lgb_mae]
}).sort_values('R² Score', ascending=False)

print("\nModel Comparison (Salary Prediction):")
print(model_comparison)
model_comparison.to_csv('results/tables/model_comparison_salary.csv', index=False)

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² Score comparison
axes[0].barh(model_comparison['Model'], model_comparison['R² Score'], color='steelblue')
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison: R² Score')
axes[0].grid(True, alpha=0.3, axis='x')

# RMSE comparison
axes[1].barh(model_comparison['Model'], model_comparison['RMSE'], color='lightcoral')
axes[1].set_xlabel('RMSE ($)')
axes[1].set_title('Model Comparison: RMSE')
axes[1].grid(True, alpha=0.3, axis='x')

# MAE comparison
axes[2].barh(model_comparison['Model'], model_comparison['MAE'], color='lightgreen')
axes[2].set_xlabel('MAE ($)')
axes[2].set_title('Model Comparison: MAE')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/plots/model_comparison_salary.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. FEATURE IMPORTANCE VISUALIZATION
# =============================================================================
print("\n4. FEATURE IMPORTANCE VISUALIZATION")

# Plot feature importance for tree-based models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Random Forest
axes[0, 0].barh(rf_feature_importance['Feature'], rf_feature_importance['Importance'], color='steelblue')
axes[0, 0].set_title('Feature Importance: Random Forest')
axes[0, 0].set_xlabel('Importance')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Gradient Boosting
axes[0, 1].barh(gb_feature_importance['Feature'], gb_feature_importance['Importance'], color='lightcoral')
axes[0, 1].set_title('Feature Importance: Gradient Boosting')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# XGBoost
axes[1, 0].barh(xgb_feature_importance['Feature'], xgb_feature_importance['Importance'], color='lightgreen')
axes[1, 0].set_title('Feature Importance: XGBoost')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# LightGBM
axes[1, 1].barh(lgb_feature_importance['Feature'], lgb_feature_importance['Importance'], color='gold')
axes[1, 1].set_title('Feature Importance: LightGBM')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. PREDICTION VISUALIZATIONS
# =============================================================================
print("\n5. PREDICTION VISUALIZATIONS")

# Get best model
best_model_name = model_comparison.iloc[0]['Model']
best_models = {
    'Linear Regression': (lr_model, y_pred_lr, 'scaled'),
    'Ridge Regression': (ridge_model, y_pred_ridge, 'scaled'),
    'Random Forest': (rf_model, y_pred_rf, 'unscaled'),
    'Gradient Boosting': (gb_model, y_pred_gb, 'unscaled'),
    'XGBoost': (xgb_model, y_pred_xgb, 'unscaled'),
    'LightGBM': (lgb_model, y_pred_lgb, 'unscaled')
}

best_model, best_predictions, scale_type = best_models[best_model_name]

# Actual vs Predicted
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scatter plot: Actual vs Predicted
axes[0, 0].scatter(y_test_salary, best_predictions, alpha=0.5, s=50)
axes[0, 0].plot([y_test_salary.min(), y_test_salary.max()], 
                [y_test_salary.min(), y_test_salary.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Salary')
axes[0, 0].set_ylabel('Predicted Salary')
axes[0, 0].set_title(f'Actual vs Predicted: {best_model_name}')
axes[0, 0].grid(True, alpha=0.3)

# Residual plot
residuals = y_test_salary - best_predictions
axes[0, 1].scatter(best_predictions, residuals, alpha=0.5, s=50)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Salary')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title(f'Residual Plot: {best_model_name}')
axes[0, 1].grid(True, alpha=0.3)

# Distribution of residuals
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Distribution of Residuals: {best_model_name}')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot of residuals
from scipy import stats as scipy_stats
scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title(f'Q-Q Plot of Residuals: {best_model_name}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/prediction_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. SAVE MODELS
# =============================================================================
print("\n6. SAVE MODELS")

# Save best model
with open(f'results/models/best_model_salary.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save scaler
with open(f'results/models/scaler_salary.pkl', 'wb') as f:
    pickle.dump(scaler_salary, f)

# Save label encoders
with open(f'results/models/label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

with open(f'results/models/label_encoder_team.pkl', 'wb') as f:
    pickle.dump(le_team, f)

print(f"Best model ({best_model_name}) saved to: results/models/best_model_salary.pkl")

# =============================================================================
# 7. BONUS % PREDICTION (OPTIONAL)
# =============================================================================
print("\n7. BONUS % PREDICTION")

# Prepare data for bonus prediction (excluding Bonus_pct from features)
feature_columns_bonus = [col for col in feature_columns if col != 'Bonus_pct']
X_bonus = df_ml[feature_columns_bonus]
y_bonus = df_ml['Bonus_pct']

# Remove rows with missing target
X_bonus = X_bonus[y_bonus.notna()]
y_bonus = y_bonus[y_bonus.notna()]

# Split data
X_train_bonus, X_test_bonus, y_train_bonus, y_test_bonus = train_test_split(
    X_bonus, y_bonus, test_size=0.2, random_state=42
)

# Train XGBoost for bonus prediction
xgb_bonus_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_bonus_model.fit(X_train_bonus, y_train_bonus)
y_pred_bonus = xgb_bonus_model.predict(X_test_bonus)

bonus_r2 = r2_score(y_test_bonus, y_pred_bonus)
bonus_rmse = np.sqrt(mean_squared_error(y_test_bonus, y_pred_bonus))
bonus_mae = mean_absolute_error(y_test_bonus, y_pred_bonus)

print(f"R² Score: {bonus_r2:.4f}")
print(f"RMSE: {bonus_rmse:.4f}%")
print(f"MAE: {bonus_mae:.4f}%")

# Save bonus model
with open(f'results/models/xgb_model_bonus.pkl', 'wb') as f:
    pickle.dump(xgb_bonus_model, f)

print("\n" + "="*80)
print("MACHINE LEARNING ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nBest Model for Salary Prediction: {best_model_name}")
print(f"R² Score: {model_comparison.iloc[0]['R² Score']:.4f}")
print(f"RMSE: ${model_comparison.iloc[0]['RMSE']:,.2f}")
print(f"MAE: ${model_comparison.iloc[0]['MAE']:,.2f}")
print(f"\nResults saved in:")
print("- Models: results/models/")
print("- Tables: results/tables/")
print("- Plots: results/plots/")
