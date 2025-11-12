"""
Machine Learning Analysis Script for Fraud Detection

This script implements machine learning models for fraud detection:
1. Data preprocessing and feature engineering
2. Model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
3. Model evaluation and validation
4. Feature importance analysis
5. Model interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
np.random.seed(42)

def load_and_prepare_data(data_path):
    """Load and prepare data for modeling."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")
    print(f"Target distribution:\n{df['isFraud'].value_counts()}")
    
    # Select features for modeling
    key_features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 
                    'addr1', 'addr2', 'dist1', 'dist2']
    key_features = [f for f in key_features if f in df.columns]
    
    # Add some C and D features if available
    c_features = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()][:10]
    d_features = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()][:10]
    
    features = key_features + c_features + d_features
    features = [f for f in features if f in df.columns]
    
    print(f"\nSelected {len(features)} features for modeling")
    
    # Prepare data
    X = df[features].copy()
    y = df['isFraud'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Missing values in X: {X.isnull().sum().sum()}")
    
    return X, y, features

def train_models(X_train, X_test, y_train, y_test, features, output_dir):
    """Train multiple ML models and evaluate them."""
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    models = {}
    predictions = {}
    predictions_proba = {}
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = lr_pred
    predictions_proba['Logistic Regression'] = lr_pred_proba
    
    print("Logistic Regression trained!")
    print(f"AUC-ROC: {roc_auc_score(y_test, lr_pred_proba):.4f}")
    print(f"Accuracy: {(lr_pred == y_test).mean():.4f}")
    
    # Model 2: Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    predictions_proba['Random Forest'] = rf_pred_proba
    
    print("Random Forest trained!")
    print(f"AUC-ROC: {roc_auc_score(y_test, rf_pred_proba):.4f}")
    print(f"Accuracy: {(rf_pred == y_test).mean():.4f}")
    
    # Model 3: XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric='auc'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    predictions_proba['XGBoost'] = xgb_pred_proba
    
    print("XGBoost trained!")
    print(f"AUC-ROC: {roc_auc_score(y_test, xgb_pred_proba):.4f}")
    print(f"Accuracy: {(xgb_pred == y_test).mean():.4f}")
    
    # Model 4: LightGBM
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_pred
    predictions_proba['LightGBM'] = lgb_pred_proba
    
    print("LightGBM trained!")
    print(f"AUC-ROC: {roc_auc_score(y_test, lgb_pred_proba):.4f}")
    print(f"Accuracy: {(lgb_pred == y_test).mean():.4f}")
    
    return models, predictions, predictions_proba, scaler

def evaluate_models(y_test, predictions, predictions_proba, output_dir):
    """Evaluate and compare models."""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    for name, pred_proba in predictions_proba.items():
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        auc = roc_auc_score(y_test, pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC curves saved to {output_dir}/roc_curves.png")
    
    # Model comparison table
    comparison = []
    for name, (pred, pred_proba) in zip(predictions.keys(), zip(predictions.values(), predictions_proba.values())):
        auc = roc_auc_score(y_test, pred_proba)
        accuracy = (pred == y_test).mean()
        comparison.append({
            'Model': name,
            'AUC-ROC': auc,
            'Accuracy': accuracy
        })
    
    comparison_df = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print("="*80)
    print(comparison_df.sort_values('AUC-ROC', ascending=False).to_string(index=False))
    
    return comparison_df

def feature_importance_analysis(models, features, X_train, output_dir):
    """Analyze feature importance."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Feature importance (using Random Forest and XGBoost)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest feature importance
    if 'Random Forest' in models:
        rf_importance = pd.DataFrame({
            'feature': features,
            'importance': models['Random Forest'].feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        axes[0].barh(range(len(rf_importance)), rf_importance['importance'])
        axes[0].set_yticks(range(len(rf_importance)))
        axes[0].set_yticklabels(rf_importance['feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest - Top 20 Feature Importance')
        axes[0].invert_yaxis()
    
    # XGBoost feature importance
    if 'XGBoost' in models:
        xgb_importance = pd.DataFrame({
            'feature': features,
            'importance': models['XGBoost'].feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        axes[1].barh(range(len(xgb_importance)), xgb_importance['importance'])
        axes[1].set_yticks(range(len(xgb_importance)))
        axes[1].set_yticklabels(xgb_importance['feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('XGBoost - Top 20 Feature Importance')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFeature importance plot saved to {output_dir}/feature_importance.png")
    
    if 'Random Forest' in models:
        print("\nTop 10 Features (Random Forest):")
        print(rf_importance.head(10).to_string(index=False))
    if 'XGBoost' in models:
        print("\nTop 10 Features (XGBoost):")
        print(xgb_importance.head(10).to_string(index=False))

def save_best_model(models, predictions_proba, y_test, scaler, output_dir):
    """Save the best performing model."""
    print("\n" + "="*80)
    print("SAVING BEST MODEL")
    print("="*80)
    
    # Find best model based on AUC-ROC
    best_model_name = None
    best_auc = 0
    
    for name, pred_proba in predictions_proba.items():
        auc = roc_auc_score(y_test, pred_proba)
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
    
    if best_model_name:
        best_model = models[best_model_name]
        model_dir = Path(output_dir).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(best_model, model_dir / 'best_model.pkl')
        joblib.dump(scaler, model_dir / 'scaler.pkl')
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best AUC-ROC: {best_auc:.4f}")
        print(f"Model saved to {model_dir / 'best_model.pkl'}")
        print(f"Scaler saved to {model_dir / 'scaler.pkl'}")
    else:
        print("No model to save!")

def main():
    """Main function to run ML analysis."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'fraud_data.csv'
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    X, y, features = load_and_prepare_data(data_path)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    print(f"Test set: {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
    
    # Train models
    models, predictions, predictions_proba, scaler = train_models(
        X_train, X_test, y_train, y_test, features, output_dir
    )
    
    # Evaluate models
    comparison_df = evaluate_models(y_test, predictions, predictions_proba, output_dir)
    
    # Feature importance analysis
    feature_importance_analysis(models, features, X_train, output_dir)
    
    # Save best model
    save_best_model(models, predictions_proba, y_test, scaler, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("MACHINE LEARNING ANALYSIS SUMMARY")
    print("="*80)
    best_model_name = comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']
    best_auc = comparison_df['AUC-ROC'].max()
    print(f"\nBest Model: {best_model_name}")
    print(f"Best AUC-ROC: {best_auc:.4f}")
    print(f"\nModels trained: {len(models)}")
    print(f"Features used: {len(features)}")
    print("\nAnalysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

