"""
Machine Learning Analysis for Cybersecurity Attacks Dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare data for ML"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    if '.' in df.columns:
        df = df.drop(columns=['.'])
    
    # Parse Time column
    if 'Time' in df.columns:
        def parse_time(time_str):
            if pd.isna(time_str):
                return None, None
            try:
                if '-' in str(time_str):
                    start, end = str(time_str).split('-')
                    return int(start), int(end)
                else:
                    return int(time_str), int(time_str)
            except:
                return None, None
        
        time_parsed = df['Time'].apply(parse_time)
        df['Time_Start'] = [t[0] for t in time_parsed]
        df['Time_End'] = [t[1] for t in time_parsed]
        df['Time_Duration'] = df['Time_End'] - df['Time_Start']
        df['Datetime_Start'] = pd.to_datetime(df['Time_Start'], unit='s', errors='coerce')
        df['Hour'] = df['Datetime_Start'].dt.hour
        df['DayOfWeek'] = df['Datetime_Start'].dt.day_name()
        df['Month'] = df['Datetime_Start'].dt.month
    
    return df

def prepare_features(df):
    """Prepare features for ML"""
    # Select features
    features = ['Source Port', 'Destination Port', 'Hour', 'Month']
    if 'Time_Duration' in df.columns:
        features.append('Time_Duration')
    
    # Encode categorical variables
    le_protocol = LabelEncoder()
    le_category = LabelEncoder()
    
    if 'Protocol' in df.columns:
        df['Protocol_encoded'] = le_protocol.fit_transform(df['Protocol'].astype(str))
        features.append('Protocol_encoded')
    
    # Encode Day of Week
    if 'DayOfWeek' in df.columns:
        day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                       'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        df['DayOfWeek_encoded'] = df['DayOfWeek'].map(day_mapping).fillna(0)
        features.append('DayOfWeek_encoded')
    
    # Target variable: Attack category
    if 'Attack category' in df.columns:
        df['Attack_category_encoded'] = le_category.fit_transform(df['Attack category'].astype(str))
        target = 'Attack_category_encoded'
    else:
        target = None
    
    # Select only available features
    features = [f for f in features if f in df.columns]
    
    # Handle missing values
    df[features] = df[features].fillna(df[features].median())
    
    X = df[features]
    y = df[target] if target else None
    
    return X, y, features, le_category

def train_models(X, y):
    """Train multiple ML models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'SVM': SVC(random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM and Logistic Regression
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results, scaler

def main():
    """Main function"""
    print("Loading data...")
    df = load_and_prepare_data('../../data/Cybersecurity_attacks.csv')
    
    print("Preparing features...")
    X, y, features, le_category = prepare_features(df)
    
    print(f"Features: {features}")
    print(f"Target classes: {len(le_category.classes_)}")
    print(f"Dataset shape: {X.shape}")
    
    print("\nTraining models...")
    results, scaler = train_models(X, y)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
    print("-"*80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} {result['f1']:<12.4f} {result['auc']:<12.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
    
    return results, best_model, scaler, features, le_category

if __name__ == "__main__":
    results, best_model, scaler, features, le_category = main()



