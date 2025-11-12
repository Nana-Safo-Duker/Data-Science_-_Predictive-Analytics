"""
Machine Learning Analysis for Email Spam Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_prepare_data():
    """Load and prepare data for ML"""
    df = pd.read_csv('../../data/emails_spam_clean.csv')
    
    # Clean text
    import re
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    return df

def feature_engineering(df):
    """Create features for ML models"""
    # TF-IDF Vectorization
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    
    # Count Vectorization (Bag of Words)
    print("Creating Bag of Words features...")
    count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_count = count_vectorizer.fit_transform(df['cleaned_text'])
    
    # Text statistics features
    print("Creating text statistics features...")
    df['text_length'] = df['cleaned_text'].str.len()
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df['avg_word_length'] = df['cleaned_text'].str.split().apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0
    )
    
    X_stats = df[['text_length', 'word_count', 'avg_word_length']].values
    
    # Combine features
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf, X_stats])
    
    y = df['spam'].values
    
    return X_tfidf, X_count, X_combined, y, tfidf_vectorizer, count_vectorizer

def train_and_evaluate_model(X, y, model, model_name, vectorizer=None):
    """Train and evaluate a model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # ROC-AUC if probabilities available
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  ROC-AUC: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"  Cross-validation F1 scores: {cv_scores}")
    print(f"  Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cv_scores': cv_scores
    }

def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, model_name, save_path):
    """Plot ROC curve"""
    if y_pred_proba is None:
        return
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_models(results):
    """Compare all models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Model': [r['model_name'] for r in results],
        'Accuracy': [r['accuracy'] for r in results],
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-Score': [r['f1'] for r in results],
        'ROC-AUC': [r['roc_auc'] if r['roc_auc'] else np.nan for r in results],
        'Mean CV F1': [r['cv_scores'].mean() for r in results]
    })
    
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for idx, metric in enumerate(metrics):
        axes[idx//2, idx%2].barh(comparison['Model'], comparison[metric], color='steelblue')
        axes[idx//2, idx%2].set_xlabel(metric)
        axes[idx//2, idx%2].set_title(f'{metric} Comparison')
        axes[idx//2, idx%2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('../../output/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find best model
    best_model_idx = comparison['F1-Score'].idxmax()
    best_model = comparison.loc[best_model_idx, 'Model']
    print(f"\nBest Model (by F1-Score): {best_model}")
    print(f"  F1-Score: {comparison.loc[best_model_idx, 'F1-Score']:.4f}")
    print(f"  Accuracy: {comparison.loc[best_model_idx, 'Accuracy']:.4f}")

def main():
    """Main function"""
    print("="*60)
    print("MACHINE LEARNING ANALYSIS - Email Spam Detection")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Spam distribution: {df['spam'].value_counts().to_dict()}")
    
    # Feature engineering
    X_tfidf, X_count, X_combined, y, tfidf_vectorizer, count_vectorizer = feature_engineering(df)
    
    # Initialize models
    models = {
        'Naive Bayes (TF-IDF)': MultinomialNB(),
        'SVM (TF-IDF)': SVC(probability=True, kernel='linear', random_state=42),
        'Random Forest (TF-IDF)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression (TF-IDF)': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'XGBoost (TF-IDF)': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    # Train and evaluate models
    results = []
    for model_name, model in models.items():
        result = train_and_evaluate_model(X_tfidf, y, model, model_name)
        results.append(result)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            result['confusion_matrix'], 
            model_name, 
            f"../../output/figures/cm_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        
        # Plot ROC curve
        plot_roc_curve(
            result['y_test'],
            result['y_pred_proba'],
            model_name,
            f"../../output/figures/roc_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
    
    # Compare models
    compare_models(results)
    
    # Save best model
    best_result = max(results, key=lambda x: x['f1'])
    import joblib
    joblib.dump(best_result['model'], f"../../models/best_model_{best_result['model_name'].replace(' ', '_')}.pkl")
    joblib.dump(tfidf_vectorizer, "../../models/tfidf_vectorizer.pkl")
    print(f"\nBest model saved: best_model_{best_result['model_name'].replace(' ', '_')}.pkl")
    
    print("\n" + "="*60)
    print("Machine Learning Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()


