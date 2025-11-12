"""
Machine Learning Analysis Script
Consumer Purchase Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # Prepare features and target
    X = df[['Gender', 'Age', 'EstimatedSalary']]
    y = df['Purchased']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Logistic Regression' or name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation score
        if name == 'Logistic Regression' or name == 'SVM':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results

def plot_results(results, y_test, output_dir='../../output'):
    """Plot model comparison and evaluation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Model comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()],
        'F1 Score': [results[m]['f1'] for m in results.keys()],
        'CV Accuracy': [results[m]['cv_mean'] for m in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.15
    
    ax.bar(x - 2*width, comparison_df['Accuracy'], width, label='Accuracy')
    ax.bar(x - width, comparison_df['Precision'], width, label='Precision')
    ax.bar(x, comparison_df['Recall'], width, label='Recall')
    ax.bar(x + width, comparison_df['F1 Score'], width, label='F1 Score')
    ax.bar(x + 2*width, comparison_df['CV Accuracy'], width, label='CV Accuracy')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("MACHINE LEARNING ANALYSIS")
    print("="*50)
    
    # Load and preprocess data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, le = \
        load_and_preprocess_data('../../data/Advertisement.csv')
    
    # Train and evaluate models
    results = train_and_evaluate_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Plot results
    plot_results(results, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]
    
    print(f"\nBEST MODEL: {best_model_name}")
    print("="*50)
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1 Score: {best_model['f1']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, best_model['y_pred']))
    
    # Save best model
    os.makedirs('../../models', exist_ok=True)
    with open('../../models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model['model'], f)
    print(f"\nBest model saved to models/best_model.pkl")
    
    print("\n" + "="*50)
    print("ML ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()

