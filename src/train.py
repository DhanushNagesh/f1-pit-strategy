import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
from pathlib import Path
import json

from features import engineer_features, get_feature_columns

def train_baseline_models(data_path: str = 'data/processed/race_data.csv'):
    """Train baseline models and save the best one."""
    
    # Load and prepare data
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    X = df[get_feature_columns()]
    y = df['target']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive class ratio: {y_train.mean():.3f}")
    
    # Train Random Forest
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    
    print(f"F1: {rf_f1:.3f}")
    print(f"Precision: {rf_precision:.3f}")
    print(f"Recall: {rf_recall:.3f}")
    print(classification_report(y_test, y_pred_rf))
    
    # Train XGBoost
    print("\n=== XGBoost ===")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    
    y_pred_xgb = xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)
    
    print(f"F1: {xgb_f1:.3f}")
    print(f"Precision: {xgb_precision:.3f}")
    print(f"Recall: {xgb_recall:.3f}")
    print(classification_report(y_test, y_pred_xgb))
    
    # Save best model
    best_model = xgb if xgb_f1 > rf_f1 else rf
    best_name = "XGBoost" if xgb_f1 > rf_f1 else "RandomForest"
    best_f1 = max(xgb_f1, rf_f1)
    
    Path('models').mkdir(exist_ok=True)
    joblib.dump(best_model, 'models/baseline_model.pkl')
    
    # Save metrics
    metrics = {
        'model_type': best_name,
        'f1_score': float(best_f1),
        'precision': float(xgb_precision if best_name == "XGBoost" else rf_precision),
        'recall': float(xgb_recall if best_name == "XGBoost" else rf_recall),
        'random_forest_f1': float(rf_f1),
        'xgboost_f1': float(xgb_f1)
    }
    
    with open('models/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved {best_name} model (F1: {best_f1:.3f}) to models/baseline_model.pkl")
    
    return best_model, metrics

if __name__ == "__main__":
    train_baseline_models()