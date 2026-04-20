import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from features import engineer_features, get_feature_columns

def explain_model(model_path='models/baseline_model.pkl', 
                  data_path='data/processed/race_data.csv',
                  output_dir='models/shap_plots'):
    """
    Generate SHAP explanations for the baseline model.
    Shows which features drive pit stop predictions.
    """
    
    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df['target']
    
    # Take a sample for SHAP (it's computationally expensive on large datasets)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    print(f"Computing SHAP values for {sample_size} samples...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Extract values (new SHAP API returns Explanation object)
    if hasattr(shap_values, 'values'):
        shap_vals = shap_values.values
    else:
        shap_vals = shap_values
    
    # If binary classification returns 3D array, take positive class
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Summary plot - shows feature importance
    print("\nGenerating summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot - mean absolute SHAP values
    print("Generating feature importance plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Compute mean absolute SHAP values for ranking
    mean_shap = np.abs(shap_vals).mean(axis=0)
    
    # Ensure mean_shap is 1D
    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap.flatten()
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (by mean |SHAP value|)")
    print("="*60)
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']:<30} {row['mean_abs_shap']:.4f}")
    
    # Save to file
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    # 4. Dependence plot for top feature
    top_feature = feature_importance.iloc[0]['feature']
    print(f"\nGenerating dependence plot for top feature: {top_feature}")
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, shap_vals, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dependence_plot_{top_feature}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ SHAP plots saved to {output_dir}/")
    
    return feature_importance


if __name__ == "__main__":
    importance = explain_model()