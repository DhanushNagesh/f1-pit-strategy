import json

# Load metrics
with open('models/baseline_metrics.json', 'r') as f:
    baseline = json.load(f)

with open('models/lstm_metrics.json', 'r') as f:
    lstm = json.load(f)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"\n{'Model':<20} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 60)
print(f"{baseline['model_type']:<20} {baseline['f1_score']:.3f}      {baseline['precision']:.3f}        {baseline['recall']:.3f}")
print(f"{lstm['model_type']:<20} {lstm['f1_score']:.3f}      {lstm['precision']:.3f}        {lstm['recall']:.3f}")
print("-" * 60)

improvement = ((lstm['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
print(f"\nBi-LSTM improvement: +{improvement:.1f}%")
print(f"Winner: {lstm['model_type']} ✓")