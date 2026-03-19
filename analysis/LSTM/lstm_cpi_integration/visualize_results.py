import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent / 'results.json'
with open(results_path, 'r') as f:
    results = json.load(f)

# Organize results
full_period = {
    'model_a': {'rmse': results['full_0']['rmse'], 'mae': results['full_0']['mae']},
    'model_b': {'rmse': results['full_1']['rmse'], 'mae': results['full_1']['mae']},
    'model_c': {'rmse': results['full_2']['rmse'], 'mae': results['full_2']['mae']}
}

anomaly_period = {
    'model_a': {'rmse': results['anomaly_0']['rmse'], 'mae': results['anomaly_0']['mae']},
    'model_b': {'rmse': results['anomaly_1']['rmse'], 'mae': results['anomaly_1']['mae']},
    'model_c': {'rmse': results['anomaly_2']['rmse'], 'mae': results['anomaly_2']['mae']}
}

models = ['Model A\n(Spread)', 'Model B\n(Spread+MMF)', 'Model C\n(Spread+MMF+CPI)']
colors = ['#2ecc71', '#e74c3c', '#3498db']

# Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LSTM Model Comparison: Spread vs Liquidity vs CPI Integration', fontsize=16, fontweight='bold')

# 1. RMSE Comparison
ax = axes[0, 0]
x = np.arange(len(models))
width = 0.35

rmse_full = [full_period['model_a']['rmse'], full_period['model_b']['rmse'], full_period['model_c']['rmse']]
rmse_anom = [anomaly_period['model_a']['rmse'], anomaly_period['model_b']['rmse'], anomaly_period['model_c']['rmse']]

bars1 = ax.bar(x - width/2, rmse_full, width, label='Full Period', color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, rmse_anom, width, label='Anomaly Period (2024-11~2025-12)', 
               color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)

ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 2. MAE Comparison
ax = axes[0, 1]
mae_full = [full_period['model_a']['mae'], full_period['model_b']['mae'], full_period['model_c']['mae']]
mae_anom = [anomaly_period['model_a']['mae'], anomaly_period['model_b']['mae'], anomaly_period['model_c']['mae']]

bars1 = ax.bar(x - width/2, mae_full, width, label='Full Period', color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, mae_anom, width, label='Anomaly Period (2024-11~2025-12)',
               color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)

ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('MAE Comparison (Lower is Better)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 3. Error Change (Full → Anomaly)
ax = axes[1, 0]
rmse_increase = [(rmse_anom[i] - rmse_full[i]) / rmse_full[i] * 100 for i in range(3)]
mae_increase = [(mae_anom[i] - mae_full[i]) / mae_full[i] * 100 for i in range(3)]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x_pos - width/2, rmse_increase, width, label='RMSE Change', color='#e67e22', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, mae_increase, width, label='MAE Change', color='#9b59b6', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Change (%)', fontsize=11, fontweight='bold')
ax.set_title('Performance Degradation: Full Period → Anomaly Period', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# 4. Model Ranking by Period
ax = axes[1, 1]
ax.axis('off')

ranking_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL PERIOD (All Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥇 Model A (Spread): RMSE=0.1336, MAE=0.0965
🥈 Model C (Spread+MMF+CPI): RMSE=0.1579, MAE=0.1346
🥉 Model B (Spread+MMF): RMSE=0.2281, MAE=0.2027

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANOMALY PERIOD (2024-11~2025-12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥇 Model A (Spread): RMSE=0.1589, MAE=0.1203
🥈 Model B (Spread+MMF): RMSE=0.3358, MAE=0.3155
🥉 Model C (Spread+MMF+CPI): RMSE=0.3690, MAE=0.3496

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY FINDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adding MMF & CPI features DEGRADES prediction
performance across both time periods.

Interest rate spread alone is sufficient.
"""

ax.text(0.05, 0.95, ranking_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved model_comparison.png")

# Create detailed RMSE timeline visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Model Performance: Full Period vs Anomaly Period', fontsize=14, fontweight='bold')

# Full Period
ax = axes[0]
models_list = ['Model A\n(Spread)', 'Model B\n(Spread+MMF)', 'Model C\n(Spread+MMF+CPI)']
rmse_vals = [full_period['model_a']['rmse'], full_period['model_b']['rmse'], full_period['model_c']['rmse']]
mae_vals = [full_period['model_a']['mae'], full_period['model_b']['mae'], full_period['model_c']['mae']]

x_pos = np.arange(len(models_list))
width = 0.35

bars1 = ax.bar(x_pos - width/2, rmse_vals, width, label='RMSE', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, mae_vals, width, label='MAE', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Error Value', fontsize=11, fontweight='bold')
ax.set_title('Full Period (Train: 2025, Test: 507)', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_list)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Anomaly Period
ax = axes[1]
rmse_vals = [anomaly_period['model_a']['rmse'], anomaly_period['model_b']['rmse'], anomaly_period['model_c']['rmse']]
mae_vals = [anomaly_period['model_a']['mae'], anomaly_period['model_b']['mae'], anomaly_period['model_c']['mae']]

bars1 = ax.bar(x_pos - width/2, rmse_vals, width, label='RMSE', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, mae_vals, width, label='MAE', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Error Value', fontsize=11, fontweight='bold')
ax.set_title('Anomaly Period (2024-11~2025-12)\n(Train: 228, Test: 57)', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_list)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'period_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved period_comparison.png")

print("\n✓ All visualizations generated successfully")
