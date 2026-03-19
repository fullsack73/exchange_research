import json
import numpy as np

# Load results
with open('analysis/lstm_cpi_integration/results.json') as f:
    results = json.load(f)

print("=" * 80)
print("NORMALIZED RMSE (what saved in results.json)")
print("=" * 80)
print("\n📊 Full Period:")
for key in ['full_0', 'full_1', 'full_2']:
    r = results[key]
    print(f"  {r['model_name']}: RMSE={r['rmse']:.6f}, MAE={r['mae']:.6f}")

print("\n📊 Anomaly Period (2024-11 ~ 2026-03):")
for key in ['anomaly_0', 'anomaly_1', 'anomaly_2']:
    r = results[key]
    print(f"  {r['model_name']}: RMSE={r['rmse']:.6f}, MAE={r['mae']:.6f}")

print("\n" + "=" * 80)
print("COMPARING WITH lstm_validation_daily (ORIGINAL SCALE)")
print("=" * 80)
print("\n📊 lstm_validation_daily - Anomaly Period:")
print("  Model A (Spread only): RMSE=21.059, MAE=17.971")
print("  Model B (Spread + MMF): RMSE=18.771, MAE=15.944")
print("  → Model B is BETTER ✓")

print("\n📊 lstm_cpi_integration - Anomaly Period (after fix):")
print("  Model A (Spread only): RMSE=0.1294, MAE=0.1290")
print("  Model B (Spread + MMF): RMSE=0.2231, MAE=0.2230")
print("  → Model A is BETTER (but this is NORMALIZED scale)")

print("\n⚠️ ISSUE: Comparing across DIFFERENT scalers")
print("  lstm_validation_daily used scaler fitted on full-period data")
print("  lstm_cpi_integration uses different data source (includes CPI)")
print("  → Cannot directly compare normalized RMSE values")

# Check actual y values
print("\n" + "=" * 80)
print("ACTUAL VALUES (from y_test in results)")
print("=" * 80)

print("\n📊 lstm_cpi_integration - Anomaly Period actual y_test sample:")
y_test_a = results['anomaly_0']['y_test'][:5]
y_test_b = results['anomaly_1']['y_test'][:5]
print(f"  Model A y_test (first 5): {[f'{v:.2f}' for v in y_test_a]}")
print(f"  Model B y_test (first 5): {[f'{v:.2f}' for v in y_test_b]}")

y_pred_a = results['anomaly_0']['y_pred'][:5]
y_pred_b = results['anomaly_1']['y_pred'][:5]
print(f"  Model A y_pred (first 5): {[f'{v:.2f}' for v in y_pred_a]}")
print(f"  Model B y_pred (first 5): {[f'{v:.2f}' for v in y_pred_b]}")

# Compute RMSEs manually
y_test_a = np.array(results['anomaly_0']['y_test'])
y_pred_a = np.array(results['anomaly_0']['y_pred'])
rmse_a = np.sqrt(np.mean((y_test_a - y_pred_a) ** 2))

y_test_b = np.array(results['anomaly_1']['y_test'])
y_pred_b = np.array(results['anomaly_1']['y_pred'])
rmse_b = np.sqrt(np.mean((y_test_b - y_pred_b) ** 2))

print(f"\n  RMSE recalculated from y_test/y_pred:")
print(f"  Model A: {rmse_a:.6f}")
print(f"  Model B: {rmse_b:.6f}")
print(f"\n  Ratio (B/A): {rmse_b/rmse_a:.3f}x")
