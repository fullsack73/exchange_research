import pandas as pd

# Load the data
file_path = 'm2/KOR/M2_KOR_processed.csv'
df = pd.read_csv(file_path)

# Convert observation_date to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Define the periods
# User requested 2026-01-01, but data ends at 2025-12-01
anomaly_start_date = '2024-11-01'
original_anomaly_end_date = '2026-01-01'
available_anomaly_end_date = '2025-12-01'

previous_start_date = '2023-09-01'
previous_end_date = '2024-11-01'

def calculate_stats(df, start_date, end_date, period_name):
    # Filter for the specific dates
    start_row = df[df['observation_date'] == start_date]
    end_row = df[df['observation_date'] == end_date]
    
    if start_row.empty:
        print(f"[{period_name}] Data not found for start date {start_date}")
        return None
    if end_row.empty:
         print(f"[{period_name}] Data not found for end date {end_date}")
         return None

    start_val = start_row['M2_KOR'].values[0]
    end_val = end_row['M2_KOR'].values[0]
    
    increase = end_val - start_val
    percentage_increase = (increase / start_val) * 100
    
    # Calculate monthly growth rate for fair comparison if lengths differ
    months = (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) * 12 + \
             (pd.to_datetime(end_date).month - pd.to_datetime(start_date).month)
    
    monthly_avg_increase_rate = percentage_increase / months if months > 0 else 0

    return {
        'start_date': start_date,
        'end_date': end_date,
        'start_val': start_val,
        'end_val': end_val,
        'increase': increase,
        'percentage_increase': percentage_increase,
        'months': months,
        'monthly_avg_increase_rate': monthly_avg_increase_rate
    }

# Calculate for Anomaly Period using available end date
print(f"Drafting analysis. User requested end date: {original_anomaly_end_date}. Using available: {available_anomaly_end_date}")
anomaly_stats = calculate_stats(df, anomaly_start_date, available_anomaly_end_date, "Anomaly Period")

# Calculate for Previous Period
previous_stats = calculate_stats(df, previous_start_date, previous_end_date, "Previous Period")

print("-" * 30)
if anomaly_stats:
    print(f"Anomaly Period Analysis ({anomaly_stats['start_date']} to {anomaly_stats['end_date']}):")
    print(f"Start M2 Value: {anomaly_stats['start_val']:,.1f}")
    print(f"End M2 Value: {anomaly_stats['end_val']:,.1f}")
    print(f"Total Increase: {anomaly_stats['increase']:,.1f}")
    print(f"Percentage Increase Rate: {anomaly_stats['percentage_increase']:.2f}%")
    print(f"Duration: {anomaly_stats['months']} months")
    print(f"Monthly Avg Increase Rate: {anomaly_stats['monthly_avg_increase_rate']:.2f}%")

print("-" * 30)
if previous_stats:
    print(f"Previous Period Analysis ({previous_stats['start_date']} to {previous_stats['end_date']}):")
    print(f"Start M2 Value: {previous_stats['start_val']:,.1f}")
    print(f"End M2 Value: {previous_stats['end_val']:,.1f}")
    print(f"Total Increase: {previous_stats['increase']:,.1f}")
    print(f"Percentage Increase Rate: {previous_stats['percentage_increase']:.2f}%")
    print(f"Duration: {previous_stats['months']} months")
    print(f"Monthly Avg Increase Rate: {previous_stats['monthly_avg_increase_rate']:.2f}%")

print("-" * 30)

if anomaly_stats and previous_stats:
    # Compare monthly average rates since durations might differ
    diff_rate = anomaly_stats['monthly_avg_increase_rate'] - previous_stats['monthly_avg_increase_rate']
    print(f"Difference in Monthly Avg Growth Rate (Anomaly - Previous): {diff_rate:.2f}%")
    
    # Also compare total percentage if user cares about the "rate" in that sense, 
    # but monthly is fairer. Let's provide the direct comparison the user probably intended:
    # "growth rate" usually means the percentage increase over the period.
    print(f"Anomaly Growth: {anomaly_stats['percentage_increase']:.2f}% vs Previous Growth: {previous_stats['percentage_increase']:.2f}%")
    
    if anomaly_stats['percentage_increase'] > previous_stats['percentage_increase']:
         print("Conclusion: The total growth was higher in the anomaly period (despite being shorter/different).")
    else:
         print("Conclusion: The total growth was lower in the anomaly period.")
         
    if anomaly_stats['monthly_avg_increase_rate'] > previous_stats['monthly_avg_increase_rate']:
        print("Conclusion: The monthly average growth accelerated in the anomaly period.")
    else:
        print("Conclusion: The monthly average growth processed did not accelerate.")
