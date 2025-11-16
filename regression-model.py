import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- 1. Data Loading and Initial Cleaning ---
data_file = 'AirQualityUCI.csv'

# Load data, parse dates, and set -200 as NaN
df = pd.read_csv(
    data_file, 
    delimiter=';',
    decimal=',',
    parse_dates={'datetime': ['Date', 'Time']},
    na_values=-200
)

df.set_index('datetime', inplace=True)
df.dropna(axis=1, how='all', inplace=True)


# --- 2. Preprocessing & Feature Engineering ---

target_pollutant = 'CO(GT)'

# [cite_start]Handle Missing Values (as required by brief) [cite: 28]
df_processed = df.interpolate(method='time')
df_processed.fillna(method='bfill', inplace=True) # Fill any remaining NaNs at the start

# [cite_start]Create derived time features [cite: 29]
df_processed['hour'] = df_processed.index.hour
df_processed['weekday'] = df_processed.index.weekday
df_processed['month'] = df_processed.index.month

# [cite_start]Create temporal (lagged) features [cite: 33]
# For 1-hour-ahead forecast, we use data from 1 hour ago (t-1)
df_processed[f'{target_pollutant}_lag1'] = df_processed[target_pollutant].shift(1)

# [cite_start]Create moving average features [cite: 33]
df_processed[f'{target_pollutant}_roll_avg_6h'] = df_processed[target_pollutant].shift(1).rolling(window=6).mean()

# Drop rows with NaNs created by lags/rolling
df_processed.dropna(inplace=True)


# --- 3. Define Features (X) and Target (y) ---

# Target 'y' is the CO concentration
y = df_processed[target_pollutant]

# Features 'X' are all other columns
# We must drop the *other* pollutant ground truth (GT) values
gt_columns_to_drop = ['NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
X = df_processed.drop(columns=[target_pollutant] + gt_columns_to_drop)


# --- 4. Temporal Data Splitting ---

# [cite_start]Per the brief: Train on 2004, test on 2005 [cite: 36]
X_train = X['2004']
y_train = y['2004']
X_test = X['2005']
y_test = y['2005']


# --- 5. Feature Scaling ---

# Identify columns to scale (all sensor and meteorological data)
columns_to_scale = [
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 
    'PT08.S5(03)', 'T', 'RH', 'AH',
    f'{target_pollutant}_lag1', f'{target_pollutant}_roll_avg_6h'
]
# Ensure we only try to scale columns that exist
columns_to_scale = [col for col in columns_to_scale if col in X_train.columns]

scaler = StandardScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])


# --- 6. Model Development and Assessment ---

print("\n--- Model Training and Evaluation (1-Hour Forecast) ---")

# [cite_start]Naïve Baseline Model [cite: 41]
# The prediction is the value from 1 hour ago (t-1)
# We pull the *unscaled* value from the original 'X' dataframe for a fair comparison
y_pred_naive = X['2005'][f'{target_pollutant}_lag1']

# [cite_start]Model 1: Linear Regression [cite: 42]
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# [cite_start]Model 2: Random Forest Regressor [cite: 42]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# [cite_start]--- 7. Model Assessment (RMSE) --- [cite: 39]

rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\n--- Results for {target_pollutant} (1-hour forecast) ---")
print(f"Naïve Baseline RMSE: {rmse_naive:.4f}")
print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}")

# --- 8. Final Plot ---
# This plot is useful for your report

# Plot predictions vs. observed values for a sample period
plot_slice = slice(100, 300) # Plot 200 hours from the test set

plt.figure(figsize=(15, 6))
plt.plot(y_test.values[plot_slice], label='Observed (Actual)', color='blue', alpha=0.7)
plt.plot(y_pred_rf[plot_slice], label=f'Random Forest (RMSE: {rmse_rf:.4f})', color='green', alpha=0.7)
plt.plot(y_pred_lr[plot_slice], label=f'Linear Regression (RMSE: {rmse_lr:.4f})', color='orange', alpha=0.7)
plt.plot(y_pred_naive.values[plot_slice], label=f'Naïve Baseline (RMSE: {rmse_naive:.4f})', color='red', linestyle='--')
plt.title(f'1-Hour Ahead Forecast for {target_pollutant} (Test Set Sample)')
plt.xlabel('Time (Hours into sample)')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('model_comparison_forecast.png')
plt.show()

print("\n--- Script Finished ---")