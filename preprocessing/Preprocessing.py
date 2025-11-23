import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Upload the file manually or place in /content
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', na_values=[-200, -200.0])

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Merge Date + Time making a Datetime index
# Explicitly specify the format to avoid parsing issues and drop rows where conversion fails
df['Date'] = df['Date'].astype(str)
df['Time'] = df['Time'].astype(str).str.replace('.', ':', regex=False)

df['timestamp'] = pd.to_datetime(df['Date'] + " " + df['Time'], dayfirst=True, errors='coerce')

# Remove invalid timestamps
df = df[~df['timestamp'].isna()].copy()
df = df.sort_values('timestamp').reset_index(drop=True)

# Fix Data Types (ensure numeric)
numeric_cols = df.columns.drop(['Date', 'Time', 'timestamp'])

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Rename CO(GT)
if 'CO(GT)' in df.columns:
    df = df.rename(columns={'CO(GT)': 'CO'})

# ENGINEERED FEATURE CREATION
df.to_csv('AirQuality_CLEAN.csv', index=False)
df_eng = df.copy()

# Time-based features
df_eng['hour'] = df_eng['timestamp'].dt.hour
df_eng['weekday'] = df_eng['timestamp'].dt.weekday
df_eng['month'] = df_eng['timestamp'].dt.month

# Lag features
for lag in [1, 3, 12, 24]:
    df_eng[f'CO_lag_{lag}'] = df_eng['CO'].shift(lag)

# Rolling means
for win in [3, 6, 12, 24]:
    df_eng[f'CO_roll_mean_{win}'] = (
        df_eng['CO'].rolling(window=win, min_periods=1).mean().shift(1)
    )

# Future targets for forecasting
for h in [1, 3, 12, 24]:
    df_eng[f'CO_fut_{h}h'] = df_eng['CO'].shift(-h)

# MODEL-READY DATASET (IMPUTED + SCALED)
df_eng.to_csv('AirQuality_Engineered.csv', index=False)
df_model = df_eng.copy()

# Predictor features
predictor_cols = [
    'hour', 'weekday', 'month',
    'T', 'RH', 'AH',
    'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
    'PT08.S4(NO2)', 'PT08.S5(O3)',
    'CO_lag_1', 'CO_lag_3', 'CO_lag_12', 'CO_lag_24',
    'CO_roll_mean_3', 'CO_roll_mean_6',
    'CO_roll_mean_12', 'CO_roll_mean_24'
]

# Keep only columns that exist (for safety)
predictor_cols = [c for c in predictor_cols if c in df_model.columns]

# Impute missing values
imputer = SimpleImputer(strategy='median')
df_model[predictor_cols] = imputer.fit_transform(df_model[predictor_cols])

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_model[predictor_cols])

# Add scaled feature columns
for i, col in enumerate(predictor_cols):
    df_model[f"{col}_scaled"] = scaled_features[:, i]

# 6. SAVE MODEL-READY DATASET
df_model.to_csv('AirQuality_ModelReady.csv', index=False)
print("Saved:", 'AirQuality_ModelReady.csv')
