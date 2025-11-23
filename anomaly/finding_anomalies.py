import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

Data_path = "../data/AirQuality_cleaned.csv"
target = "CO(GT)_fut_1h"
feature = [c for c in pd.read_csv(Data_path).columns if
           c.endswith("_scaled") and not c.startswith("CO(GT)_fut")]
datetime_column = "timestamp"

def load_data(data_path, feature_cols, target_col):
    """Loads the full data"""
    df = pd.read_csv(data_path, parse_dates=[datetime_column])
    df = df.sort_values(datetime_column)

    # Filter data for 2004 (Train) and 2005 (Test)
    df_train_raw = df[df[datetime_column].dt.year == 2004].copy()
    df_test_raw = df[df[datetime_column].dt.year == 2005].copy()

    df_train_raw = df_train_raw.dropna()
    df_test_raw = df_test_raw.dropna()

    # Define the minimal features needed for the residual model (lagged CO and scaled T, RH, AH)
    residual_features = [c for c in feature_cols if "CO(GT)_lag" in c]
    residual_features.extend(["T_scaled", "RH_scaled", "AH_scaled"])

    # Filter features to ensure they exist after the load
    residual_features = [f for f in residual_features if f in df_train_raw.columns]

    X_train = df_train_raw[[datetime_column] + residual_features].set_index(datetime_column)
    y_train = df_train_raw[[datetime_column, target_col]].set_index(datetime_column)

    X_test = df_test_raw[[datetime_column] + residual_features].set_index(datetime_column)
    y_test = df_test_raw[[datetime_column, target_col]].set_index(datetime_column)

    df_raw = df_test_raw.set_index(datetime_column)

    print(f"Loaded Train Set (2004, N={len(X_train)}).")
    print(f"Loaded Test Set (2005, N={len(X_test)}) for target: {target_col}.")

    return X_train, y_train, X_test, y_test, df_raw, residual_features


def model_XG(X_train, y_train, target_col):
    """Trains the forecasting model using XGBoost Regressor"""
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    model.fit(X_train, y_train[target_col])

    return model


def detect_residual_anomalies(model, X_test, y_test, target):
    """Calculates residuals and identifies anomalies based on a statistical threshold."""

    y_pred = model.predict(X_test)
    y_true = y_test[target].values

    residuals = y_true - y_pred

    threshold = np.percentile(np.abs(residuals), 99)
    residual_scores = np.abs(residuals)
    anomaly_idx = np.where(residual_scores >= threshold)[0]

    print(f"\n[Residual Detection] Threshold (99th percentile): {threshold:.4f}")
    print(f"[Residual Detection] Detected {len(anomaly_idx)} anomalies (1%).")

    return residuals, residual_scores, anomaly_idx, threshold

def detect_unsupervised_scores(X_test):
    """Applies Isolation Forest to the full feature space."""
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_test)

    unsupervised_scores = -iso.decision_function(X_test)
    iso_threshold = np.percentile(unsupervised_scores, 99)

    print(f"[Isolation Forest] Threshold (99th percentile): {iso_threshold:.4f}")

    return unsupervised_scores, iso_threshold

def interpret_anomalies(df_raw, anomaly_idx, target, save_dir):
    """Analyzes the detected anomalies against meteorological and calendar features."""

    df = df_raw.iloc[anomaly_idx].copy()

    print("\n--- Anomaly Interpretation ---")
    print(f"Analyzing {len(df)} residual anomalies for {target}...")

    target_unscaled = target.replace("_fut_1h", "").replace("_scaled", "")
    analysis_cols = [target_unscaled, "T", "RH", "AH", "is_weekend"]

    # Check if the needed columns exist in the raw test data
    analysis_cols_present = [c for c in analysis_cols if c in df.columns]

    print("\nTop 5 Anomalies and Corresponding Features:")
    print(df[analysis_cols_present].head())

    # Analysis of Causes
    avg_temp_anomaly = df['T'].mean()
    avg_temp_normal = df_raw['T'].mean()
    weekend_ratio = df['is_weekend'].mean()

    print(f"\nStatistics for Anomalous Time Periods (2005):")
    print(
        f"  > Average Temperature (T) during anomalies: {avg_temp_anomaly:.2f}°C (Normal 2005 Avg: {avg_temp_normal:.2f}°C)")
    print(f"  > Percentage occurring on Weekends: {weekend_ratio * 100:.1f}%")

    # Visualizing Anomalous Periods (Example: Temperature)
    plt.figure(figsize=(12, 5))
    plt.plot(df_raw.index, df_raw['T'], label='Temperature (T)', color='gray', alpha=0.6)
    plt.scatter(df.index, df['T'], color='red', s=50, label='Residual Anomaly Events', zorder=5)
    plt.title(f"Temperature (T) Profile During Anomalies in {target} (2005)")
    plt.xlabel(datetime_column)
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_T_anomalies.png")
    plt.close()

    print(f"\nInterpretation complete. See {target}_T_anomalies.png for visualization.")


def evaluate_precision_recall(residual_scores, unsupervised_scores, target, save_dir):
    """Evaluates the precision-recall trade-off."""

    iso_threshold = np.percentile(unsupervised_scores, 99)
    y_true_proxy = (unsupervised_scores > iso_threshold).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_true_proxy, residual_scores)

    auprc = auc(recall, precision)

    print("\n--- Evaluation: Precision-Recall Trade-Off ---")
    print("NOTE: Isolation Forest (top 1%) is used as the proxy for 'True Anomalies'.")
    print(f"AUPRC (Residual Scores): {auprc:.4f}")

    # Plotting the Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'Residual Scores (AUPRC={auprc:.4f})')

    # Plot baseline random classifier
    no_skill = len(y_true_proxy[y_true_proxy == 1]) / len(y_true_proxy)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (AUPRC={no_skill:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Anomaly Detection of {target}')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_pr_curve.png")
    plt.close()


if __name__ == "__main__":
    # Setup and Data Loading
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test, df_raw, residual_features = load_data(
        Data_path, feature, target
    )

    # Part 1: Residual-Based Anomaly Detection
    print("\n========================================================")
    print("STARTING RESIDUAL-BASED ANOMALY DETECTION (XGBOOST)")
    print("========================================================")

    # Train Model
    model = model_XG(X_train, y_train, target)

    # Detect Anomalies
    residuals, residual_scores, residual_idx, residual_threshold = detect_residual_anomalies(
        model, X_test, y_test, target
    )

    # Plotting residuals
    plt.figure(figsize=(14, 5))
    plt.plot(X_test.index, residuals, label='Residuals', color='blue', alpha=0.6)
    plt.scatter(X_test.index[residual_idx], residuals[residual_idx],
                color='red', label='Anomalies (Residual > Threshold)', s=10)
    plt.axhline(residual_threshold, color='red', linestyle='--', label='Threshold')
    plt.axhline(-residual_threshold, color='red', linestyle='--')
    plt.title(f"Residuals for {target} (XGBoost) with {len(residual_idx)} Anomalies Detected")
    plt.xlabel(datetime_column)
    plt.ylabel("Residual (True - Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{target}_residuals.png")
    plt.close()
    print(f"Residual visualization saved to {target}_residuals.png")

    # Part 2: Unsupervised Anomaly Detection (Isolation Forest)
    print("\n========================================================")
    print("STARTING UNSUPERVISED (ISOLATION FOREST) DETECTION")
    print("========================================================")

    unsupervised_scores, iso_threshold = detect_unsupervised_scores(X_test)

    # Plotting Unsupervised Scores
    iso_anomaly_idx = np.where(unsupervised_scores >= iso_threshold)[0]
    plt.figure(figsize=(14, 5))
    plt.plot(X_test.index, unsupervised_scores, label='Isolation Forest Score', color='green', alpha=0.6)
    plt.scatter(X_test.index[iso_anomaly_idx], unsupervised_scores[iso_anomaly_idx],
                color='darkred', label='Anomalies (Score > Threshold)', s=10)
    plt.axhline(iso_threshold, color='darkred', linestyle='--', label='Threshold')
    plt.title(f"Isolation Forest Anomaly Scores (N={len(iso_anomaly_idx)} Anomalies) on 2005 data")
    plt.xlabel(datetime_column)
    plt.ylabel("Anomaly Score (Higher is more anomalous)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "isolation_forest_scores.png")
    plt.close()
    print(f"Isolation Forest visualization saved to isolation_forest_scores.png")

    # Part 3: Analysis and Interpretation
    print("\n========================================================")
    print("STARTING ANOMALY INTERPRETATION")
    print("========================================================")

    # Use the residual anomalies for interpretation
    interpret_anomalies(df_raw, residual_idx, target, save_dir)

    # --- Step 4: Evaluation (Precision-Recall) ---
    print("\n========================================================")
    print("STARTING PRECISION-RECALL EVALUATION")
    print("========================================================")

    # Evaluate Residual Detection using Isolation Forest results as proxy
    evaluate_precision_recall(residual_scores, unsupervised_scores, target, save_dir)

    print(f"Precision-Recall curve saved to {target}_pr_curve.png")
    print("\nAnomaly detection pipeline completed successfully.")
