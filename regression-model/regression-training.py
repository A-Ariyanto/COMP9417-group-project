import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# Import our updated model loader
from src.models.regression_models import get_regression_models

def load_global_config():
    """Helper to load config if needed (placeholder based on your file structure)"""
    return {}

def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    
    # R2 can be negative if model is terrible
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_for_feature_file(cfg, features_file):
    print(f"\nTraining MULTI-OUTPUT regression models using: {features_file}")

    root = Path(__file__).resolve().parents[2]
    
    # Load Data
    df = pd.read_csv(features_file)
    
    # --- CRITICAL DATA CLEANING FOR REPORT ---
    # 1. Replace sentinel value -200 with NaN
    print("   -> Replacing -200 with NaN...")
    df.replace(-200, np.nan, inplace=True)
    
    # 2. Drop NMHC(GT) completely (Unreliable data)
    # We drop columns containing "NMHC" to catch lags and targets too
    cols_to_drop = [c for c in df.columns if "NMHC" in c]
    if cols_to_drop:
        print(f"   -> Dropping unreliable NMHC columns: {len(cols_to_drop)} columns removed.")
        df.drop(columns=cols_to_drop, inplace=True)
    # -----------------------------------------

    # Detect Granularity
    prefix = Path(features_file).stem.replace("_features", "")
    granularity = prefix 
    print(f"   -> Detected granularity: {granularity}")

    # Split Data (Train 2004, Test 2005)
    # Assuming 'Date' or 'DateTime' column exists or index needs handling.
    # Based on your previous code, simple time split:
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        train_df = df[df["Date"].dt.year == 2004].copy()
        # For testing, we might want validation (last part of 2004) or test (2005)
        # Your report says Train=2004, Test=2005. 
        # But usually we need a validation set for Early Stopping in XGBoost.
        # Let's take 2005 as Test, and last 20% of 2004 as Val.
        
        test_df = df[df["Date"].dt.year == 2005].copy()
        
        # Sort just in case
        train_df = train_df.sort_values("Date")
        
        # Create a validation split from the end of training data (for XGBoost early stopping)
        val_size = int(len(train_df) * 0.15) 
        val_df = train_df.iloc[-val_size:]
        train_df = train_df.iloc[:-val_size]
        
    else:
        # Fallback if Date not found (using index)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.1)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size+val_size]
        test_df = df.iloc[train_size+val_size:]

    # Separate X and y
    # Assuming columns starting with "target_" are targets
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols and c not in ["Date", "DateTime", "Time"]]

    print(f"   -> Training samples: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    print(f"   -> Features: {len(feature_cols)}, Targets: {len(target_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    # Load Models
    model_dict = get_regression_models()
    
    results = []
    best_models = {}

    for name, model in model_dict.items():
        print(f"\n   Training {name}...")
        
        # Handling for XGBoost (needs eval_set for early stopping)
        if name == "XGBoost":
            # XGBoost sklearn API handles multi-output automatically BUT early_stopping
            # usually requires fitting one regressor at a time or specific support.
            # However, the MultiOutputRegressor wrapper is often needed for complex cases.
            # For simplicity in this script, we fit directly. XGBRegressor supports 2D y natively.
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            # Linear Regression & Random Forest (Pipelines)
            model.fit(X_train, y_train)

        # Evaluate on Validation Set
        val_preds = model.predict(X_val)
        val_metrics = evaluate_regression(y_val, val_preds)
        print(f"     -> Validation RMSE: {val_metrics['RMSE']:.4f}, R2: {val_metrics['R2']:.4f}")

        # Save model
        best_models[name] = model
        
        results.append({
            "Model": name,
            **{f"Val_{k}": v for k, v in val_metrics.items()},
        })

    # Save Results
    results_root = root / "results" / "regression"
    models_dir = results_root / granularity / "best_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in best_models.items():
        dump(model, models_dir / f"{name}_best.joblib")

    result_df = pd.DataFrame(results)
    result_df.to_csv(results_root / granularity / "regression_val_metrics.csv", index=False)
    
    print(f"\n   Done! Models saved to {models_dir}")
    return result_df

if __name__ == "__main__":
    cfg = load_global_config()
    # Update this path to point to your actual features file
    feature_dir = Path(__file__).resolve().parents[2] / "data" / "features"
    
    # Example usage loop
    for f_file in feature_dir.glob("*_features.csv"):
        train_for_feature_file(cfg, str(f_file))