import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import matplotlib.pyplot as plt
import inspect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Helper to load config placeholder
def load_global_config():
    return {}

def compute_multi_output_metrics(y_true_df, y_pred):
    metrics = {}
    sig = inspect.signature(mean_squared_error)
    has_squared = ("squared" in sig.parameters)

    for i, col in enumerate(y_true_df.columns):
        y_t = y_true_df[col].values
        # Handle shape mismatch if y_pred is 1D vs 2D
        if y_pred.ndim > 1:
            y_p = y_pred[:, i]
        else:
            y_p = y_pred

        mae = mean_absolute_error(y_t, y_p)
        if has_squared:
            rmse = mean_squared_error(y_t, y_p, squared=False)
        else:
            rmse = mean_squared_error(y_t, y_p) ** 0.5
        r2 = r2_score(y_t, y_p)

        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics

def plot_per_target(y_true, y_pred, target, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k", s=10)
    
    # Perfect prediction line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    
    plt.title(f"{model_name}: {target}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_trend.png")
    plt.close()

def plot_residuals(y_true, y_pred, target, model_name, save_dir):
    """
    Plots residuals (Actual - Predicted) to check for bias.
    """
    save_dir = Path(save_dir)
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color='r', linestyle='--')
    plt.plot(residuals, alpha=0.7, lw=1)
    plt.title(f"{model_name} Residuals: {target}")
    plt.ylabel("Error (Actual - Pred)")
    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target}_residuals.png")
    plt.close()

def evaluate_models_for_granularity(root, granularity):
    models_dir = root / "results/regression" / granularity / "best_models"
    plots_dir = root / "results/regression" / granularity / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data (Features) to get Test Set
    # We need to replicate the cleaning/split logic to ensure X_test is identical
    feature_file = root / "data/features" / f"{granularity}_features.csv"
    if not feature_file.exists():
        print(f"Feature file not found: {feature_file}")
        return

    df = pd.read_csv(feature_file)
    
    # --- REPEAT CLEANING ---
    df.replace(-200, np.nan, inplace=True)
    cols_to_drop = [c for c in df.columns if "NMHC" in c]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    # -----------------------

    # Re-Split to get Test Set (2005)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        test_df = df[df["Date"].dt.year == 2005].copy()
    else:
        # Fallback split
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.1)
        test_df = df.iloc[train_size+val_size:]

    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols and c not in ["Date", "DateTime", "Time"]]

    X_test = test_df[feature_cols]
    y_test_df = test_df[target_cols]

    print(f"Evaluating models for {granularity} on {len(X_test)} samples...")

    results = []

    for model_path in models_dir.glob("*_best.joblib"):
        model_name = model_path.stem.replace("_best", "")
        print(f"   -> Model: {model_name}")

        try:
            model = load(model_path)
            y_pred = model.predict(X_test)
            
            metrics = compute_multi_output_metrics(y_test_df, y_pred)
            
            flat = {"Model": model_name}
            
            for target, m in metrics.items():
                print(f"      [{target}] RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")
                
                idx = list(y_test_df.columns).index(target)
                y_p_target = y_pred[:, idx]
                y_t_target = y_test_df[target].values
                
                # Plot Trend
                plot_per_target(y_t_target, y_p_target, target, model_name, plots_dir)
                
                # Plot Residuals (Great for report discussion!)
                plot_residuals(y_t_target, y_p_target, target, model_name, plots_dir)

                flat[f"{target}_MAE"] = m["MAE"]
                flat[f"{target}_RMSE"] = m["RMSE"]
                flat[f"{target}_R2"] = m["R2"]
            
            results.append(flat)

        except Exception as e:
            print(f"      ERROR evaluating {model_name}: {e}")

    if results:
        out_path = root / "results/regression" / granularity / "test_metrics.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Test metrics saved to {out_path}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    # Evaluate for both hourly and daily if they exist
    for granularity in ["hourly", "daily"]:
        evaluate_models_for_granularity(root, granularity)