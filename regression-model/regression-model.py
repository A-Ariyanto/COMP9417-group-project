import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def get_linear_regression():
    """
    Returns a Pipeline: Imputer (Median) -> Linear Regression.
    Linear Regression cannot handle NaNs, so we must impute.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', LinearRegression())
    ])

def get_random_forest_regressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    verbose=False,
):
    """
    Returns a Pipeline: Imputer (Median) -> Random Forest.
    Random Forest cannot handle NaNs, so we must impute.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        ))
    ])

def get_xgboost_regressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
):
    """
    Returns an XGBRegressor.
    XGBoost HANDLES NaNs NATIVELY. We do NOT impute here.
    This allows the model to learn from the 'missingness' pattern.
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        tree_method="auto", # Uses GPU if available, else CPU
        missing=np.nan      # Explicitly tell XGBoost that np.nan represents missing values
    )

def get_regression_models():
    """
    Returns the dictionary of the 3 selected models for the report.
    1. Linear Regression (Baseline)
    2. Random Forest (Bagging Benchmark)
    3. XGBoost (Boosting Challenger / Missing Data Specialist)
    """
    models = {
        "LinearRegression": get_linear_regression(),
        "RandomForestRegressor": get_random_forest_regressor(verbose=0),
        "XGBoost": get_xgboost_regressor(),
    }
    return models