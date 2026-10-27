"""Train and evaluate regression pipelines."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

def evaluate_models(df: pd.DataFrame, include_previous_grades: bool = False, random_state: int = 42):
    """Compare baseline mean, Linear Regression, Ridge, and Random Forest."""
    from src.preprocessing import select_features
    X, y = select_features(df, include_previous_grades=include_previous_grades)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    preprocessor = build_preprocessor(X)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state, min_samples_leaf=2)
    }
    rows, fitted = [], {}
    baseline_prediction = np.repeat(y_train.mean(), len(y_test))
    rows.append({
        "model": "MeanBaseline", "mae": mean_absolute_error(y_test, baseline_prediction),
        "rmse": mean_squared_error(y_test, baseline_prediction) ** 0.5,
        "r2": r2_score(y_test, baseline_prediction)
    })
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "r2": r2_score(y_test, pred)
        })
        fitted[name] = (pipe, pred, y_test, X_test)
    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_name = metrics.loc[metrics["model"] != "MeanBaseline"].sort_values("mae").iloc[0]["model"]
    return metrics, fitted, best_name, (X_train, X_test, y_train, y_test)

def save_model_outputs(metrics: pd.DataFrame, fitted: dict, best_name: str, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "regression_metrics.csv", index=False)

    pipe, pred, y_test, X_test = fitted[best_name]
    comparison = pd.DataFrame({
        "actual_G3": y_test.to_numpy(),
        "predicted_G3": pred,
        "residual": y_test.to_numpy() - pred
    })
    comparison.to_csv(output_dir / "predictions.csv", index=False)
    return comparison
