"""
=============================================================
PSLP: Customer Demand Forecasting Model
ML Engine – Preprocessing, EDA, Modelling, Evaluation
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load CSV, clean, engineer features."""
    df = pd.read_csv(path)

    # ── Missing value imputation ──────────────────────────────
    for col in ["Price", "Advertising_Spend"]:
        df[col] = df[col].fillna(df[col].median())
    df["Discount"] = df["Discount"].fillna(0)

    # ── Date feature engineering ──────────────────────────────
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["Quarter"]    = df["Date"].dt.quarter
    df["Year"]       = df["Date"].dt.year

    # ── Encode Season ─────────────────────────────────────────
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    df["Season_Code"] = df["Season"].map(season_map)

    # ── Effective Price after Discount ────────────────────────
    df["Effective_Price"] = df["Price"] * (1 - df["Discount"] / 100)

    # ── Log-transform advertising (right-skewed) ──────────────
    df["Log_Adspend"] = np.log1p(df["Advertising_Spend"])

    return df


# ─────────────────────────────────────────────────────────────
# 2. EDA STATISTICS
# ─────────────────────────────────────────────────────────────

def eda_stats(df: pd.DataFrame) -> dict:
    """Return key EDA metrics."""
    numeric = df[["Price", "Discount", "Advertising_Spend", "Sales"]].describe().round(2)

    corr_cols = ["Price", "Discount", "Advertising_Spend",
                 "Holiday", "Weekend", "Month", "Sales"]
    corr = df[corr_cols].corr()["Sales"].drop("Sales").round(3).to_dict()

    seasonal_sales = df.groupby("Season")["Sales"].mean().round(1).to_dict()

    monthly_sales  = (
        df.groupby("Month")["Sales"].mean().round(1)
        .rename(index={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        .to_dict()
    )

    holiday_avg = {
        "Holiday":    round(df[df["Holiday"]==1]["Sales"].mean(), 2),
        "Non-Holiday": round(df[df["Holiday"]==0]["Sales"].mean(), 2),
    }

    return {
        "describe":      numeric.to_dict(),
        "correlations":  corr,
        "seasonal_avg":  seasonal_sales,
        "monthly_avg":   monthly_sales,
        "holiday_avg":   holiday_avg,
        "total_records": len(df),
        "missing_before": {"Price": 60, "Advertising_Spend": 60, "Discount": 60},
    }


# ─────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────

def run_simple_linear(df: pd.DataFrame) -> dict:
    """Linear Regression: Sales ~ Advertising_Spend."""
    X = df[["Advertising_Spend"]].values
    y = df["Sales"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    slope     = round(float(model.coef_[0]), 4)
    intercept = round(float(model.intercept_), 4)
    r2        = round(r2_score(y_test, y_pred), 4)
    mae       = round(mean_absolute_error(y_test, y_pred), 2)
    mse       = round(mean_squared_error(y_test, y_pred), 2)
    mean_err  = round(float(np.mean(y_test - y_pred)), 4)

    # scatter data (sample 200 pts for speed)
    idx = np.random.choice(len(X_test), min(200, len(X_test)), replace=False)
    x_line = np.linspace(X_test.min(), X_test.max(), 100)

    return {
        "equation":   f"Sales = {slope} × Advertising_Spend + {intercept}",
        "slope":      slope,
        "intercept":  intercept,
        "r2":         r2,
        "mae":        mae,
        "mse":        mse,
        "rmse":       round(np.sqrt(mse), 2),
        "mean_error": mean_err,
        "scatter_x":  X_test[idx, 0].tolist(),
        "scatter_y":  y_test[idx].tolist(),
        "line_x":     x_line.tolist(),
        "line_y":     (model.predict(x_line.reshape(-1,1))).tolist(),
        "actual":     y_test[:100].tolist(),
        "predicted":  y_pred[:100].tolist(),
    }


def run_multiple_linear(df: pd.DataFrame) -> dict:
    """Multiple Linear Regression with feature scaling."""
    features = ["Price", "Discount", "Advertising_Spend",
                "Holiday", "Weekend", "Month", "Season_Code",
                "Effective_Price", "Log_Adspend"]

    X = df[features].values
    y = df["Sales"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    coefficients = {f: round(c, 4) for f, c in zip(features, model.coef_)}
    r2            = round(r2_score(y_test, y_pred), 4)
    mae           = round(mean_absolute_error(y_test, y_pred), 2)
    mse           = round(mean_squared_error(y_test, y_pred), 2)
    mean_err      = round(float(np.mean(y_test - y_pred)), 4)

    # Actual vs Predicted (100 pts)
    idx = np.arange(min(100, len(y_test)))

    return {
        "features":     features,
        "coefficients": coefficients,
        "intercept":    round(float(model.intercept_), 4),
        "r2":           r2,
        "adj_r2":       round(1-(1-r2)*(len(y_train)-1)/(len(y_train)-len(features)-1), 4),
        "mae":          mae,
        "mse":          mse,
        "rmse":         round(np.sqrt(mse), 2),
        "mean_error":   mean_err,
        "actual":       y_test[idx].tolist(),
        "predicted":    y_pred[idx].tolist(),
        "residuals":    (y_test[:200] - y_pred[:200]).tolist(),
    }


# ─────────────────────────────────────────────────────────────
# 4. DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────

def distribution_analysis(df: pd.DataFrame) -> dict:
    """Normal & Poisson distribution stats for Sales."""
    sales = df["Sales"].values

    # Normal
    mu, sigma = float(np.mean(sales)), float(np.std(sales))
    stat, pvalue = stats.normaltest(sales)
    x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)

    # Poisson
    lam = float(np.mean(sales))
    k   = np.arange(max(0, int(lam)-100), int(lam)+100)
    y_pois = stats.poisson.pmf(k, lam)

    # Histogram of actual sales
    hist, bin_edges = np.histogram(sales, bins=30)

    return {
        "normal": {
            "mean":    round(mu, 2),
            "std":     round(sigma, 2),
            "p_value": round(float(pvalue), 6),
            "is_normal": bool(pvalue > 0.05),
            "x":       x_norm.tolist(),
            "y":       y_norm.tolist(),
        },
        "poisson": {
            "lambda": round(lam, 2),
            "k":      k.tolist(),
            "pmf":    y_pois.tolist(),
        },
        "histogram": {
            "counts":      hist.tolist(),
            "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────
# 5. PREDICTION
# ─────────────────────────────────────────────────────────────

_model_cache: dict = {}

def train_prediction_model(df: pd.DataFrame) -> None:
    """Train & cache MLR model for live prediction."""
    features = ["Price", "Discount", "Advertising_Spend",
                "Holiday", "Weekend", "Month", "Season_Code",
                "Effective_Price", "Log_Adspend"]

    df_clean = df.dropna(subset=features + ["Sales"])
    X = df_clean[features].values
    y = df_clean["Sales"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_sc, y)

    _model_cache["model"]    = model
    _model_cache["scaler"]   = scaler
    _model_cache["features"] = features


def predict_demand(inputs: dict) -> dict:
    """Predict sales for given input features."""
    if "model" not in _model_cache:
        raise RuntimeError("Model not trained yet.")

    model    = _model_cache["model"]
    scaler   = _model_cache["scaler"]
    features = _model_cache["features"]

    price       = float(inputs.get("price", 100))
    discount    = float(inputs.get("discount", 0))
    ad_spend    = float(inputs.get("advertising_spend", 500))
    holiday     = int(inputs.get("holiday", 0))
    weekend     = int(inputs.get("weekend", 0))
    month       = int(inputs.get("month", 6))
    season_code = int(inputs.get("season_code", 1))

    eff_price   = price * (1 - discount / 100)
    log_adspend = np.log1p(ad_spend)

    row = np.array([[price, discount, ad_spend, holiday,
                     weekend, month, season_code, eff_price, log_adspend]])
    row_sc = scaler.transform(row)
    pred   = float(model.predict(row_sc)[0])
    pred   = max(0, round(pred, 0))

    # Confidence interval (±1.96 * residual std ≈ rough 95% CI)
    std_approx = 28.0
    lo = max(0, pred - 1.96 * std_approx)
    hi = pred + 1.96 * std_approx

    return {
        "predicted_sales":  int(pred),
        "confidence_lower": int(lo),
        "confidence_upper": int(hi),
        "inputs_echo": {
            "price":             price,
            "discount":          discount,
            "advertising_spend": ad_spend,
            "holiday":           holiday,
            "weekend":           weekend,
            "month":             month,
            "season":            ["Winter","Spring","Summer","Autumn"][season_code],
        }
    }
