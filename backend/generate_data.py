"""
=============================================================
PSLP: Customer Demand Forecasting Model
Data Generation Module - Walmart-style Sales Dataset
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_walmart_dataset(n_records=2000, seed=42):
    """
    Generate a realistic Walmart-style sales dataset.
    Simulates real-world patterns including seasonality, promotions, and trends.
    """
    np.random.seed(seed)

    # ── Date Range ──────────────────────────────────────────
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i % 365 + (i // 365) * 365) for i in range(n_records)]

    # ── Store & Product IDs ──────────────────────────────────
    store_ids   = np.random.choice([f"S{i:02d}" for i in range(1, 11)], n_records)
    product_ids = np.random.choice([f"P{i:03d}" for i in range(1, 51)], n_records)

    # ── Base Price (product-dependent, 5–500) ────────────────
    product_prices = {f"P{i:03d}": np.random.uniform(5, 500) for i in range(1, 51)}
    base_price = np.array([product_prices[p] for p in product_ids])
    price_noise = np.random.normal(0, base_price * 0.05)
    price = np.clip(base_price + price_noise, 1, 600).round(2)

    # ── Discount (0–40%) ────────────────────────────────────
    discount = np.random.choice([0, 5, 10, 15, 20, 25, 30, 40],
                                 n_records,
                                 p=[0.3, 0.15, 0.15, 0.15, 0.1, 0.07, 0.05, 0.03])

    # ── Advertising Spend (₹ / $) ───────────────────────────
    advertising_spend = np.random.exponential(scale=500, size=n_records).round(2)

    # ── Seasons ─────────────────────────────────────────────
    month_map = {d.month for d in dates}
    def get_season(m):
        if m in [12, 1, 2]: return "Winter"
        if m in [3, 4, 5]:  return "Spring"
        if m in [6, 7, 8]:  return "Summer"
        return "Autumn"
    season = np.array([get_season(d.month) for d in dates])

    # ── Holiday Flag ─────────────────────────────────────────
    holidays = {(1,1),(1,26),(3,10),(8,15),(10,2),(10,12),(11,3),(12,25),(12,31)}
    holiday = np.array([1 if (d.month, d.day) in holidays else 0 for d in dates])

    # ── Weekend Flag ─────────────────────────────────────────
    weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])

    # ── Target: Sales (realistic formula with noise) ─────────
    season_multiplier = {"Winter": 1.3, "Spring": 1.0, "Summer": 0.85, "Autumn": 1.1}
    s_mult = np.array([season_multiplier[s] for s in season])

    # Sales = f(price, discount, advertising, seasonality, holidays)
    base_sales = (
        200
        - 0.15  * price
        + 2.5   * discount
        + 0.08  * advertising_spend
        + 50    * s_mult
        + 40    * holiday
        + 25    * weekend
        + np.random.normal(0, 30, n_records)
    )
    sales = np.clip(base_sales, 5, 2000).round(0).astype(int)

    # ── Assemble DataFrame ───────────────────────────────────
    df = pd.DataFrame({
        "Date":              [d.strftime("%Y-%m-%d") for d in dates],
        "Store_ID":          store_ids,
        "Product_ID":        product_ids,
        "Price":             price,
        "Discount":          discount,
        "Advertising_Spend": advertising_spend,
        "Season":            season,
        "Holiday":           holiday,
        "Weekend":           weekend,
        "Sales":             sales,
    })

    # ── Introduce ~3% missing values (realistic) ─────────────
    for col in ["Price", "Advertising_Spend", "Discount"]:
        mask = np.random.random(n_records) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_walmart_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/walmart_sales.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())
