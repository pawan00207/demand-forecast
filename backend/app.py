"""
=============================================================
PSLP: Customer Demand Forecasting Model
Flask REST API Backend
=============================================================
Run: python app.py
=============================================================
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import sqlite3

from generate_data import generate_walmart_dataset
from ml_engine import (
    load_and_preprocess,
    eda_stats,
    run_simple_linear,
    run_multiple_linear,
    distribution_analysis,
    train_prediction_model,
    predict_demand,
)

# ─────────────────────────────────────────────────────────────
app  = Flask(__name__)
CORS(app)

import tempfile
DATA_DIR = os.path.join(tempfile.gettempdir(), "demand_data")
CSV_PATH = os.path.join(DATA_DIR, "walmart_sales.csv")
DB_PATH  = os.path.join(DATA_DIR, "forecast.db")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Bootstrap: generate data + db on first run ────────────────
def bootstrap():
    if not os.path.exists(CSV_PATH):
        df = generate_walmart_dataset(n_records=2000)
        df.to_csv(CSV_PATH, index=False)

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    DEFAULT (datetime('now')),
            price       REAL,
            discount    REAL,
            ad_spend    REAL,
            holiday     INTEGER,
            weekend     INTEGER,
            month       INTEGER,
            season      TEXT,
            predicted   INTEGER,
            ci_lower    INTEGER,
            ci_upper    INTEGER
        )
    """)
    conn.commit(); conn.close()

    # Pre-train model
    df = pd.read_csv(CSV_PATH)
    df = load_and_preprocess(CSV_PATH)
    train_prediction_model(df)
    print("[OK] Bootstrap complete - model trained.")

bootstrap()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_df():
    return load_and_preprocess(CSV_PATH)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Demand Forecast API running"})


@app.route("/api/dataset/info")
def dataset_info():
    df = get_df()
    sample = pd.read_csv(CSV_PATH).head(10).to_dict(orient="records")
    return jsonify({
        "shape":   {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "dtypes":  df.dtypes.astype(str).to_dict(),
        "sample":  sample,
    })


@app.route("/api/eda")
def eda():
    df = get_df()
    return jsonify(eda_stats(df))


@app.route("/api/eda/scatter")
def eda_scatter():
    """Return sampled scatter data for EDA charts."""
    df = get_df().sample(min(500, len(get_df())), random_state=1)
    return jsonify({
        "price_vs_sales": {
            "x": df["Price"].tolist(),
            "y": df["Sales"].tolist(),
        },
        "adspend_vs_sales": {
            "x": df["Advertising_Spend"].tolist(),
            "y": df["Sales"].tolist(),
        },
        "discount_vs_sales": {
            "x": df["Discount"].tolist(),
            "y": df["Sales"].tolist(),
        },
    })


@app.route("/api/model/simple-linear")
def simple_linear():
    df = get_df()
    return jsonify(run_simple_linear(df))


@app.route("/api/model/multiple-linear")
def multiple_linear():
    df = get_df()
    return jsonify(run_multiple_linear(df))


@app.route("/api/model/distributions")
def distributions():
    df = get_df()
    return jsonify(distribution_analysis(df))


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    result = predict_demand(body)

    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
            (price, discount, ad_spend, holiday, weekend, month, season, predicted, ci_lower, ci_upper)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        body.get("price"),
        body.get("discount"),
        body.get("advertising_spend"),
        body.get("holiday"),
        body.get("weekend"),
        body.get("month"),
        result["inputs_echo"]["season"],
        result["predicted_sales"],
        result["confidence_lower"],
        result["confidence_upper"],
    ))
    conn.commit(); conn.close()

    return jsonify(result)


@app.route("/api/predictions/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()
    cols = ["id","timestamp","price","discount","ad_spend","holiday",
            "weekend","month","season","predicted","ci_lower","ci_upper"]
    conn.close()
    return jsonify([dict(zip(cols, r)) for r in rows])


@app.route("/api/predictions/history/<int:pid>", methods=["DELETE"])
def delete_prediction(pid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions WHERE id=?", (pid,))
    conn.commit(); conn.close()
    return jsonify({"deleted": pid})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
