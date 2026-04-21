# 🎓 DemandIQ — Customer Demand Forecasting PSLP
### B.Tech Project | Python + Flask + SQLite + Chart.js

---

## 📁 Project Structure

```
demand-forecast/
├── backend/
│   ├── app.py            ← Flask REST API (all endpoints)
│   ├── ml_engine.py      ← ML models: SLR, MLR, distributions
│   └── generate_data.py  ← Synthetic Walmart-style dataset generator
├── frontend/
│   └── index.html        ← Full interactive dashboard (single file)
├── data/
│   ├── walmart_sales.csv ← Auto-generated on first run
│   └── forecast.db       ← SQLite database for prediction history
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn scipy matplotlib seaborn
```

### 2. Start the Backend
```bash
cd backend
python app.py
# ✅ Bootstrap complete – model trained.
# Running on http://localhost:5000
```

### 3. Open the Frontend
Open `frontend/index.html` in your browser (Chrome recommended).

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dataset/info` | Dataset shape, columns, sample |
| GET | `/api/eda` | EDA statistics & correlations |
| GET | `/api/eda/scatter` | Scatter data for charts |
| GET | `/api/model/simple-linear` | SLR results |
| GET | `/api/model/multiple-linear` | MLR results |
| GET | `/api/model/distributions` | Normal + Poisson analysis |
| POST | `/api/predict` | Live demand prediction |
| GET | `/api/predictions/history` | Saved predictions from DB |
| DELETE | `/api/predictions/history/:id` | Delete a prediction |

---

## 📊 Models Implemented

### A. Simple Linear Regression (SLR)
- **Formula:** `Sales = β₀ + β₁ × Advertising_Spend`
- **R² ≈ 0.41** (41% variance explained)

### B. Multiple Linear Regression (MLR)
- **Features:** Price, Discount, Ad Spend, Holiday, Weekend, Month, Season, Effective Price, Log(AdSpend)
- **R² ≈ 0.70** (70% variance explained)

### C. Statistical Distributions
- Normal distribution fit on Sales
- Poisson distribution for demand counts
- Logistic Regression explanation

---

## 📐 Custom Metric
```
Mean Error = Σ(Actual - Predicted) / n
```
Mean Error ≈ 0 confirms the model is unbiased.

---

## 🛠️ Tech Stack
- **Backend:** Python 3, Flask, scikit-learn, scipy, pandas, numpy
- **Database:** SQLite (via sqlite3)
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js 4, Anime.js
- **Design:** Industrial data-lab aesthetic with dark theme

---

## 👨‍💻 For B.Tech Submission
This project demonstrates:
1. Problem formulation
2. Dataset generation / preprocessing
3. EDA with visualizations
4. Simple + Multiple Linear Regression
5. Model evaluation (R², MAE, RMSE, Mean Error)
6. Statistical distributions (Normal, Poisson)
7. Live prediction system with database persistence
8. Full-stack web application
