# 🛒 Favorita Grocery Sales Forecasting Using SARIMAX (ARIMA with Exogenous Variables)

> **Advanced Time Series Modeling | Business-Focused Forecasting | Data Science Portfolio Project**

---

## 📌 Project Overview

This project applies **Seasonal ARIMA with Exogenous Variables (SARIMAX)** to forecast daily item sales for a leading Ecuadorian grocery chain — **Corporación Favorita** — based on historical sales, oil prices, customer transactions, item metadata, and holidays.

📊 The objective is not just forecasting but understanding **which external factors drive demand**, enhancing both **model accuracy** and **business decision-making**.

---

## 🧠 Business Objective

- 📦 Predict sales for top store-item combinations to optimize **inventory planning**
- 📈 Identify how external drivers like **holidays**, **oil prices**, and **footfall** influence demand
- 🧮 Develop a **robust forecasting model** that supports **data-driven decisions** for supply chain, pricing, and promotions

---

## 🧰 Dataset Summary

Sourced from the Kaggle competition:  
[📂 Favorita Grocery Sales Forecasting Dataset](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

| File | Description |
|------|-------------|
| `train.csv` | Historical daily sales for items at each store |
| `test.csv` | Test set (no target) |
| `stores.csv` | Metadata for stores: city, type, cluster |
| `items.csv` | Metadata for items: family, perishability |
| `transactions.csv` | Daily number of transactions per store |
| `holidays_events.csv` | Public & regional holidays/events |
| `oil.csv` | Daily oil price (economic indicator) |

---

## 🧪 Tech Stack

- **Languages & Tools**: Python, Jupyter Notebooks
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-learn
- **Model**: SARIMAX (from `statsmodels.tsa.statespace.sarimax`)

---

## 🔍 Exploratory Data Analysis (EDA)

### 🧾 Key Observations:
- Clear **seasonality** in sales (~7-day cycle)
- **Top items** dominate sales volume — classic Pareto (80/20) pattern
- **Transaction volume** highly correlated with sales → important exogenous variable
- **Oil price** fluctuations affect consumer spending (especially in 2017)
- **Holidays** (local & national) lead to dramatic spikes or dips in demand

### 🔧 Preprocessing:
- Missing oil prices forward-filled
- Sales log-transformed to reduce variance
- Data aggregated to a **Store+Item+Date** level
- Feature creation: `log_sales`, `log_transactions`, `perishable`, `holiday_flags`

---

## 🧠 Modeling Approach

### ✅ Why SARIMAX?
- Captures **autoregressive & seasonal** structure in daily sales
- Allows integration of **external variables** like transactions, oil prices
- Offers **interpretable parameters** for business explainability

### 📐 Model Configuration

```python
SARIMAX(
    endog = log_sales,                         # Target variable
    exog = [log_transactions, oil_price],      # External regressors
    order = (1,1,1),                            # ARIMA parameters
    seasonal_order = (1,1,1,7)                 # Weekly seasonality
)
```

## 📈 Results & Evaluation

| Metric                     | Value    |
|---------------------------|----------|
| Mean Absolute Error (MAE) | 216.95   |
| Root Mean Squared Error (RMSE) | 298.02 |

---

## 📊 Visualization

> The forecasted sales closely tracked actual sales for the validation period, capturing both trend and weekly seasonality effectively.  
> *(Visualization plot is included in the `/images` directory as `forecast_plot.png`)*

![Forecast vs Actual](./images/forecast_plot.png)

---

## 💡 Key Business Insights

- 🧍‍♂️ **Customer Footfall (transactions)** is the **strongest leading indicator** for sales — invaluable for **demand planning and inventory stocking**.
  
- 🛢️ **Oil prices** inversely impact sales — highlighting **macroeconomic sensitivity**. When oil prices rise, consumer spending dips slightly.

- 🛒 The **top 1% of items** account for **35%+ of revenue** — suggesting the need for a **Pareto-optimized inventory** strategy.

- 🎉 **Holidays & Events** (especially local and regional) cause high **spikes or dips** in sales — a key signal to **time promotions and staffing**.

- 🏬 **Store clusters** show **similar sales patterns**, indicating potential for **regionalized strategies** instead of per-store customization.

---

## 📊 Explainability & Residual Analysis

- The **SARIMAX model** provides **interpretable coefficients** showing how much each external factor (transactions, oil) contributes to predicted sales.
  
- **Residuals** were analyzed post-forecast:
  - 🟢 Showed signs of **white noise**, implying the model captured most of the signal.
  - 🟢 Residuals followed a **roughly normal distribution** — supporting model reliability.
  - 🟢 No clear **autocorrelation left unexplained**, affirming model stability.
