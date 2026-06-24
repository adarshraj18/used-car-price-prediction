# 🚗 Used Car Price Prediction

Predicting the resale price of used cars in India using machine learning, based on
real listings scraped from [CarDekho.com](https://www.cardekho.com/).

> **Author:** Adarsh Raj
> **Type:** End-to-end supervised regression project + interactive web app
> **Best model:** Random Forest Regressor — **0.93 test R²**

---

## 🖥️ Try the App

An interactive **Streamlit app** ([`app.py`](app.py)) lets anyone enter a car's
details (model, age, km driven, engine, power, fuel, transmission…) and get an
instant predicted resale price.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app trains the model once on startup (cached), so there's no large model file to
download — just clone and run. It also deploys for free on
[Streamlit Community Cloud](https://streamlit.io/cloud) straight from this repo.

---

## 📌 Problem

Used-car pricing in India is inconsistent and opaque — sellers rarely have a reliable
reference for what their vehicle is worth. This project builds a model that predicts a
car's **selling price** from its core attributes, which could power a price-suggestion
feature that gives sellers an instant, data-backed estimate.

## 📊 Dataset

| | |
|---|---|
| Source | CarDekho.com listings (scraped) |
| Rows | 15,411 |
| Columns | 13 |
| Target | `selling_price` (₹) |

**Features:** `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`,
`transmission_type`, `mileage`, `engine`, `max_power`, `seats`.

The dataset lives in [`data/cardekho_imputated.csv`](data/cardekho_imputated.csv).

## 🛠️ Approach

1. **Data cleaning** — null/duplicate checks, drop redundant identifier columns.
2. **Feature engineering** — label-encode high-cardinality `model`, one-hot encode
   low-cardinality categoricals, and standard-scale numerics using a scikit-learn
   `ColumnTransformer`.
3. **Model comparison** — benchmark 6 regressors: Linear Regression, Lasso, Ridge,
   K-Neighbors, Decision Tree, and Random Forest.
4. **Hyperparameter tuning** — `RandomizedSearchCV` on the two strongest models.
5. **Evaluation** — select the model with the best generalisation (test R²).

## 📈 Results

| Model | Test R² | Test RMSE (₹) |
|---|---|---|
| Linear / Lasso / Ridge | 0.66 | ~502,500 |
| Decision Tree | 0.87 | ~309,800 |
| K-Neighbors | 0.91 | ~253,100 |
| **Random Forest (tuned)** | **0.93** | **~228,400** |

The **Random Forest Regressor** explains ~93% of the variance in resale price with the
lowest test error — robust where a single Decision Tree badly overfit (0.9995 train R²
vs 0.87 test). Linear models underfit, confirming the price relationship is non-linear.

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/used-car-price-prediction.git
cd used-car-price-prediction

# 2. (Optional) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Used_Car_Price_Prediction.ipynb
```

## 📁 Project Structure

```
.
├── Used_Car_Price_Prediction.ipynb   # Main analysis & modelling notebook
├── app.py                             # Interactive Streamlit price-predictor app
├── data/
│   └── cardekho_imputated.csv         # Dataset
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT
└── README.md
```

## 🔭 Next Steps

- Add feature-importance and SHAP analysis for explainability.
- Persist the trained pipeline (`joblib`) and serve it via a small API / Streamlit app.
- Engineer richer features (brand premium, mileage-per-year, vehicle segment).

## 📄 License

Released under the [MIT License](LICENSE).

---

Built by **Adarsh Raj**.
