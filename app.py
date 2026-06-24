"""
🚗 Used Car Price Predictor — Streamlit app
Author: Adarsh Raj

Enter a car's details and get an estimated resale price, powered by a
Random Forest model trained on 15,000+ CarDekho listings.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/cardekho_imputated.csv"
CATEGORICAL = ["model", "seller_type", "fuel_type", "transmission_type"]
NUMERIC = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
TARGET = "selling_price"

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_data
def load_data():
    return pd.read_csv(DATA, index_col=[0])


@st.cache_resource
def train_model(df):
    """Train the pipeline once and cache it for the life of the container."""
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


df = load_data()
with st.spinner("Warming up the model (first run only)…"):
    model = train_model(df)

# ---------- UI ----------
st.title("🚗 Used Car Price Predictor")
st.markdown(
    "Estimate the resale price of a used car in India. "
    "Powered by a Random Forest model trained on **15,000+ CarDekho listings** "
    "(~0.93 R²). Built by **Adarsh Raj**."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Car model", sorted(df["model"].unique()))
    fuel_type = st.selectbox("Fuel type", sorted(df["fuel_type"].unique()))
    transmission_type = st.selectbox(
        "Transmission", sorted(df["transmission_type"].unique())
    )
    seller_type = st.selectbox("Seller type", sorted(df["seller_type"].unique()))
    seats = st.selectbox(
        "Seats", sorted(int(s) for s in df["seats"].unique()), index=0
    )

with col2:
    vehicle_age = st.slider(
        "Vehicle age (years)",
        int(df["vehicle_age"].min()),
        int(df["vehicle_age"].max()),
        int(df["vehicle_age"].median()),
    )
    km_driven = st.slider(
        "Kilometres driven",
        0,
        int(df["km_driven"].quantile(0.99)),
        int(df["km_driven"].median()),
        step=1000,
    )
    mileage = st.slider(
        "Mileage (kmpl)",
        float(round(df["mileage"].min(), 1)),
        float(round(df["mileage"].max(), 1)),
        float(round(df["mileage"].median(), 1)),
    )
    engine = st.slider(
        "Engine (cc)",
        int(df["engine"].min()),
        int(df["engine"].quantile(0.99)),
        int(df["engine"].median()),
        step=50,
    )
    max_power = st.slider(
        "Max power (bhp)",
        float(round(df["max_power"].min(), 1)),
        float(round(df["max_power"].quantile(0.99), 1)),
        float(round(df["max_power"].median(), 1)),
    )

st.divider()

if st.button("💰 Predict Price", type="primary", use_container_width=True):
    row = pd.DataFrame(
        [
            {
                "model": model_name,
                "seller_type": seller_type,
                "fuel_type": fuel_type,
                "transmission_type": transmission_type,
                "vehicle_age": vehicle_age,
                "km_driven": km_driven,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
            }
        ]
    )
    price = float(model.predict(row)[0])
    st.success("### Estimated resale price")
    st.metric(label="Predicted selling price", value=f"₹ {price:,.0f}")
    st.caption(
        "Estimate only — actual prices vary with condition, location, and demand. "
        "Model mean absolute error is ≈ ₹1,00,000."
    )

st.divider()
st.caption(
    "Data: CarDekho listings · Model: Random Forest Regressor · "
    "[Source code](https://github.com/adarshraj18/used-car-price-prediction)"
)
