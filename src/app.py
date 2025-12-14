import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción Índice de Crecimiento", layout="wide")
st.title("Dashboard — Predicción de índice de crecimiento")

rf = joblib.load("models/model_rf.joblib")
scaler_nn = joblib.load("models/scaler_nn.joblib")
nn = tf.keras.models.load_model("models/model_nn.keras")
feature_cols = joblib.load("models/feature_cols.joblib")

uploaded = st.file_uploader("Carga tu archivo CSV (population.csv)", type=["csv"])

year_min, year_max = st.slider("Rango de años", 1800, 2023, (1950, 2023))

if uploaded:
    df = pd.read_csv(uploaded)

    df = df.rename(
        columns={
            "Entity": "region",
            "Year": "year",
            "Population (historical)": "population",
        }
    )

    need = {"region", "year", "population"}
    if not need.issubset(df.columns):
        st.error(f"Tu CSV debe incluir columnas: {sorted(list(need))}")
        st.stop()

    df = df[["region", "year", "population"]].dropna()
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()

    st.subheader("Resumen estadístico")
    st.dataframe(df.describe(include="all").T)

    df = df.sort_values(["region", "year"])
    df["pop_lag1"] = df.groupby("region")["population"].shift(1)
    df["pop_lag2"] = df.groupby("region")["population"].shift(2)
    df["growth_rate"] = (df["population"] - df["pop_lag1"]) / df["pop_lag1"]
    df["growth_lag1"] = df.groupby("region")["growth_rate"].shift(1)
    df = df.dropna().reset_index(drop=True)

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    X = df.reindex(columns=feature_cols, fill_value=0)

    st.subheader("Predicciones")
    col1, col2 = st.columns(2)

    pred_rf = rf.predict(X)
    pred_nn = np.argmax(nn.predict(scaler_nn.transform(X), verbose=0), axis=1)

    with col1:
        st.markdown("### RandomForest")
        st.write(pd.Series(pred_rf).value_counts().rename({0: "Bajo", 1: "Medio", 2: "Alto"}))

    with col2:
        st.markdown("### Red neuronal (MLP)")
        st.write(pd.Series(pred_nn).value_counts().rename({0: "Bajo", 1: "Medio", 2: "Alto"}))

    st.subheader("Gráfica (growth_rate)")
    fig = plt.figure()
    plt.plot(df["growth_rate"].values)
    plt.title("Growth rate (serie)")
    st.pyplot(fig)
else:
    st.info("Sube el population.csv para ver estadísticas y predicciones.")
