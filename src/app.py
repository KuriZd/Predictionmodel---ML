import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

YEAR_COL = "year"
GROUP_COL = "region"
POP_COL = "population"

st.set_page_config(page_title="Predicción Índice de Crecimiento", layout="wide")
st.title("Dashboard — Predicción de índice de crecimiento")

rf = joblib.load("models/model_rf.joblib")
scaler_nn = joblib.load("models/scaler_nn.joblib")
nn = tf.keras.models.load_model("models/model_nn.keras")
feature_cols = joblib.load("models/feature_cols.joblib")

uploaded = st.file_uploader("Carga tu archivo CSV de pruebas", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Resumen estadístico")
    st.dataframe(df.describe(include="all").T)

    if YEAR_COL not in df.columns or POP_COL not in df.columns:
        st.error(f"Tu CSV debe incluir columnas: '{YEAR_COL}' y '{POP_COL}'")
        st.stop()

    if GROUP_COL in df.columns:
        df = df.sort_values([GROUP_COL, YEAR_COL])
        df["pop_lag1"] = df.groupby(GROUP_COL)[POP_COL].shift(1)
        df["pop_lag2"] = df.groupby(GROUP_COL)[POP_COL].shift(2)
    else:
        df = df.sort_values(YEAR_COL)
        df["pop_lag1"] = df[POP_COL].shift(1)
        df["pop_lag2"] = df[POP_COL].shift(2)

    df["growth_rate"] = (df[POP_COL] - df["pop_lag1"]) / df["pop_lag1"]

    if GROUP_COL in df.columns:
        df["growth_lag1"] = df.groupby(GROUP_COL)["growth_rate"].shift(1)
    else:
        df["growth_lag1"] = df["growth_rate"].shift(1)

    df = df.dropna().reset_index(drop=True)

    if GROUP_COL in df.columns:
        df = pd.get_dummies(df, columns=[GROUP_COL], drop_first=True)

    X = df.reindex(columns=feature_cols, fill_value=0)

    st.subheader("Predicciones")
    col1, col2 = st.columns(2)

    pred_rf = rf.predict(X)
    Xs = scaler_nn.transform(X)
    pred_nn = np.argmax(nn.predict(Xs, verbose=0), axis=1)

    with col1:
        st.markdown("### RandomForest")
        st.write(pd.Series(pred_rf).value_counts().rename({0: "Bajo", 1: "Medio", 2: "Alto"}))

    with col2:
        st.markdown("### Red neuronal (MLP)")
        st.write(pd.Series(pred_nn).value_counts().rename({0: "Bajo", 1: "Medio", 2: "Alto"}))

    st.subheader("Gráfica rápida (growth_rate)")
    fig = plt.figure()
    plt.plot(df["growth_rate"].values)
    plt.title("Growth rate (serie)")
    st.pyplot(fig)
else:
    st.info("Sube un CSV para ver estadísticas y predicciones.")
