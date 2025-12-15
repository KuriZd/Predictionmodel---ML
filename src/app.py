import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt


LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}


def prepare_generic_timeseries(
    raw: pd.DataFrame,
    region_col: str | None,
    year_col: str,
    value_col: str,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    df = raw.copy()

    # Renombrar a nombres internos estándar
    rename_map = {year_col: "year", value_col: "population"}
    if region_col:
        rename_map[region_col] = "region"
    df = df.rename(columns=rename_map)

    need = {"year", "population"}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    keep = ["year", "population"] + (["region"] if "region" in df.columns else [])
    df = df[keep].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna()

    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()

    # Features (con o sin region)
    if "region" in df.columns:
        df = df.sort_values(["region", "year"])
        df["pop_lag1"] = df.groupby("region")["population"].shift(1)
        df["pop_lag2"] = df.groupby("region")["population"].shift(2)
        df["growth_rate"] = np.where(
            df["pop_lag1"] == 0, np.nan, (df["population"] - df["pop_lag1"]) / df["pop_lag1"]
        )
        df["growth_lag1"] = df.groupby("region")["growth_rate"].shift(1)
    else:
        df = df.sort_values(["year"])
        df["pop_lag1"] = df["population"].shift(1)
        df["pop_lag2"] = df["population"].shift(2)
        df["growth_rate"] = np.where(
            df["pop_lag1"] == 0, np.nan, (df["population"] - df["pop_lag1"]) / df["pop_lag1"]
        )
        df["growth_lag1"] = df["growth_rate"].shift(1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # One-hot solo si existe region
    if "region" in df.columns:
        df = pd.get_dummies(df, columns=["region"], drop_first=True)

    return df



st.set_page_config(page_title="Predicción Índice de Crecimiento", layout="wide")
st.title("Dashboard — Predicción de índice de crecimiento")

rf = joblib.load("models/model_rf.joblib")
scaler_nn = joblib.load("models/scaler_nn.joblib")
nn = tf.keras.models.load_model("models/model_nn.keras")
feature_cols = joblib.load("models/feature_cols.joblib")

metrics = None
try:
    with open("models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
except Exception:
    metrics = None

st.subheader("Métricas del entrenamiento (test 50/50)")
if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy RF", f"{metrics['rf_accuracy']:.3f}")
    c2.metric("Accuracy NN (MLP)", f"{metrics['nn_accuracy']:.3f}")
    c3.metric("Rango años (train)", f"{metrics['year_min']}–{metrics['year_max']}")

    st.markdown("**Matriz de confusión (test)**")
    a, b = st.columns(2)
    with a:
        st.write("RandomForest")
        st.table(metrics["rf_confusion"])
    with b:
        st.write("Red neuronal (MLP)")
        st.table(metrics["nn_confusion"])
else:
    st.info("No encontré models/metrics.json. Ejecuta primero: python src/train.py")

st.divider()

uploaded = st.file_uploader("Carga tu archivo CSV", type=["csv"])
year_min, year_max = st.slider("Rango de años (dashboard)", 1800, 2023, (1950, 2023))

if uploaded:
    raw = pd.read_csv(uploaded)
    cols = raw.columns.tolist()

    st.subheader("Mapeo de columnas (elige qué significa cada columna)")
    c1, c2, c3 = st.columns(3)

    with c1:
        region_col = st.selectbox("Columna región (opcional)", ["(ninguna)"] + cols, index=0)
    with c2:
        year_col = st.selectbox("Columna año/tiempo", cols, index=0)
    with c3:
        value_col = st.selectbox("Columna valor numérico", cols, index=0)

    region_col = None if region_col == "(ninguna)" else region_col

    # ✅ Validaciones (EVITA colisiones como Entity=year y Entity=population)
    if year_col == value_col:
        st.error("La columna año/tiempo y la columna valor numérico NO pueden ser la misma.")
        st.stop()

    if region_col and region_col == year_col:
        st.error("La columna región no puede ser la misma que la de año/tiempo.")
        st.stop()

    if region_col and region_col == value_col:
        st.error("La columna región no puede ser la misma que la del valor numérico.")
        st.stop()

    st.subheader("Resumen estadístico")
    try:
        tmp = raw.rename(columns={
            (region_col or ""): "region",
            year_col: "year",
            value_col: "population",
        })
        show_cols = ["year", "population"] + (["region"] if region_col else [])
        st.dataframe(tmp[show_cols].describe(include="all").T)
    except Exception:
        st.dataframe(raw.describe(include="all").T)

    try:
        df = prepare_generic_timeseries(raw, region_col, year_col, value_col, year_min, year_max)
    except Exception as e:
        st.error(str(e))
        st.stop()

    X = df.reindex(columns=feature_cols, fill_value=0)

    pred_rf = rf.predict(X)
    pred_nn = np.argmax(nn.predict(scaler_nn.transform(X), verbose=0), axis=1)

    st.subheader("Predicciones")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### RandomForest")
        st.write(pd.Series(pred_rf).map(LABELS).value_counts())

    with col2:
        st.markdown("### Red neuronal (MLP)")
        st.write(pd.Series(pred_nn).map(LABELS).value_counts())

    st.subheader("Gráfica (growth_rate)")
    chart_height = st.slider("Altura de la gráfica (px)", 160, 520, 220, 10)

    plot = df[["year", "growth_rate"]].copy()
    plot = plot.sort_values("year").groupby("year", as_index=False)["growth_rate"].mean()
    plot = plot.set_index("year")
    st.line_chart(plot, height=chart_height)

else:
    st.info("Sube un CSV para ver estadísticas y predicciones.")

