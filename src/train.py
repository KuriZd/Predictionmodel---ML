import os, json, argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from schema import infer_mapping, apply_mapping

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if "region" in df.columns:
        df = df.sort_values(["region", "year"])
        df["pop_lag1"] = df.groupby("region")["population"].shift(1)
        df["pop_lag2"] = df.groupby("region")["population"].shift(2)
        df["growth_rate"] = np.where(df["pop_lag1"] == 0, np.nan, (df["population"] - df["pop_lag1"]) / df["pop_lag1"])
        df["growth_lag1"] = df.groupby("region")["growth_rate"].shift(1)
    else:
        df = df.sort_values(["year"])
        df["pop_lag1"] = df["population"].shift(1)
        df["pop_lag2"] = df["population"].shift(2)
        df["growth_rate"] = np.where(df["pop_lag1"] == 0, np.nan, (df["population"] - df["pop_lag1"]) / df["pop_lag1"])
        df["growth_lag1"] = df["growth_rate"].shift(1)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna().reset_index(drop=True)

def make_target_classes(df: pd.DataFrame) -> pd.Series:
    q1, q2 = df["growth_rate"].quantile([0.33, 0.66]).values
    return pd.cut(df["growth_rate"], [-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype(int)

def build_mlp(input_dim: int, n_layers=3, units=96, dropout=0.25, lr=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main(csv_path: str, region_col: str | None, year_col: str | None, value_col: str | None,
         year_min=1950, year_max=2023, nn_layers=3, nn_units=96, nn_dropout=0.25, nn_lr=1e-3,
         nn_epochs=150, nn_batch=32, nn_patience=10):

    np.random.seed(42)
    tf.random.set_seed(42)

    print("Leyendo CSV:", csv_path)
    raw = pd.read_csv(csv_path)

    mapping = infer_mapping(raw)
    if region_col: mapping["region"] = region_col
    if year_col:   mapping["year"] = year_col
    if value_col:  mapping["population"] = value_col

    df = apply_mapping(raw, mapping)

    need = {"year", "population"}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Faltan columnas requeridas: {missing}. Mapeo detectado: {mapping}")

    keep = ["year", "population"] + (["region"] if "region" in df.columns else [])
    df = df[keep].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()

    df = make_features(df)
    y = make_target_classes(df)

    if "region" in df.columns:
        df = pd.get_dummies(df, columns=["region"], drop_first=True)
        region_feats = [c for c in df.columns if c.startswith("region_")]
    else:
        region_feats = []

    feature_cols = region_feats + ["year", "pop_lag1", "pop_lag2", "growth_lag1"]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    pre_rf = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
    rf = Pipeline([("pre", pre_rf), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    cm_rf = confusion_matrix(y_test, pred_rf)

    pre_nn = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
    X_train_s = pre_nn.fit_transform(X_train)
    X_test_s = pre_nn.transform(X_test)

    nn = build_mlp(X_train_s.shape[1], n_layers=nn_layers, units=nn_units, dropout=nn_dropout, lr=nn_lr)
    nn.fit(X_train_s, y_train, validation_split=0.2, epochs=nn_epochs, batch_size=nn_batch,
           callbacks=[keras.callbacks.EarlyStopping(patience=nn_patience, restore_best_weights=True)],
           verbose=0)
    pred_nn = np.argmax(nn.predict(X_test_s, verbose=0), axis=1)
    acc_nn = accuracy_score(y_test, pred_nn)
    cm_nn = confusion_matrix(y_test, pred_nn)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/model_rf.joblib")
    joblib.dump(pre_nn, "models/scaler_nn.joblib")
    nn.save("models/model_nn.keras")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    metrics = {
        "year_min": int(year_min),
        "year_max": int(year_max),
        "rf_accuracy": float(acc_rf),
        "nn_accuracy": float(acc_nn),
        "rf_confusion": cm_rf.tolist(),
        "nn_confusion": cm_nn.tolist(),
        "class_map": {"0": "Bajo", "1": "Medio", "2": "Alto"},
        "mapping_used": mapping,
    }
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/life-expectancy.csv")
    p.add_argument("--region-col", default=None)
    p.add_argument("--year-col", default=None)
    p.add_argument("--value-col", default=None)
    args = p.parse_args()
    main(args.csv, args.region_col, args.year_col, args.value_col)
