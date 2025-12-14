import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "year"])
    df["pop_lag1"] = df.groupby("region")["population"].shift(1)
    df["pop_lag2"] = df.groupby("region")["population"].shift(2)

    df["growth_rate"] = np.where(
        df["pop_lag1"] == 0,
        np.nan,
        (df["population"] - df["pop_lag1"]) / df["pop_lag1"],
    )

    df["growth_lag1"] = df.groupby("region")["growth_rate"].shift(1)
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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(
    csv_path: str,
    year_min: int = 1950,
    year_max: int = 2023,
    nn_layers: int = 3,
    nn_units: int = 96,
    nn_dropout: float = 0.25,
    nn_lr: float = 1e-3,
    nn_epochs: int = 150,
    nn_batch: int = 32,
    nn_patience: int = 10,
):
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "Entity": "region",
            "Year": "year",
            "Population (historical)": "population",
        }
    )

    need = {"region", "year", "population"}
    if not need.issubset(df.columns):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df = df[["region", "year", "population"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna()

    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()

    df = make_features(df)
    y = make_target_classes(df)

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    feature_cols = [c for c in df.columns if c.startswith("region_")] + [
        "year",
        "pop_lag1",
        "pop_lag2",
        "growth_lag1",
    ]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    pre_rf = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])

    rf = Pipeline(
        steps=[
            ("pre", pre_rf),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)
    cm_rf = confusion_matrix(y_test, pred_rf)

    pre_nn = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])
    X_train_s = pre_nn.fit_transform(X_train)
    X_test_s = pre_nn.transform(X_test)

    nn = build_mlp(
        input_dim=X_train_s.shape[1],
        n_layers=nn_layers,
        units=nn_units,
        dropout=nn_dropout,
        lr=nn_lr,
    )
    nn.fit(
        X_train_s,
        y_train,
        validation_split=0.2,
        epochs=nn_epochs,
        batch_size=nn_batch,
        callbacks=[keras.callbacks.EarlyStopping(patience=nn_patience, restore_best_weights=True)],
        verbose=0,
    )
    pred_nn = np.argmax(nn.predict(X_test_s, verbose=0), axis=1)
    acc_nn = accuracy_score(y_test, pred_nn)
    cm_nn = confusion_matrix(y_test, pred_nn)

    print("\n=== RandomForest ===")
    print("Accuracy:", acc_rf)
    print(classification_report(y_test, pred_rf))
    print("Confusion:\n", cm_rf)

    print("\n=== Neural Net (MLP) ===")
    print("Accuracy:", acc_nn)
    print(classification_report(y_test, pred_nn))
    print("Confusion:\n", cm_nn)

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
        "nn_params": {
            "layers": int(nn_layers),
            "units": int(nn_units),
            "dropout": float(nn_dropout),
            "lr": float(nn_lr),
            "epochs": int(nn_epochs),
            "batch": int(nn_batch),
            "patience": int(nn_patience),
        },
    }

    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main("data/raw/population.csv")
