import os
import pandas as pd
import numpy as np

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


YEAR_COL = "year"
GROUP_COL = "region"
POP_COL = "population"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if YEAR_COL not in df.columns or POP_COL not in df.columns:
        raise ValueError(f"Faltan columnas requeridas: '{YEAR_COL}' y/o '{POP_COL}'")

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

    return df.dropna().reset_index(drop=True)


def make_target_classes(df: pd.DataFrame) -> pd.Series:
    q1, q2 = df["growth_rate"].quantile([0.33, 0.66]).values
    return pd.cut(
        df["growth_rate"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=[0, 1, 2],
    ).astype(int)


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


def main(train_csv: str):
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv(train_csv)
    df = make_features(df)
    y = make_target_classes(df)

    base_feature_cols = [YEAR_COL, "pop_lag1", "pop_lag2", "growth_lag1"]

    if GROUP_COL in df.columns:
        df = pd.get_dummies(df, columns=[GROUP_COL], drop_first=True)
        region_cols = [c for c in df.columns if c.startswith(GROUP_COL + "_")]
        feature_cols = region_cols + base_feature_cols
    else:
        feature_cols = base_feature_cols

    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    pre_rf = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())],
        remainder="drop",
    )

    rf = Pipeline(
        steps=[
            ("pre", pre_rf),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=42)),
        ]
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)

    pre_nn = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())],
        remainder="drop",
    )
    X_train_s = pre_nn.fit_transform(X_train)
    X_test_s = pre_nn.transform(X_test)

    nn = build_mlp(input_dim=X_train_s.shape[1])
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    nn.fit(
        X_train_s,
        y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    pred_nn = np.argmax(nn.predict(X_test_s, verbose=0), axis=1)
    acc_nn = accuracy_score(y_test, pred_nn)

    print("\n=== RandomForest ===")
    print("Accuracy:", acc_rf)
    print(classification_report(y_test, pred_rf))
    print("Confusion:\n", confusion_matrix(y_test, pred_rf))

    print("\n=== Neural Net (MLP) ===")
    print("Accuracy:", acc_nn)
    print(classification_report(y_test, pred_nn))
    print("Confusion:\n", confusion_matrix(y_test, pred_nn))

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/model_rf.joblib")
    joblib.dump(pre_nn, "models/scaler_nn.joblib")
    nn.save("models/model_nn.keras")
    joblib.dump(feature_cols, "models/feature_cols.joblib")


if __name__ == "__main__":
    main("data/raw/train.csv")
