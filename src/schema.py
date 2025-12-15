import re
import pandas as pd

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def infer_mapping(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    ncols = [_norm(c) for c in cols]

    def pick(candidates):
        for cand in candidates:
            candn = _norm(cand)
            if candn in ncols:
                return cols[ncols.index(candn)]
        return None

    region = pick(["region", "entity", "country", "location", "state", "area", "name"])
    year   = pick(["year", "anio", "año", "date", "fecha", "time"])

    # 1) intenta encontrar un "valor" por nombres típicos
    value  = pick(["population", "pop", "value", "total", "count", "metric", "amount"])

    # 2) si no encontró, usa numéricas pero evitando la columna de year
    if value is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()

        # quita la columna que detectaste como year si es numérica
        if year in num_cols:
            num_cols = [c for c in num_cols if c != year]

        value = num_cols[0] if num_cols else None

    # 3) defensa: si por alguna razón value == year, intenta corregirlo
    if value == year:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c != year]
        value = num_cols[0] if num_cols else None

    return {"region": region, "year": year, "population": value}

def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    ren = {}
    for dst, src in mapping.items():
        if src and src in df.columns:
            ren[src] = dst
    return df.rename(columns=ren)
