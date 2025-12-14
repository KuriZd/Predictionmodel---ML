# Proyecto: Predicción de Crecimiento Poblacional (NN + Dashboard)

Este proyecto entrena modelos de **clasificación** para predecir el **nivel de crecimiento poblacional** (*Bajo / Medio / Alto*) usando:
- Un modelo base (ej. RandomForest)
- Una **Red Neuronal** (TensorFlow/Keras)
- Un **dashboard** simple con Streamlit para cargar un CSV, ver estadísticas y generar predicciones

---

## Requisitos
- Python 3.10+ (recomendado 3.10 o 3.11)
- pip actualizado

---

## Estructura recomendada

```txt
poblacion-nn/
  data/
    raw/
      train.csv
  models/
  src/
    train.py
    app.py
  README.md
```

- `data/raw/train.csv`: dataset de entrenamiento  
- `src/train.py`: script de entrenamiento (genera/guarda modelos)  
- `src/app.py`: dashboard Streamlit para inferencia y visualización  
- `models/`: carpeta sugerida para guardar modelos (si tu código los guarda ahí)

---

## Instalación (recomendado con entorno virtual)

### Windows (PowerShell)

```powershell
mkdir poblacion-nn
cd poblacion-nn
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Mac/Linux

```bash
mkdir poblacion-nn
cd poblacion-nn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## Dependencias

Instala dependencias con pip:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit matplotlib joblib
```

---

## Ejecución rápida (comandos requeridos)

Ejecuta en este orden:

```bash
python train.py
streamlit run app.py
```

> Si tus archivos están dentro de `src/`, usa:

```bash
python src/train.py
streamlit run src/app.py
```

---

## Dataset (formato esperado)

Tu CSV debe incluir al menos:
- Una columna de **año** (por ejemplo: `year`)
- Una columna de **población** (por ejemplo: `population`)
- (Opcional) una columna de **región/país/estado** (por ejemplo: `region`)

El script suele crear variables derivadas como:
- población rezagada (*lag1, lag2*)
- tasa de crecimiento (*growth_rate*)
- clase (*Bajo/Medio/Alto*) basada en cuantiles

---

## Qué hace el entrenamiento

`train.py` típicamente:
1. Limpia y ordena los datos
2. Genera *features* (lags, crecimiento)
3. Crea la variable objetivo por clases
4. Divide entrenamiento/prueba (50/50)
5. Entrena:
   - Modelo base (ej. RandomForest)
   - Red neuronal (MLP con Keras)
6. Reporta métricas (accuracy, matriz de confusión, reporte)
7. Guarda modelos para usarlos en el dashboard

---

## Qué hace el dashboard

`app.py` permite:
- Subir un CSV
- Ver resumen estadístico
- Generar predicciones con el modelo entrenado
- Mostrar al menos una gráfica de apoyo

---

## Solución de problemas

### 1) Streamlit no arranca

Prueba:

```bash
python -m streamlit run app.py
```

### 2) Error instalando TensorFlow

- Asegura Python 3.10/3.11
- Actualiza pip:

```bash
python -m pip install --upgrade pip
```

- Si sigue fallando, intenta:

```bash
pip install tensorflow-cpu
```

### 3) No encuentra archivos (train.py / app.py)

Revisa si están en `src/`:

```bash
python src/train.py
streamlit run src/app.py
```

---

## Autor
Proyecto académico — Unidad 3 (Redes Neuronales / Clasificación)
