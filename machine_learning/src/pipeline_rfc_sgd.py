# ============================================================
# pipeline_rfc_sgd.py — Pipeline productivo RFC + SGD
# - PASO 1: Carga y splits (2015–2021)
# - PASO 2: Tuning RFC/SGD en 2021-H1 y evaluación en 2021-H2
# - PASO 3: Calibración con VALID y evaluación en TEST (sin fuga)
# - PASO 4: Rolling mensual 2021–2024 con embargo + calibración -> data/processed
# - PASO 5: Rolling mensual 2025 (ene–ago) con embargo + calibración -> data/out + models/
# ============================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.calibration")
# ------------------------------
# Imports generales
# ------------------------------
import numpy as np
import pandas as pd
import joblib
import json

from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
# ------------------------------
# Configuración de rutas
# ------------------------------
from pathlib import Path

# Ruta base = carpeta raíz del proyecto (2 niveles arriba de este script)
BASE_DIR = Path(__file__).resolve().parent.parent

# Subcarpetas principales
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"        # Datos originales descargados
SRC_DIR       = DATA_DIR / "src"        # Datos intermedios (ej. 2025 con indicadores)
PROCESSED_DIR = DATA_DIR / "processed"  # Datos enriquecidos (probas rolling 2021–2024/2025)
OUT_DIR       = DATA_DIR / "out"        # Salidas finales listas para backtest/dashboards
REPORTS_DIR   = BASE_DIR / "reports"    # Métricas, predicciones, análisis
MODELS_DIR    = BASE_DIR / "models"     # Modelos entrenados y calibrados
REPORTS_DIR   = BASE_DIR / "reports"    # Parametros y reportes

# Crear subcarpetas si no existen
for d in [RAW_DIR, SRC_DIR, PROCESSED_DIR, OUT_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Archivos de entrada
CSV_PATH       = SRC_DIR / "btc_enriched_with_target.csv"         # dataset histórico base (2015–2024)
CSV_2025_SRC   = SRC_DIR / "btc_yahoo_2025_whit_indicators.csv"   # dataset 2025 con indicadores

# Archivos procesados (pipeline)
OOS_CSV_2021_2024 = PROCESSED_DIR / "probas_oos_rfc_sgd_2021_2024.csv"
OOS_CSV_2025      = PROCESSED_DIR / "probas_oos_rfc_sgd_2025.csv"  # opcional si quieres guardar también aquí

# Archivos de salida enriquecidos
CSV_2025_WITH_PROBS = OUT_DIR / "btc_2025_with_probs.csv"

# Ejemplos de reports (no usados en el pipeline productivo, pero disponibles)
PRED_RFC_CSV    = REPORTS_DIR / "predictions_RFC.csv"
METRICS_RFC_CSV = REPORTS_DIR / "metrics_RFC.csv"
PRED_SGD_CSV    = REPORTS_DIR / "predictions_SGD.csv"
METRICS_SGD_CSV = REPORTS_DIR / "metrics_SGD.csv"

# ============================================================
# PASO 1 — Carga y splits
# ============================================================
DATE_COL   = "Date"
TARGET_COL = "Target"  # asumimos binaria (0/1)

# Rangos de tiempo (inclusivos) para train/val/test iniciales
TRAIN_START, TRAIN_END = "2015-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2021-06-30"
TEST_START,  TEST_END  = "2021-07-01", "2021-12-31"

print(">> Cargando dataset histórico:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

assert DATE_COL in df.columns, f"No se encontró la columna de fecha '{DATE_COL}' en el CSV."
df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=False, errors="coerce")
df = df.sort_values(DATE_COL).dropna(subset=[DATE_COL]).reset_index(drop=True)
df = df.set_index(DATE_COL)

# Limpieza mínima
df = df[~df.index.duplicated(keep="last")]
for c in df.columns:
    if c != TARGET_COL:
        df[c] = pd.to_numeric(df[c], errors="ignore")

assert TARGET_COL in df.columns, f"No se encontró la columna objetivo '{TARGET_COL}' en el CSV."
if not set(df[TARGET_COL].dropna().unique()).issubset({0, 1}):
    raise ValueError(f"La columna '{TARGET_COL}' no es binaria 0/1.")

# Quitar filas con NaN (por ventanas de indicadores)
df = df.dropna()

# Features y target
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)

X_all = df[num_cols].copy()
y_all = df[TARGET_COL].astype(int).copy()

# Splits temporales
X_train = X_all.loc[TRAIN_START:TRAIN_END]
y_train = y_all.loc[TRAIN_START:TRAIN_END]
X_val   = X_all.loc[VAL_START:VAL_END]
y_val   = y_all.loc[VAL_START:VAL_END]
X_test  = X_all.loc[TEST_START:TEST_END]
y_test  = y_all.loc[TEST_START:TEST_END]

def _resumen_split(X, y):
    return {
        "rango_fechas": (X.index.min().date() if len(X) else None,
                         X.index.max().date() if len(X) else None),
        "muestras": len(X),
        "features": X.shape[1],
        "positivos_%": (float(y.mean())*100 if len(y) else np.nan)
    }

print(">> Columnas de entrada (X):")
print(num_cols)
print("\n>> Resumen de splits:")
print("TRAIN (2015-2020):", _resumen_split(X_train, y_train))
print("VALID (2021-H1):  ", _resumen_split(X_val,   y_val))
print("TEST  (2021-H2):  ", _resumen_split(X_test,  y_test))

# Verificación de solapamientos
assert len(X_train.index.intersection(X_val.index))  == 0, "Solapamiento TRAIN-VALID."
assert len(X_val.index.intersection(X_test.index))   == 0, "Solapamiento VALID-TEST."
assert len(X_train.index.intersection(X_test.index)) == 0, "Solapamiento TRAIN-TEST."
print("\n>> Verificación de solapamientos: OK")

# ============================================================
# PASO 2 — RFC y SGD: cargar parámetros desde JSON y evaluar
#      
# ============================================================

def evaluar(nombre, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    pr  = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan

    print(f"\n=== {nombre} (thr=0.5) ===")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"ROC AUC:  {roc:.3f} | PR AUC:   {pr:.3f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc": roc, "pr": pr}

# 1) Cargar mejores parámetros desde JSON (obligatorio)
params_path = REPORTS_DIR / "best_params.json"
if not params_path.exists():
    raise FileNotFoundError(
        f"No se encontró {params_path}. Genera primero ese archivo con los mejores parámetros."
    )

print("\n>> Cargando parámetros desde best_params.json...")
with open(params_path, "r") as f:
    best_params = json.load(f)

best_rfc_params = best_params["RFC"]
best_sgd_params = best_params["SGD"]
print("Mejores params RFC:", best_rfc_params)
print("Mejores params SGD:", best_sgd_params)

# 2) Reentrenar con TRAIN+VALID y evaluar en TEST (2021-H2)
X_trval = pd.concat([X_train, X_val], axis=0).sort_index()
y_trval = pd.concat([y_train, y_val], axis=0).sort_index()

rfc_final = RandomForestClassifier(
    **best_rfc_params,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rfc_final.fit(X_trval, y_trval)
rfc_test_proba = rfc_final.predict_proba(X_test)[:, 1]

sgd_final = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", SGDClassifier(
        **best_sgd_params,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
sgd_final.fit(X_trval, y_trval)
if hasattr(sgd_final.named_steps["clf"], "predict_proba"):
    sgd_test_proba = sgd_final.predict_proba(X_test)[:, 1]
else:
    sgd_scores = sgd_final.decision_function(X_test)
    sgd_test_proba = 1 / (1 + np.exp(-sgd_scores))

_ = evaluar("RFC Test 2021-H2", y_test.values, rfc_test_proba, threshold=0.5)
_ = evaluar("SGD Test 2021-H2", y_test.values, sgd_test_proba, threshold=0.5)

print("\n--- Classification report (RFC) ---")
print(classification_report(y_test.values, (rfc_test_proba >= 0.5).astype(int), zero_division=0))
print("\n--- Classification report (SGD) ---")
print(classification_report(y_test.values, (sgd_test_proba >= 0.5).astype(int), zero_division=0))


# ============================================================
# PASO 3 — Calibración + Umbral óptimo (sin fuga)
# - Reentrena en TRAIN (2015–2020)
# - Calibra en VALID (2021-H1)
# - Evalúa en TEST (2021-H2)
# ============================================================
def evaluar_con_umbral(nombre, y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan
    pr  = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan

    print(f"\n=== {nombre} (thr={thr:.3f}) ===")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"ROC AUC:  {roc:.3f} | PR AUC:   {pr:.3f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, roc=roc, pr=pr, thr=thr)

def buscar_umbral_f1(y_true, proba, n=200):
    q05, q95 = np.quantile(proba, 0.05), np.quantile(proba, 0.95)
    grid = np.linspace(q05, q95, n)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1

# RFC: reentreno en TRAIN, calibración en VALID
rfc_train = RandomForestClassifier(
    **best_rfc_params,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rfc_train.fit(X_train, y_train)
rfc_cal = CalibratedClassifierCV(rfc_train, method="isotonic", cv="prefit")
rfc_cal.fit(X_val, y_val)
rfc_val_proba = rfc_cal.predict_proba(X_val)[:, 1]
rfc_thr_opt, rfc_f1_val = buscar_umbral_f1(y_val.values, rfc_val_proba, n=200)
print(f">> RFC — Umbral óptimo VALID: {rfc_thr_opt:.4f} | F1_VALID={rfc_f1_val:.3f}")
rfc_test_proba_cal = rfc_cal.predict_proba(X_test)[:, 1]
_ = evaluar_con_umbral("RFC calibrado — TEST 2021-H2", y_test.values, rfc_test_proba_cal, rfc_thr_opt)

# SGD: reentreno (pipeline) en TRAIN, calibración en VALID
sgd_base = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", SGDClassifier(
        **best_sgd_params,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])
sgd_base.fit(X_train, y_train)
sgd_cal = CalibratedClassifierCV(sgd_base, method="sigmoid", cv="prefit")
sgd_cal.fit(X_val, y_val)
sgd_val_proba = sgd_cal.predict_proba(X_val)[:, 1]
sgd_thr_opt, sgd_f1_val = buscar_umbral_f1(y_val.values, sgd_val_proba, n=200)
print(f">> SGD — Umbral óptimo VALID: {sgd_thr_opt:.4f} | F1_VALID={sgd_f1_val:.3f}")
sgd_test_proba_cal = sgd_cal.predict_proba(X_test)[:, 1]
_ = evaluar_con_umbral("SGD calibrado — TEST 2021-H2", y_test.values, sgd_test_proba_cal, sgd_thr_opt)

# ============================================================
# PASO 4 — Rolling mensual 2021–2024 con embargo y calibración
# ============================================================
ROLL_START_2124 = pd.Timestamp("2021-01-01")
ROLL_END_2124   = pd.Timestamp("2024-12-31")
CAL_WINDOW_DAYS_MIN = 60
CAL_WINDOW_DAYS_DEF = 90

def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthEnd(0)

def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthBegin(0)

def fit_rfc_base_hist(X_hist, y_hist):
    model = RandomForestClassifier(
        **best_rfc_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_hist, y_hist)
    return model

def fit_sgd_base_hist(X_hist, y_hist):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            **best_sgd_params,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    pipe.fit(X_hist, y_hist)
    return pipe

rows = []
current = month_start(ROLL_START_2124)
while current <= ROLL_END_2124:
    m_start = month_start(current)
    m_end   = month_end(current)
    train_end = m_start - pd.Timedelta(days=1)

    if train_end < X_all.index.min():
        current = m_start + pd.offsets.MonthBegin(1)
        continue

    cal_end = train_end
    cal_start = cal_end - pd.Timedelta(days=CAL_WINDOW_DAYS_DEF - 1)
    if cal_start < X_all.index.min():
        cal_start = X_all.index.min()
    if (cal_end - cal_start).days + 1 < CAL_WINDOW_DAYS_MIN:
        cal_start = cal_end - pd.Timedelta(days=CAL_WINDOW_DAYS_MIN - 1)
        if cal_start < X_all.index.min():
            cal_start = X_all.index.min()

    train_base_end = cal_start - pd.Timedelta(days=1)
    if train_base_end < X_all.index.min():
        train_base_end = train_end

    X_hist = X_all.loc[:train_base_end]
    y_hist = y_all.loc[:train_base_end]
    if len(X_hist) < 50:
        X_hist = X_all.loc[:train_end]
        y_hist = y_all.loc[:train_end]

    X_cal = X_all.loc[cal_start:cal_end]
    y_cal = y_all.loc[cal_start:cal_end]
    if len(X_hist) < 50 or len(X_cal) < 30:
        current = m_start + pd.offsets.MonthBegin(1)
        continue

    # Embargo: descartar el día 1 del mes (no se predice)
    X_month_pred = X_all.loc[(m_start + pd.Timedelta(days=1)):m_end]
    y_month_true = y_all.loc[(m_start + pd.Timedelta(days=1)):m_end]
    if len(X_month_pred) == 0:
        current = m_start + pd.offsets.MonthBegin(1)
        continue

    rfc_base = fit_rfc_base_hist(X_hist, y_hist)
    rfc_cal_ = CalibratedClassifierCV(rfc_base, method="isotonic", cv="prefit")
    rfc_cal_.fit(X_cal, y_cal)
    rfc_proba_month = rfc_cal_.predict_proba(X_month_pred)[:, 1]

    sgd_base_ = fit_sgd_base_hist(X_hist, y_hist)
    sgd_cal_  = CalibratedClassifierCV(sgd_base_, method="sigmoid", cv="prefit")
    sgd_cal_.fit(X_cal, y_cal)
    sgd_proba_month = sgd_cal_.predict_proba(X_month_pred)[:, 1]

    tmp = pd.DataFrame({
        "date": X_month_pred.index,
        "proba_rfc_cal": rfc_proba_month,
        "proba_sgd_cal": sgd_proba_month,
        "y_true": y_month_true.values,
        "month": m_start.strftime("%Y-%m")
    }).set_index("date")
    rows.append(tmp)

    current = m_start + pd.offsets.MonthBegin(1)

oos_df = pd.concat(rows).sort_index()
oos_df = oos_df.loc["2021-01-01":"2024-12-31"]  # seguridad

oos_df.to_csv(OOS_CSV_2021_2024, float_format="%.6f")
print(f"\nArchivo generado (2021–2024): {OOS_CSV_2021_2024}")
print(oos_df.head().to_string())
print(oos_df.tail().to_string())
print("\nResumen por año (2021–2024):")
print(oos_df.groupby(oos_df.index.year)["y_true"].agg(['count','mean']))

# ============================================================
# PASO 5 — Rolling mensual 2025 con embargo y calibración
# ============================================================

print(f">> Cargando dataset 2025: {CSV_2025_SRC}")
df_2025_raw = pd.read_csv(CSV_2025_SRC)

# === 1) Alinear nombres EXACTOS a los del entrenamiento ===
# (mismo set y mismas mayúsculas/minúsculas)
train_cols = list(X_all.columns)  # features usadas en fit()
lower_to_train = {c.lower(): c for c in train_cols}

# detectar y parsear fecha
date_col = "Date" if "Date" in df_2025_raw.columns else ("date" if "date" in df_2025_raw.columns else None)
if date_col is None:
    raise ValueError("El CSV 2025 no tiene columna 'Date'/'date'.")

# renombrar para que coincidan con entrenamiento (sin forzar minúsculas)
ren = {c: (lower_to_train[c.lower()] if c.lower() in lower_to_train else c) for c in df_2025_raw.columns}
df_2025 = df_2025_raw.rename(columns=ren).copy()

# parseo fecha y orden
df_2025[date_col] = pd.to_datetime(df_2025[date_col], utc=False, errors="coerce")
df_2025 = df_2025.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

# asegurar tipos numéricos de las features de entrenamiento
for c in train_cols:
    if c in df_2025.columns:
        df_2025[c] = pd.to_numeric(df_2025[c], errors="coerce")

# verificar que estén las mismas features que vio el modelo
missing = [c for c in train_cols if c not in df_2025.columns]
if missing:
    raise ValueError(f"Faltan columnas en 2025 que el modelo espera: {missing}")

# opcional: si alguna ventana generó NaN, limpialas SOLO de las features usadas
df_2025 = df_2025.dropna(subset=train_cols)

# === 2) DataFrame de salida y columnas de probabilidades ===
df_out = df_2025.copy()
for col in ["proba_rfc_cal", "proba_sgd_cal"]:
    if col not in df_out.columns:
        df_out[col] = np.nan

# === 3) Parámetros del rolling 2025 ===
ROLL_START = pd.Timestamp("2025-01-01")
ROLL_END   = pd.Timestamp("2025-08-31")
CAL_WINDOW_DAYS_DEF = 90
CAL_WINDOW_DAYS_MIN = 60

def fit_rfc_base(X, y):
    model = RandomForestClassifier(
        **best_rfc_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def fit_sgd_base(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            **best_sgd_params,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    pipe.fit(X, y)
    return pipe

# === 4) Rolling mensual (2025-01 a 2025-08) ===
current = pd.Timestamp(ROLL_START)
rows_info = []

while current <= ROLL_END:
    m_start = pd.Timestamp(current.replace(day=1))
    m_end   = m_start + pd.offsets.MonthEnd(0)

    # Entrenamiento/calibración SOLO con histórico pre-2025
    train_end = m_start - pd.Timedelta(days=1)
    cal_end   = train_end
    cal_start = cal_end - pd.Timedelta(days=CAL_WINDOW_DAYS_DEF - 1)

    if cal_start < X_all.index.min():
        cal_start = X_all.index.min()
    if (cal_end - cal_start).days + 1 < CAL_WINDOW_DAYS_MIN:
        cal_start = cal_end - pd.Timedelta(days=CAL_WINDOW_DAYS_MIN - 1)
        if cal_start < X_all.index.min():
            cal_start = X_all.index.min()

    train_base_end = cal_start - pd.Timedelta(days=1)
    if train_base_end < X_all.index.min():
        train_base_end = train_end

    X_hist = X_all.loc[:train_base_end]
    y_hist = y_all.loc[:train_base_end]
    X_cal  = X_all.loc[cal_start:cal_end]
    y_cal  = y_all.loc[cal_start:cal_end]

    if len(X_hist) < 50 or len(X_cal) < 30:
        print(f"[{m_start.strftime('%Y-%m')}] Insuficiente histórico/calibración. Mes omitido.")
        current = m_start + pd.offsets.MonthBegin(1)
        continue

    # Embargo: saltar el día 1
    pred_start = m_start + pd.Timedelta(days=1)

    # IMPORTANTE: usar EXACTAMENTE las mismas features y en el mismo orden
    X_pred = df_out.loc[pred_start:m_end, train_cols]
    if len(X_pred) == 0:
        print(f"[{m_start.strftime('%Y-%m')}] No hay días para predecir (tras embargo). Mes omitido.")
        current = m_start + pd.offsets.MonthBegin(1)
        continue

    # Entrenar + calibrar
    rfc_base = fit_rfc_base(X_hist, y_hist)
    rfc_cal  = CalibratedClassifierCV(rfc_base, method="isotonic", cv="prefit")
    rfc_cal.fit(X_cal, y_cal)
    rfc_proba = rfc_cal.predict_proba(X_pred)[:, 1]

    sgd_base = fit_sgd_base(X_hist, y_hist)
    sgd_cal  = CalibratedClassifierCV(sgd_base, method="sigmoid", cv="prefit")
    sgd_cal.fit(X_cal, y_cal)
    sgd_proba = sgd_cal.predict_proba(X_pred)[:, 1]

    # Volcar probabilidades (solo días 2..fin)
    df_out.loc[X_pred.index, "proba_rfc_cal"] = rfc_proba
    df_out.loc[X_pred.index, "proba_sgd_cal"] = sgd_proba

    model_tag = m_start.strftime('%Y-%m')
    joblib.dump(rfc_cal, MODELS_DIR / f"rfc_2025_{model_tag}.pkl")
    joblib.dump(sgd_cal, MODELS_DIR / f"sgd_2025_{model_tag}.pkl")

    rows_info.append({
        "month": model_tag,
        "n_pred_days": int(len(X_pred)),
        "train_hist_end": str(train_base_end.date()) if hasattr(train_base_end, "date") else str(train_base_end),
        "cal_start": str(cal_start.date()) if hasattr(cal_start, "date") else str(cal_start),
        "cal_end": str(cal_end.date()) if hasattr(cal_end, "date") else str(cal_end),
    })
    print(f"[{model_tag}] OK — entrenado, calibrado y predicho {len(X_pred)} días. Modelos guardados.")

    current = m_start + pd.offsets.MonthBegin(1)

# === 5) Guardar salida final ===
df_out.to_csv(CSV_2025_WITH_PROBS, float_format="%.6f")
print(f"\nArchivo con probabilidades 2025 guardado en: {CSV_2025_WITH_PROBS}")
if rows_info:
    print(pd.DataFrame(rows_info).to_string(index=False))
