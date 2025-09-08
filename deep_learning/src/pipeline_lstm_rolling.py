# ============================================================
# pipeline.py — Rolling mensual LSTM (BiLSTM+Atención) con ventana 24m
# - Usa tu arquitectura y feature engineering actual (sent_pos únicamente)
# - Fine-tuning mensual con ventana de 24 meses hasta fin de mes (corte)
# - Val tail corta del corte para early-stopping y calibración de temperatura
# - UMBRAL FIJO para señales: 0.45 (thr fijo)
# - Versiona artefactos por corte en models/rolling/YYYY-MM-DD/
# - Predice el mes siguiente y guarda CSV por mes + agregado
# ============================================================

import json, math, shutil, hashlib, argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import joblib

# ------------------------------
# Rutas base (robustas desde /src o raíz del proyecto)
# ------------------------------
THIS_DIR  = Path(__file__).resolve().parent
BASE_DIR  = THIS_DIR if (THIS_DIR / "data").exists() and (THIS_DIR / "models").exists() else THIS_DIR.parent
DATA_DIR  = BASE_DIR / "data"
SRC_DIR   = DATA_DIR / "src"
OUT_DIR   = DATA_DIR / "out"
MODELS_DIR= BASE_DIR / "models"
ROLLING_DIR = MODELS_DIR / "rolling"
(OUT_DIR / "rolling").mkdir(parents=True, exist_ok=True)
ROLLING_DIR.mkdir(parents=True, exist_ok=True)

# Entrada por defecto: CSV maestro con 2023, 2024 y 2025
DEFAULT_INPUT = SRC_DIR / "btc_2022_2025.csv"  # ajusta si tu maestro se llama distinto
# Artefactos base existentes (pesos entrenados hasta 2023, scaler y config compatibles)
BASE_CFG  = MODELS_DIR / "lstm_config.json"
BASE_SCL  = MODELS_DIR / "lstm_scaler.pkl"
BASE_W    = MODELS_DIR / "lstm_weights.pth"

# Parámetros del rolling
ROLL_START = pd.Timestamp("2024-12-31")  # primer corte
ROLL_END   = pd.Timestamp("2025-08-31")  # último corte a generar (ajusta si tienes más datos)
WINDOW_MONTHS = 24
VAL_TAIL_DAYS = 21   # valida con ~3 semanas al final de la ventana
MAX_EPOCHS    = 10
PATIENCE      = 3
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 256
EMBARGO_DAY1  = True  # embargo día 1 en las predicciones del mes

# UMBRAL FIJO
THRESHOLD_FIXED = 0.45

# ------------------------------
# Arquitectura
# ------------------------------
class BiLSTMWithAttention(nn.Module):
    def __init__(self, n_features, d_model=64, h1=64, h2=32, attn_dim=32, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.bilstm1 = nn.LSTM(d_model, h1, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(2*h1); self.do1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(2*h1, h2, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(2*h2); self.do2 = nn.Dropout(dropout)
        self.attn = nn.Sequential(
            nn.Linear(2*h2, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2*h2, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.proj(x)
        x, _ = self.bilstm1(x); x = self.ln1(x); x = self.do1(x)
        x, _ = self.bilstm2(x); x = self.ln2(x); x = self.do2(x)
        scores  = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return self.fc(context)  # logits

# ------------------------------
# Feature engineering (idéntico a tu pipeline actual)
# ------------------------------
def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["log_close"] = np.log(df["close"])
    df["ret_1d"] = df["log_close"].diff()
    df["ret_5d"] = df["log_close"].diff(5)
    df["vol_7"]  = df["ret_1d"].rolling(7).std()
    df["vol_14"] = df["ret_1d"].rolling(14).std()
    df["mom_7"]  = df["ret_1d"].rolling(7).sum()
    df["mom_14"] = df["ret_1d"].rolling(14).sum()
    df["gap_open"]   = np.log(df["open"] / df["close"].shift(1))
    df["hl_range"]   = (df["high"] - df["low"]) / df["close"]
    df["body_rel"]   = (df["close"] - df["open"]) / df["open"]
    df["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / df["open"]
    df["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"]) / df["open"]
    df["vol_log"]    = np.log(df["volume"].replace(0, np.nan))
    df["vol_ret_1d"] = df["vol_log"].diff()
    for p in ["proba_rfc_cal","proba_sgd_cal","proba_xgb_cal"]:
        if p in df.columns:
            df[f"{p}_ema5"]  = df[p].ewm(span=5,  adjust=False).mean()
            df[f"{p}_ema10"] = df[p].ewm(span=10, adjust=False).mean()
        else:
            df[f"{p}_ema5"]  = np.nan
            df[f"{p}_ema10"] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def month_first_mask(dates_np):
    d = pd.to_datetime(dates_np)
    try:
        return (d.day == 1)
    except Exception:
        return (pd.Series(pd.to_datetime(dates_np)).dt.day == 1).to_numpy()

# ------------------------------
# Utilidades datos/secuencias
# ------------------------------
def make_sequences_from_frame(df_feat, feature_cols, window):
    X = df_feat[feature_cols].values.astype(np.float32)
    dates = df_feat["Date"].values
    X_seq, seq_dates = [], []
    for i in range(window - 1, len(X)):
        X_seq.append(X[i - window + 1 : i + 1])
        seq_dates.append(dates[i])
    X_seq = np.asarray(X_seq, dtype=np.float32)
    seq_dates = pd.to_datetime(seq_dates)
    return X_seq, seq_dates

def split_train_val(df_feat, val_tail_days):
    cutoff = df_feat["Date"].max() - pd.Timedelta(days=val_tail_days)
    tr = df_feat[df_feat["Date"] <= cutoff].copy()
    va = df_feat[df_feat["Date"] >  cutoff].copy()
    return tr, va

def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
    best_T, best_metric = 1.0, -1.0
    y = y.astype(int)
    for T in grid:
        proba = 1.0/(1.0 + np.exp(-logits / max(T,1e-4)))
        try:
            m = roc_auc_score(y, proba)
        except Exception:
            pred = (proba >= 0.5).astype(int)
            m = f1_score(y, pred, zero_division=0)
        if m > best_metric:
            best_metric, best_T = m, T
    return float(best_T)

# ------------------------------
# Dataset y entrenamiento
# ------------------------------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq, y):
        self.X = X_seq
        self.y = y.reshape(-1,1).astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_one_cut(model, train_loader, val_loader, device, max_epochs=10, patience=3, lr=3e-4, wd=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val = -1e9
    best_state = None
    wait = 0
    for epoch in range(1, max_epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
        # validación simple: AUC con logits
        model.eval()
        with torch.no_grad():
            logits_list, y_list = [], []
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                lg = model(xb)
                logits_list.append(lg.cpu().numpy()); y_list.append(yb.cpu().numpy())
            logits_val = np.vstack(logits_list).ravel()
            y_val = np.vstack(y_list).ravel()
            try:
                auc = roc_auc_score(y_val, 1.0/(1.0+np.exp(-logits_val)))
            except Exception:
                pred = (logits_val >= 0).astype(int)
                auc = f1_score(y_val, pred, zero_division=0)
        if auc > best_val:
            best_val = auc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return float(best_val)

# ------------------------------
# Roll mensual
# ------------------------------
def load_base_config():
    if not BASE_CFG.exists(): raise FileNotFoundError(f"No se encontró {BASE_CFG}")
    if not BASE_SCL.exists(): raise FileNotFoundError(f"No se encontró {BASE_SCL}")
    if not BASE_W.exists():   raise FileNotFoundError(f"No se encontró {BASE_W}")
    with open(BASE_CFG, "r") as f:
        cfg = json.load(f)
    # asegurar feature_cols (compatibles con tus pesos)
    feature_cols = cfg.get("feature_cols")
    if feature_cols is None:
        feature_cols = cfg["tech_cols"] + cfg["eng_cols"] + cfg["proba_cols"] + cfg["sent_cols"]
    window = int(cfg.get("window_size", 5))
    arch = dict(
        d_model=int(cfg.get("d_model", 64)),
        h1=int(cfg.get("h1", 64)),
        h2=int(cfg.get("h2", 32)),
        attn_dim=int(cfg.get("attn_dim", 32)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    temperature = float(cfg.get("temperature", 1.0))
    return cfg, feature_cols, window, arch, temperature

def ensure_index_csv():
    idx = ROLLING_DIR / "index.csv"
    if not idx.exists():
        pd.DataFrame(columns=[
            "cut_date","train_start","train_end","val_start","val_end",
            "n_epochs","val_metric","best_threshold","best_threshold_f1",
            "weights_path","scaler_path","config_path","notes"
        ]).to_csv(idx, index=False)
    return idx

def append_index_row(row: dict):
    idx = ensure_index_csv()
    df = pd.read_csv(idx)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(idx, index=False)

def hash_series(s: pd.Series):
    m = hashlib.md5()
    m.update(pd.util.hash_pandas_object(s, index=False).values.tobytes())
    return m.hexdigest()

def run_rolling():
    # Cargar datos maestro
    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(f"No existe el CSV maestro: {DEFAULT_INPUT}")
    df = pd.read_csv(DEFAULT_INPUT)
    required = [
        "Date","close","high","low","open","volume",
        "RSI","MACD","MACD_SIGNAL","SMA20","EMA20",
        "BB_UPPER","BB_LOWER","ATR","CCI",
        "proba_rfc_cal","proba_sgd_cal","proba_xgb_cal",
        "proba_sentiment_neg","proba_sentiment_neu","proba_sentiment_pos",
        "y_true"
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas requeridas: {miss}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Feature engineering global (para no recalcular mil veces)
    df_feat_all = add_engineered_features(df)

    # Config base y pesos iniciales
    base_cfg, feature_cols, window, arch, base_temperature = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparar agregados de predicciones
    preds_all = []

    # Cortes mensuales
    cut = ROLL_START
    while cut <= ROLL_END:
        # Ventana 24m hasta cut (incl.) y split train/val
        start_win = (cut - pd.DateOffset(months=WINDOW_MONTHS)).normalize() + pd.offsets.Day(0)
        df_cut = df_feat_all[(df_feat_all["Date"] >= start_win) & (df_feat_all["Date"] <= cut)].copy()
        if df_cut["Date"].nunique() < (window + VAL_TAIL_DAYS + 10):
            print(f"[{cut.date()}] Ventana insuficiente, salto.")
            cut = (cut + pd.offsets.MonthEnd(1))
            continue

        # Limpieza focalizada
        needed = list(set(feature_cols + ["Date","close","open","high","low","volume","y_true"]))
        df_cut = df_cut.dropna(subset=[c for c in needed if c in df_cut.columns]).copy()
        for c in feature_cols:
            df_cut[c] = pd.to_numeric(df_cut[c], errors="coerce")
        df_cut = df_cut.dropna(subset=feature_cols)

        # split train/val
        tr_df, va_df = split_train_val(df_cut, VAL_TAIL_DAYS)
        if len(tr_df) < window*2 or len(va_df) < window+5:
            print(f"[{cut.date()}] Muy pocos datos para train/val, salto.")
            cut = (cut + pd.offsets.MonthEnd(1))
            continue

        # Scaler fit SOLO con train
        scaler = StandardScaler()
        scaler.fit(tr_df[feature_cols].values.astype(np.float32))

        # Estandarización vectorizada
        tr_X_std = scaler.transform(tr_df[feature_cols].values.astype(np.float32))
        va_X_std = scaler.transform(va_df[feature_cols].values.astype(np.float32))
        tr_std = tr_df.copy(); tr_std[feature_cols] = tr_X_std
        va_std = va_df.copy(); va_std[feature_cols] = va_X_std

        Xtr, dtr = make_sequences_from_frame(tr_std, feature_cols, window)
        Xva, dva = make_sequences_from_frame(va_std, feature_cols, window)
        ytr = tr_std["y_true"].values[window-1:].astype(np.float32)
        yva = va_std["y_true"].values[window-1:].astype(np.float32)

        # Modelo
        n_features = Xtr.shape[2]
        model = BiLSTMWithAttention(n_features=n_features, **arch).to(device)

        # Pesos iniciales: base en el primer corte; luego el del corte anterior
        prev_cut_dir = (ROLLING_DIR / (cut - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"))
        if cut == ROLL_START:
            state = torch.load(BASE_W, map_location=device)
        elif (prev_cut_dir / "lstm_weights.pth").exists():
            state = torch.load(prev_cut_dir / "lstm_weights.pth", map_location=device)
        else:
            state = torch.load(BASE_W, map_location=device)
        model.load_state_dict(state, strict=True)

        # Congelar primera BiLSTM para estabilidad
        for p in model.bilstm1.parameters():
            p.requires_grad = False

        # DataLoaders
        tr_loader = torch.utils.data.DataLoader(SeqDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        va_loader = torch.utils.data.DataLoader(SeqDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        # Entrenamiento
        best_val = train_one_cut(
            model, tr_loader, va_loader, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, wd=WEIGHT_DECAY
        )

        # Calibración de temperatura (sobre validación)
        model.eval()
        with torch.no_grad():
            logits_val = []
            for xb, _ in va_loader:
                logits_val.append(model(xb.to(device)).cpu().numpy())
            logits_val = np.vstack(logits_val).ravel()
        T_star = temperature_scale(logits_val, yva)

        # Guardar artefactos del corte
        cut_dir = ROLLING_DIR / cut.strftime("%Y-%m-%d")
        cut_dir.mkdir(parents=True, exist_ok=True)
        # scaler
        joblib.dump(scaler, cut_dir / "lstm_scaler.pkl")
        # pesos
        torch.save(model.state_dict(), cut_dir / "lstm_weights.pth")
        # config del corte (incluye umbral fijo)
        cfg_cut = dict(base_config_path=str(BASE_CFG),
                       feature_cols=feature_cols,
                       window_size=window,
                       d_model=arch["d_model"], h1=arch["h1"], h2=arch["h2"],
                       attn_dim=arch["attn_dim"], dropout=arch["dropout"],
                       temperature=T_star,
                       threshold=THRESHOLD_FIXED,
                       threshold_mode="fixed")
        with open(cut_dir / "lstm_config.json", "w") as f:
            json.dump(cfg_cut, f, indent=2)
        # metrics y meta
        metrics = dict(val_metric=best_val,
                       threshold_used=THRESHOLD_FIXED,
                       threshold_mode="fixed")
        with open(cut_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        meta = dict(train_start=str(tr_df["Date"].min().date()),
                    train_end=str(tr_df["Date"].max().date()),
                    val_start=str(va_df["Date"].min().date()),
                    val_end=str(va_df["Date"].max().date()),
                    cut_date=str(cut.date()),
                    window_months=WINDOW_MONTHS,
                    val_tail_days=VAL_TAIL_DAYS,
                    lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE)
        with open(cut_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # index.csv (mantiene columnas esperadas; se registra el umbral fijo)
        append_index_row(dict(
            cut_date=str(cut.date()),
            train_start=meta["train_start"],
            train_end=meta["train_end"],
            val_start=meta["val_start"],
            val_end=meta["val_end"],
            n_epochs=MAX_EPOCHS,
            val_metric=best_val,
            best_threshold=THRESHOLD_FIXED,
            best_threshold_f1=np.nan,
            weights_path=str((cut_dir / "lstm_weights.pth").relative_to(BASE_DIR)),
            scaler_path=str((cut_dir / "lstm_scaler.pkl").relative_to(BASE_DIR)),
            config_path=str((cut_dir / "lstm_config.json").relative_to(BASE_DIR)),
            notes="threshold_mode=fixed"
        ))

        # Predicción del mes siguiente
        next_month_start = (cut + pd.offsets.Day(1)).normalize()
        next_month_end   = (cut + pd.offsets.MonthEnd(1))
        df_pred = df_feat_all[(df_feat_all["Date"] >= next_month_start) & (df_feat_all["Date"] <= next_month_end)].copy()
        if len(df_pred) > 0:
            # limpieza y std con scaler del corte
            df_pred = df_pred.dropna(subset=[c for c in feature_cols + ["Date"] if c in df_pred.columns]).copy()
            for c in feature_cols:
                df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")
            df_pred = df_pred.dropna(subset=feature_cols)
            Xp_std  = scaler.transform(df_pred[feature_cols].values.astype(np.float32))
            df_pred_std = df_pred.copy(); df_pred_std[feature_cols] = Xp_std
            Xp, dp = make_sequences_from_frame(df_pred_std, feature_cols, window)
            if len(Xp) > 0:
                with torch.no_grad():
                    logits_p = []
                    for i in range(0, len(Xp), 1024):
                        xb = torch.from_numpy(Xp[i:i+1024]).to(device)
                        logits_p.append(model(xb).cpu().numpy())
                    logits_p = np.vstack(logits_p).ravel()
                proba_p = 1.0/(1.0 + np.exp(-logits_p / max(T_star,1e-4)))
                out = pd.DataFrame({"Date": dp, "proba_lstm": proba_p})
                if "y_true" in df_pred.columns:
                    out = out.merge(df_pred[["Date","y_true"]], on="Date", how="left")
                # embargo día 1
                if EMBARGO_DAY1:
                    mask_first = month_first_mask(out["Date"].values)
                    out.loc[mask_first, "proba_lstm"] = np.nan
                # señales con umbral FIJO
                out["signal"] = (out["proba_lstm"] >= THRESHOLD_FIXED).astype(int)
                # guardar mensuales
                out_path = OUT_DIR / "rolling" / f"preds_{next_month_start.strftime('%Y-%m')}.csv"
                out.sort_values("Date").to_csv(out_path, index=False, float_format="%.6f")
                preds_all.append(out.assign(cut_date=cut.strftime("%Y-%m-%d"),
                                            thr_used=THRESHOLD_FIXED, T_used=T_star))
                print(f"[{cut.date()}] Guardado {out_path}, filas: {len(out)}")

        # siguiente corte
        cut = (cut + pd.offsets.MonthEnd(1))

    # Agregado final 2025
    if preds_all:
        full = pd.concat(preds_all, ignore_index=True).sort_values("Date")
        full.to_csv(OUT_DIR / "rolling" / "preds_2025_full.csv", index=False, float_format="%.6f")
        print(f">> Guardado agregado: {OUT_DIR / 'rolling' / 'preds_2025_full.csv'}")
    else:
        print("No se generaron predicciones mensuales.")

# ------------------------------
# Entrypoint
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rolling mensual LSTM, ventana 24m, fine-tune y umbral fijo=0.45")
    # sin argumentos obligatorios; todo por defecto
    args = parser.parse_args()
    run_rolling()

if __name__ == "__main__":
    main()
