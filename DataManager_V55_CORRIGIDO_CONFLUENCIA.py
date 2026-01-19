#!/usr/bin/env python3
# ============================================================
# DataManager_V55_PENDLE.py
# PENDLEUSDT – Binance Data Vision + TREINO V27 EXATO
# 
# COMPONENTES:
# 1. Download aggTrades (COPIADO DO V51)
# 2. Geração de timeframes 15m,30m,1h,4h,8h,1d (COPIADO DO V51)
# 3. Feature Engine (COPIADO DO V27)
# 4. Treino K1-K6 (COPIADO DO V27)
# 5. Salvamento scaler, kmeans, PKLs (COPIADO DO V27)
# 6. Gravação em /data (persistente Render)
# 7. Export ZIPs + CatBox
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

warnings.filterwarnings("ignore")

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# IMPORTS ML (COPIADO DO V27)
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
SYMBOL = "PENDLEUSDT"

# ⚠️ TESTE: 1 ANO
START_DT = datetime(2025, 12, 1, 0, 0, 0)
END_DT = datetime(2025, 12, 31, 23, 59, 59)

# 🔴 PRODUÇÃO: 1 ANO (descomentar)
# START_DT = datetime(2025, 1, 1, 0, 0, 0)
# END_DT = datetime(2025, 12, 31, 23, 59, 59)

# =============================================================================
# PENDLE PERSISTENT DISK - CORREÇÃO COMPLETA
# =============================================================================
# Primeiro tenta criar /data se não existir (Render permite)
try:
    os.makedirs("/data", exist_ok=True)
except:
    pass

# Verifica se /data existe E é gravável
def check_disk_writable(path):
    try:
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except:
        return False

if os.path.exists("/data") and check_disk_writable("/data"):
    BASE_DISK = "/data"
    print("✅ Usando disco persistente: /data", flush=True)
elif os.path.exists("/opt/render/project") and check_disk_writable("/opt/render/project"):
    BASE_DISK = "/opt/render/project"
    print("⚠️ Usando /opt/render/project (pode ser apagado!)", flush=True)
else:
    BASE_DISK = "."
    print("⚠️ Usando diretório local", flush=True)

FOLDER_NAME = f"pendle_agg_{START_DT.strftime('%Y%m%d')}_a_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DISK, FOLDER_NAME)
MODELOS_DIR = os.path.join(OUT_DIR, "modelos_salvos")

# Criar diretórios com verificação
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# Verificar se criou corretamente
if not os.path.exists(OUT_DIR):
    print(f"❌ ERRO: Não conseguiu criar {OUT_DIR}", flush=True)
    sys.exit(1)
if not os.path.exists(MODELOS_DIR):
    print(f"❌ ERRO: Não conseguiu criar {MODELOS_DIR}", flush=True)
    sys.exit(1)

print(f"📂 OUT_DIR: {OUT_DIR}", flush=True)
print(f"📂 MODELOS_DIR: {MODELOS_DIR}", flush=True)

# Listar arquivos existentes no disco (para debug)
try:
    existing_files = os.listdir(OUT_DIR)
    if existing_files:
        print(f"📁 Arquivos existentes: {existing_files}", flush=True)
except:
    pass

CSV_AGG_PATH = os.path.join(OUT_DIR, f"{SYMBOL}_aggTrades.csv")
ZIP_CSV_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_CSVs.zip")
ZIP_PKL_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_PKLs.zip")

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# =============================================================================
# V27 CONFIG - HORIZONTE E KS
# =============================================================================
HORIZONTE_BACKTEST = 6  # Treina K1 até K6
KS = [1, 2, 3, 4, 5]  # K6 usa threshold especial

# =============================================================================
# CACHES GLOBAIS (COPIADO DO V27)
# =============================================================================
modelos_cache = {}
caminhos_modelos = {}
features_por_target = {}

# =============================================================================
# FUNÇÕES DE DOWNLOAD (COPIADO DO V51)
# =============================================================================
def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }

def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                wait = min(5 * (2 ** attempt), 60)
                time.sleep(wait)
            
            response = session.get(url, headers=get_headers(), timeout=90)
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    files = z.namelist()
                    if not files:
                        return None
                    
                    csv_filename = files[0]
                    
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any('transact_time' in str(val) for val in df_test.iloc[0])
                    
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=0 if has_header else None)
                        return df
            
            elif response.status_code == 404:
                return None
            
            elif response.status_code in [418, 429]:
                continue
            
            else:
                continue
        
        except Exception:
            if attempt == retry_count - 1:
                return None
            continue
    
    return None

def process_binance_data(df):
    if df is None or df.empty:
        return None
    
    if 'transact_time' not in df.columns:
        df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                      'last_trade_id', 'transact_time', 'is_buyer_maker']
    
    def convert_side(val):
        return 1 if (val is True or val == 'True' or val == 'true') else 0
    
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('Int64'),
        'price': pd.to_numeric(df['price'], errors='coerce').astype(float),
        'qty': pd.to_numeric(df['quantity'], errors='coerce').astype(float),
        'side': df['is_buyer_maker'].apply(convert_side)
    })
    
    return df_processed.dropna()

# =============================================================================
# GERAÇÃO DE TIMEFRAMES (CÓPIA EXATA DO V51 + OTIMIZAÇÃO MEMÓRIA)
# =============================================================================
def gerar_timeframe_tratado(csv_agg_path, csv_tf_path, timeframe_min=15, min_val_usd=500, chunksize=100_000):
    """
    Converte o CSV de aggTrades (ts,price,qty,side) para um dataset de timeframe.
    
    CÓPIA EXATA DO V51 - MESMA LÓGICA:
    - bucket = floor(datetime UTC, Xmin)
    - OHLCV do preço/qty
    - baleias: val_usd = price*qty >= min_val_usd (500 USD - IGUAL V51!)
      buy_vol: soma qty das baleias com side == 0
      sell_vol: soma qty das baleias com side == 1
      delta = buy_vol - sell_vol
    
    OTIMIZAÇÃO: gc.collect() periódico para evitar estouro de memória
    """
    import gc
    
    print(f">>> Gerando {timeframe_min}m tratado (lógica V51 exata)...", flush=True)
    print(f"    Filtro baleia: >= ${min_val_usd} USD", flush=True)

    # Estrutura incremental por bucket (IGUAL V51)
    buckets = {}
    chunks_processados = 0

    # Lê em chunks (IGUAL V51)
    for chunk in pd.read_csv(csv_agg_path, chunksize=chunksize):
        # Garante tipos (IGUAL V51)
        chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")
        chunk["side"] = pd.to_numeric(chunk["side"], errors="coerce")

        chunk = chunk.dropna(subset=["ts", "price", "qty", "side"])
        if chunk.empty:
            continue

        # Bucket UTC (IGUAL V51)
        dt = pd.to_datetime(chunk["ts"].astype("int64"), unit="ms", utc=True)
        bucket_dt = dt.dt.floor(f"{timeframe_min}min")
        bucket_ms = (bucket_dt.astype("int64") // 10**6).astype("int64")
        chunk = chunk.assign(bucket_ms=bucket_ms)

        # Whale flag (IGUAL V51)
        val_usd = chunk["price"] * chunk["qty"]
        is_whale = val_usd >= float(min_val_usd)

        # Itera em linhas (IGUAL V51 - mantém ordem cronológica!)
        for ts_ms, price, qty, side, bms, whale in zip(
            chunk["ts"].astype("int64"),
            chunk["price"].astype(float),
            chunk["qty"].astype(float),
            chunk["side"].astype(int),
            chunk["bucket_ms"].astype("int64"),
            is_whale.astype(bool),
        ):
            st = buckets.get(bms)
            if st is None:
                st = {
                    "ts": int(bms),
                    "open": float(price),
                    "high": float(price),
                    "low": float(price),
                    "close": float(price),
                    "volume": 0.0,
                    "buy_vol": 0.0,
                    "sell_vol": 0.0,
                    "trades": 0,  # NOVO: Contador de trades
                }
                buckets[bms] = st
            else:
                # OHLC (IGUAL V51)
                if price > st["high"]:
                    st["high"] = float(price)
                if price < st["low"]:
                    st["low"] = float(price)
                st["close"] = float(price)

            # Volume total (IGUAL V51)
            st["volume"] += float(qty)
            
            # Contador de trades (NOVO)
            st["trades"] += 1

            # Baleias (IGUAL V51)
            if whale:
                if side == 0:
                    st["buy_vol"] += float(qty)
                else:
                    st["sell_vol"] += float(qty)
        
        # OTIMIZAÇÃO MEMÓRIA: Limpar chunk e gc periódico
        chunks_processados += 1
        del chunk
        
        if chunks_processados % 10 == 0:
            gc.collect()
            print(f"    Chunks: {chunks_processados}, Buckets: {len(buckets)}", flush=True)

    # Libera memória antes de montar DataFrame
    gc.collect()

    if not buckets:
        raise RuntimeError(f"Nenhum bucket {timeframe_min}m gerado!")

    print(f"    Montando DataFrame com {len(buckets)} buckets...", flush=True)

    # Ordena por tempo e monta rows (IGUAL V51 + trades)
    rows = []
    for bms in sorted(buckets.keys()):
        st = buckets[bms]
        rows.append([
            st["ts"],
            st["open"],
            st["high"],
            st["low"],
            st["close"],
            st["volume"],
            st["buy_vol"],
            st["sell_vol"],
            st["buy_vol"] - st["sell_vol"],  # delta
            st["trades"],  # NOVO
        ])

    # Libera buckets
    del buckets
    gc.collect()

    # DataFrame (IGUAL V51 + trades)
    df_tf = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close",
        "volume", "buy_vol", "sell_vol", "delta", "trades"
    ])
    del rows
    gc.collect()

    # ============================================================
    # ENRIQUECIMENTO V1 (IGUAL V51 - EXATO)
    # ============================================================
    
    # Aliases compatíveis com V1
    df_tf["buy_vol_agg"]   = df_tf["buy_vol"]
    df_tf["sell_vol_agg"]  = df_tf["sell_vol"]
    df_tf["total_vol_agg"] = df_tf["buy_vol_agg"] + df_tf["sell_vol_agg"]

    df_tf["taker_buy_base"]  = df_tf["buy_vol_agg"]
    df_tf["taker_sell_base"] = df_tf["sell_vol_agg"]
    df_tf["taker_buy_quote"] = df_tf["taker_buy_base"] * df_tf["close"]

    # Campos V1 comuns
    df_tf["quote_volume"] = df_tf["volume"] * df_tf["close"]
    # trades já está preenchido com contagem real!
    df_tf["close_time"] = df_tf["ts"] + (timeframe_min * 60 * 1000) - 1

    # Métricas adicionais (IGUAL V51)
    df_tf = df_tf.sort_values("ts").reset_index(drop=True)
    df_tf["cum_delta"] = df_tf["delta"].cumsum()

    df_tf["price_range"] = df_tf["high"] - df_tf["low"]
    df_tf["absorcao"] = df_tf["delta"] / (df_tf["price_range"].replace(0, 1e-9))
    df_tf["vpin"] = (df_tf["buy_vol_agg"] - df_tf["sell_vol_agg"]).abs() / (
        df_tf["total_vol_agg"].replace(0, 1e-9)
    )

    # Saneamento (IGUAL V51)
    num_cols = [
        "open","high","low","close",
        "volume","buy_vol","sell_vol","delta",
        "buy_vol_agg","sell_vol_agg","total_vol_agg",
        "taker_buy_base","taker_sell_base","taker_buy_quote",
        "quote_volume","cum_delta","price_range","absorcao","vpin"
    ]
    for c in num_cols:
        if c in df_tf.columns:
            df_tf[c] = pd.to_numeric(df_tf[c], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    df_tf["ts"] = pd.to_numeric(df_tf["ts"], errors="coerce").fillna(0).astype("int64")
    df_tf["trades"] = pd.to_numeric(df_tf["trades"], errors="coerce").fillna(0).astype("int64")
    df_tf["close_time"] = pd.to_numeric(df_tf["close_time"], errors="coerce").fillna(0).astype("int64")

    # Ordem V1 Final
    cols_v1 = [
        "ts",
        "open","high","low","close",
        "volume","quote_volume","trades",
        "taker_buy_base","taker_sell_base","taker_buy_quote",
        "buy_vol_agg","sell_vol_agg","total_vol_agg",
        "delta","cum_delta",
        "close_time",
        "vpin","price_range","absorcao"
    ]
    for c in cols_v1:
        if c not in df_tf.columns:
            df_tf[c] = 0.0

    df_tf = df_tf[cols_v1]
    df_tf.to_csv(csv_tf_path, index=False)
    
    # fsync original
    try:
        with open(csv_tf_path, 'rb') as f:
            os.fsync(f.fileno())
    except:
        pass

    print(f"    ✅ {len(df_tf)} candles → {csv_tf_path}", flush=True)
    return df_tf
# =============================================================================
# FEATURE ENGINE (COPIADO EXATO DO V27)
# =============================================================================
def realized_vol(close: pd.Series) -> pd.Series:
    """Volatilidade Realizada protegida contra o futuro."""
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean()).shift(1)

def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """Parkinson protegida com shift(1)."""
    return np.sqrt((1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()).shift(1)

def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell protegida com shift(1)."""
    rs = (np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
          np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"]))
    return np.sqrt(rs.rolling(20).mean()).shift(1)

def yang_zhang(df: pd.DataFrame) -> pd.Series:
    """Yang-Zhang blindada contra leakage."""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))
    log_co = np.log(df["close"] / df["open"])
    rs = (log_ho**2 + log_lo**2).rolling(20).mean()
    close_vol = log_co.rolling(20).std() ** 2
    open_vol = log_oc.rolling(20).std() ** 2
    return np.sqrt(0.34 * open_vol + 0.34 * close_vol + 0.27 * rs).shift(1)

def slope_regression(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Inclinação da regressão linear isolada no passado."""
    X = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * window
    for i in range(window, len(series)):
        y = series[i-window:i]
        slopes.append(LinearRegression().fit(X, y).coef_[0])
    return pd.Series(slopes).shift(1).values

def adicionar_fractais_elliott(df):
    df = df.copy()

    df["fractal_high"] = (
        (df["high"].shift(2) < df["high"].shift(1)) &
        (df["high"].shift(1) < df["high"]) &
        (df["high"].shift(-1) < df["high"]) &
        (df["high"].shift(-2) < df["high"])
    ).astype(int)

    df["fractal_low"] = (
        (df["low"].shift(2) > df["low"].shift(1)) &
        (df["low"].shift(1) > df["low"]) &
        (df["low"].shift(-1) > df["low"]) &
        (df["low"].shift(-2) > df["low"])
    ).astype(int)

    df["pivot_high"] = df["high"].where(
        (df["high"] > df["high"].shift(1)) &
        (df["high"] > df["high"].shift(-1))
    )
    df["pivot_low"] = df["low"].where(
        (df["low"] < df["low"].shift(1)) &
        (df["low"] < df["low"].shift(-1))
    )

    df["last_pivot_high"] = df["pivot_high"].ffill()
    df["last_pivot_low"] = df["pivot_low"].ffill()

    df["swing_dir"] = 0
    df.loc[df["last_pivot_high"] > df["last_pivot_high"].shift(1), "swing_dir"] = 1
    df.loc[df["last_pivot_low"]  > df["last_pivot_low"].shift(1),  "swing_dir"] = 2
    df.loc[df["last_pivot_high"] < df["last_pivot_high"].shift(1), "swing_dir"] = -1
    df.loc[df["last_pivot_low"]  < df["last_pivot_low"].shift(1),  "swing_dir"] = -2

    df["wave_amplitude"] = df["close"] - df["close"].shift(5)
    df["wave_amplitude_abs"] = df["wave_amplitude"].abs()

    df["correction_pct"] = (
        (df["wave_amplitude"].shift(1) - df["wave_amplitude"]) /
        (df["wave_amplitude"].shift(1).abs() + 1e-9)
    )

    if "atr14" in df.columns:
        df["wave_strength"] = df["wave_amplitude_abs"] / (df["atr14"] + 1e-9)
    else:
        df["wave_strength"] = np.nan

    df["dist_to_pivot_high"] = df["last_pivot_high"] - df["close"]
    df["dist_to_pivot_low"] = df["close"] - df["last_pivot_low"]

    df["impulse_count"] = (df["swing_dir"] != df["swing_dir"].shift(1)).cumsum()

    return df


def adicionar_vwap(df):
    df = df.copy().reset_index(drop=True)

    if "ts" not in df.columns:
        print("[AVISO] Coluna 'ts' ausente — VWAP ignorado.")
        return df

    dt = pd.to_datetime(df["ts"], unit="ms")

    df["day"] = dt.dt.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

    vol_day = df["volume"].groupby(df["day"]).cumsum()
    tpv_day = (df["tp"] * df["volume"]).groupby(df["day"]).cumsum()
    df["vwap"] = tpv_day / (vol_day + 1e-9)

    df["week"] = dt.dt.isocalendar().week.astype(int)
    vol_week = df["volume"].groupby(df["week"]).cumsum()
    tpv_week = (df["tp"] * df["volume"]).groupby(df["week"]).cumsum()
    df["vwap_week"] = tpv_week / (vol_week + 1e-9)

    df["month"] = dt.dt.month.astype(int)
    vol_month = df["volume"].groupby(df["month"]).cumsum()
    tpv_month = (df["tp"] * df["volume"]).groupby(df["month"]).cumsum()
    df["vwap_month"] = tpv_month / (vol_month + 1e-9)

    if "pivot_low" in df.columns and "pivot_high" in df.columns:
        pivot_anchor = df["pivot_low"].combine_first(df["pivot_high"]).fillna(0)
        df["anchored_vwap"] = np.nan
        sum_tp = 0.0
        sum_vol = 0.0

        for i in range(len(df)):
            if pivot_anchor.iloc[i] != 0:
                sum_tp = 0.0
                sum_vol = 0.0
            sum_tp += df["tp"].iloc[i] * df["volume"].iloc[i]
            sum_vol += df["volume"].iloc[i]
            df.at[i, "anchored_vwap"] = sum_tp / (sum_vol + 1e-9)
    else:
        df["anchored_vwap"] = np.nan

    df["vwap_std"] = df["close"].rolling(20).std()
    df["vwap_upper"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower"] = df["vwap"] - df["vwap_std"]
    df["vwap_zscore"] = (df["close"] - df["vwap"]) / (df["vwap_std"] + 1e-9)

    if "atr14" in df.columns:
        df["dist_vwap_atr"] = (df["close"] - df["vwap"]) / (df["atr14"] + 1e-9)
    else:
        df["dist_vwap_atr"] = np.nan

    return df


def adicionar_micro_squeeze(df):
    df = df.copy()

    ma_5 = df["close"].rolling(5).mean()
    std_5 = df["close"].rolling(5).std()

    df["bb_upper_5"] = ma_5 + 2 * std_5
    df["bb_lower_5"] = ma_5 - 2 * std_5
    df["bb_width_5"] = (df["bb_upper_5"] - df["bb_lower_5"]) / (df["close"] + 1e-9)

    if "tr" in df.columns:
        atr_5 = df["tr"].rolling(5).mean()
    else:
        atr_5 = (df["high"] - df["low"]).rolling(5).mean()

    df["kc_upper_5"] = ma_5 + 1.5 * atr_5
    df["kc_lower_5"] = ma_5 - 1.5 * atr_5
    df["kc_range_5"] = (df["kc_upper_5"] - df["kc_lower_5"]) / (df["close"] + 1e-9)

    df["micro_squeeze"] = (
        (df["bb_upper_5"] < df["kc_upper_5"]) &
        (df["bb_lower_5"] > df["kc_lower_5"])
    ).astype(int)

    df["pre_breakout_pressure"] = (
        (df["close"] - df["bb_upper_5"]) /
        (df["bb_upper_5"] - df["bb_lower_5"] + 1e-9)
    )

    df["micro_vol_squeeze"] = (
        df["range"].rolling(5).std() /
        (df["range"].rolling(20).std() + 1e-9)
    )

    return df


def adicionar_inside_nr(df):
    df = df.copy()

    df["inside_bar"] = (
        (df["high"] < df["high"].shift(1)) &
        (df["low"]  > df["low"].shift(1))
    ).astype(int)

    prev_range = (df["high"].shift(1) - df["low"].shift(1))
    df["inside_bar_strength"] = (df["high"] - df["low"]) / (prev_range + 1e-9)

    df["range"] = df["high"] - df["low"]
    df["nr4"] = (df["range"] == df["range"].rolling(4).min()).astype(int)
    df["nr7"] = (df["range"] == df["range"].rolling(7).min()).astype(int)

    df["breakout_bias"] = (
        (df["close"] - df["low"]) /
        (df["high"] - df["low"] + 1e-9)
    )

    return df


def adicionar_zscore_intrabar(df):
    df = df.copy()

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df["body_z"] = (df["body"] - df["body"].rolling(10).mean()) / \
                   (df["body"].rolling(10).std() + 1e-9)

    df["range_z"] = (df["range"] - df["range"].rolling(10).mean()) / \
                    (df["range"].rolling(10).std() + 1e-9)

    df["close_pos"] = (df["close"] - df["low"]) / \
                      (df["high"] - df["low"] + 1e-9)

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["wick_up_ratio_5"] = upper_wick.rolling(5).mean() / (df["range"] + 1e-9)
    df["wick_down_ratio_5"] = lower_wick.rolling(5).mean() / (df["range"] + 1e-9)

    df["reversal_prob"] = (
        (-df["body_z"] * 0.5) +
        (df["wick_down_ratio_5"] * 0.7) -
        (df["wick_up_ratio_5"] * 0.7)
    )

    return df

# NOTA: As funções realized_vol, parkinson_vol, rogers_satchell, yang_zhang, slope_regression
# já estão definidas acima (linhas 436-468). Removidas duplicatas.

def feature_engine(df):
    """Pipeline de Geração de Features - Versão Auditada V25."""
    df = df.copy()

    # 1. VARIÁVEIS BÁSICAS (Isoladas do candle de decisão)
    df["body"] = (df["close"] - df["open"]).shift(1)
    df["range"] = (df["high"] - df["low"]).shift(1)
    
    # 2. RETORNOS PROTEGIDOS (Vazamento Crítico aqui)
    df["ret1"] = df["close"].pct_change(1).shift(1)
    df["ret5"] = df["close"].pct_change(5).shift(1)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).shift(1)

    # 3. EMAs E DISTÂNCIAS (Devem olhar para T-1)
    df["ema9"] = df["close"].ewm(span=9).mean().shift(1)
    df["ema20"] = df["close"].ewm(span=20).mean().shift(1)
    df["dist_ema9"] = (df["close"].shift(1) - df["ema9"])
    df["dist_ema20"] = (df["close"].shift(1) - df["ema20"])

    # 4. SLOPES E VOLATILIDADES AUDITADAS
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)

    # 5. AGRESSÃO PROTEGIDA
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["taker_buy_base"]).shift(1)  # ← CORREÇÃO ADICIONADA
        df["aggression_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).shift(1)
    
    # 6. RSI 14 (ADICIONADO - FALTAVA!)
    delta_price = df["close"].diff()
    gain = delta_price.where(delta_price > 0, 0).rolling(14).mean()
    loss = (-delta_price.where(delta_price < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    return df

# =====================================================================
# FEATURES AVANÇADAS (LIMPO)
# =====================================================================

def adicionar_features_avancadas(df):
    # ==================================================================
    # 🛡️ BLINDAGEM TOTAL: GARANTIA DE INTEGRIDADE DE DADOS
    # ==================================================================
    df = df.copy()

    # 1. PRICE ACTION BÁSICO (Resolve 'upper_wick', 'range', 'body')
    if "range" not in df.columns: df["range"] = df["high"] - df["low"]
    if "body" not in df.columns: df["body"] = df["close"] - df["open"]
    if "upper_wick" not in df.columns: 
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    if "lower_wick" not in df.columns: 
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # 2. PERCENTUAIS (Resolve o erro ATUAL 'range_pct')
    if "range_pct" not in df.columns: 
        # Adicionamos 1e-9 para evitar divisão por zero
        df["range_pct"] = df["range"] / (df["close"] + 1e-9)

    # 3. RETORNOS (Resolve erros futuros de 'ret2', 'ret5', etc.)
    if "ret1" not in df.columns: df["ret1"] = df["close"].pct_change(1)
    if "ret2" not in df.columns: df["ret2"] = df["close"].pct_change(2)
    if "ret5" not in df.columns: df["ret5"] = df["close"].pct_change(5)
    if "ret10" not in df.columns: df["ret10"] = df["close"].pct_change(10)

    # 4. INDICADORES TÉCNICOS (Resolve erros de normalização)
    if "ema20" not in df.columns: 
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    if "dist_ema20" not in df.columns: 
        df["dist_ema20"] = df["close"] - df["ema20"]
    if "atr14" not in df.columns:
        tr = pd.concat([
            (df["high"] - df["low"]), 
            (df["high"] - df["close"].shift(1)).abs(), 
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
    # ==================================================================

    # ---> AQUI COMEÇA O SEU CÓDIGO ORIGINAL (Mantenha tudo abaixo) <---
    # Agora a linha 1092 vai encontrar 'range_pct' e funcionar:
    if "volatility_acceleration" not in df.columns: # Proteção extra
         df["volatility_acceleration"] = df["range_pct"].diff()

    # CORREÇÃO: Removido 'return df' prematuro que impedia features avançadas
    # ... continue com o resto da sua função ...
    # return df    # ← BUG REMOVIDO!
    # ... (mantenha o resto da função original) ...
    # ------------------------------------------------------------
    # 2. MOMENTUM / VELOCIDADE
    # ------------------------------------------------------------
    df["momentum_1"] = df["ret1"]
    df["momentum_2"] = df["ret2"]
    df["momentum_acceleration"] = df["momentum_1"] - df["momentum_2"]

    # ------------------------------------------------------------
    # 3. WICK / STRUCTURE (duplicadas REMOVIDAS)
    # ------------------------------------------------------------
    df["body_to_range"] = df["body"] / (df["range"] + 1e-8)
    df["wick_ratio_up"] = df["upper_wick"] / (df["range"] + 1e-8)
    df["wick_ratio_down"] = df["lower_wick"] / (df["range"] + 1e-8)

    # Duplicatas removidas:
    # df["body_ratio"] = ...
    # df["close_position_in_range"] = ...
    # df["wick_upper_ratio"] = ...
    # df["wick_lower_ratio"] = ...

    # ------------------------------------------------------------
    # 4. MICROESTRUTURA (duplicadas removidas)
    # micro_range = range (duplicado)
    # micro_volatility = range_pct (duplicado)
    # ------------------------------------------------------------
    # df["micro_range"] = df["range"]                          # REMOVIDO
    # df["micro_volatility"] = df["range_pct"]                 # REMOVIDO

    # ------------------------------------------------------------
    # 5. VOLUME / SURGE (duplicado de exp_rate_20 REMOVIDO)
    # ------------------------------------------------------------
    df["volume_diff"] = df["volume"].diff()
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / \
                           (df["volume"].rolling(20).std() + 1e-8)

    # df["volume_surge"] = df["volume"] / df["volume"].rolling(20).mean()   # REMOVIDO — igual a exp_rate_20

    # ------------------------------------------------------------
    # 6. AGRESSÃO (duplicatas REMOVIDAS)
    # ------------------------------------------------------------
    df["aggression_imbalance"] = df["aggression_delta"] / (df["volume"] + 1e-8)
    df["aggression_pressure"] = df["aggression_delta"] * df["ret1"]

    # Duplicatas REMOVIDAS:
    # df["aggression_delta"] (já existe no Feature Engine)
    # df["aggression_ratio"] (já existe)
    
    # ------------------------------------------------------------
    # 7. ACELERAÇÕES / DERIVADAS
    # ------------------------------------------------------------
    df["delta_acc"] = df["aggression_delta"].diff()
    df["volatility_acceleration"] = df["range_pct"].diff()

    # =====================================================================
    # 7. ATR E VOLATILIDADE INSTITUCIONAL
    # =====================================================================

    # True Range
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )

    # ATR clássico de 14 períodos
    df["atr14"] = df["tr"].rolling(14).mean()

    # ATR normalizado
    df["atr_to_close"] = df["atr14"] / (df["close"] + 1e-8)

    # Range relativo ao ATR (rompimentos)
    df["range_to_atr"] = df["range"] / (df["atr14"] + 1e-8)


    # =====================================================================
    # 8. REGIMES (VOLATILIDADE, TENDÊNCIA, LIQUIDEZ)
    # =====================================================================

    # Volatility regime (estado de volatilidade de 50 períodos)
    df["vol_regime"] = df["atr14"].rolling(50).mean()

    # Compressão de volatilidade (squeeze)
    df["atr_compression"] = df["atr14"] / (df["vol_regime"] + 1e-8)

    # Trend regime — baseado em slope de 100 períodos
    df["trend_regime"] = slope_regression(df["close"], 100)

    # Liquidez regime — média de volume
    df["liquidity_regime"] = df["volume"].rolling(50).mean()


    # =====================================================================
    # 9. RETORNOS AVANÇADOS / MOMENTUM EXPANDIDO
    # =====================================================================

    df["ret3"] = df["close"].pct_change(3)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)

    # Retornos normalizados pela volatilidade
    df["ret20_norm"] = df["ret20"] / (df["atr14"] + 1e-8)

    # Momentum agregado
    df["momentum_long"] = df["ret3"] + df["ret10"] + df["ret20"]


    # =====================================================================
    # 10. Z-SCORE DIRECIONAL (REVERSÃO / EXAUSTÃO)
    # =====================================================================

    df["price_z"] = (
        (df["close"] - df["close"].rolling(20).mean()) /
        (df["close"].rolling(20).std() + 1e-8)
    )


    # =====================================================================
    # 11. SQUEEZE AVANÇADO (VOL, RANGE, RET)
    # =====================================================================

    df["vol_squeeze"] = (
        df["close"].rolling(20).std() /
        (df["close"].rolling(100).std() + 1e-8)
    )

    df["range_squeeze"] = (
        df["range_pct"].rolling(14).std() /
        (df["range_pct"].rolling(50).std() + 1e-8)
    )


    # =====================================================================
    # 12. FEATURES DE Agressão (Fluxo Institucional)
    # =====================================================================

    # Buy / Sell ratio (muito forte)
    df["buy_ratio"] = df["aggression_buy"] / (df["volume"] + 1e-8)
    df["sell_ratio"] = df["aggression_sell"] / (df["volume"] + 1e-8)

    # Agressão acumulada — detecta pressão contínua
    df["aggr_cumsum_20"] = df["aggression_delta"].rolling(20).sum()

    # 🚀 FEATURE DE ELITE: DOMINÂNCIA DE FLUXO (PESO DINÂMICO)
    # Esta feature amplifica o sinal de agressão do TF Líder para o modelo dar prioridade
    tf_lider = globals().get("TF_LIDER_FLUXO", "15m")
    
    # Se estivermos no TF Líder, criamos a feature de dominância
    if "aggression_delta" in df.columns:
        # Se for o TF Líder (ou se não houver líder definido e for o base)
        df["flow_dominance"] = df["aggression_delta"] * (1 + df["vpin"].fillna(0))
        df["flow_acceleration"] = df["aggression_delta"].diff()
        # print(f"✔ Feature de Dominância de Fluxo ativada.")


    # =====================================================================
    # 13. DISTÂNCIAS IMPORTANTES (NORMALIZAÇÕES INSTITUCIONAIS)
    # =====================================================================
    
    # Criar EMAs se não existirem
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    if "ema100" not in df.columns:
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    
    # Criar distâncias se não existirem
    if "dist_ema50" not in df.columns:
        df["dist_ema50"] = df["close"] - df["ema50"]
    if "dist_ema100" not in df.columns:
        df["dist_ema100"] = df["close"] - df["ema100"]
    
    # Normalizar distâncias
    df["dist_ema20_norm"] = df["dist_ema20"] / (df["atr14"] + 1e-8)
    df["dist_ema50_norm"] = df["dist_ema50"] / (df["atr14"] + 1e-8)
    df["dist_ema100_norm"] = df["dist_ema100"] / (df["atr14"] + 1e-8)


    # =====================================================================
    # 14. SLOPES LONGOS (TENDÊNCIA PROFUNDA)
    # =====================================================================

    df["slope100"] = slope_regression(df["close"], 100)
    df["slope200"] = slope_regression(df["close"], 200)


    return df
# =============================================================================
# APLICAR PESO TEMPORAL (COPIADO EXATO DO V27)
# =============================================================================
def aplicar_peso_temporal(df):
    """
    Função restaurada e oficial para o pipeline V37.
    Cria df['sample_weight'] ANTES do treino.
    COPIADO EXATO DO V27.
    """
    df = df.copy()

    # 1) Validar timestamp ts
    if "ts" not in df.columns:
        print("[ERRO] 'ts' não encontrado no dataset. Peso temporal não será aplicado.", flush=True)
        return df

    ts = pd.to_datetime(df["ts"], unit="ms")
    dt_max = ts.max()
    dt_min = ts.min()

    idade_dias = (dt_max - ts).dt.days.astype(float)
    total_dias = max(1, (dt_max - dt_min).days)

    # 2) Meia-vida automática
    meia_vida = max(30, total_dias * 0.20)
    peso_tempo = np.power(0.5, idade_dias / meia_vida)

    # 3) Peso por regime (compatível)
    if "trend_regime" in df.columns:
        reg = df["trend_regime"].fillna(0)
        peso_regime = np.where(
            reg == -1, 0.9,
            np.where(reg == 1, 1.2, 1.0)
        )
    else:
        peso_regime = 1.0

    # 4) Peso final e normalização
    peso_final = peso_tempo * peso_regime
    peso_final = peso_final / peso_final.mean()

    df["sample_weight"] = peso_final

    print("    ✅ Peso temporal aplicado (V40 automático + regime).", flush=True)
    return df

# =============================================================================
# DETECTAR REGIMES COMPLETO (COPIADO EXATO DO V27)
# =============================================================================
def detectar_regimes(df, n_clusters_ret=4, n_clusters_vol=4):
    """
    Detecta regimes de mercado usando KMeans duplo (retorno + volatilidade).
    Cria: ret_1, ret_7, ret_14, vol_7, vol_14, slope_6, reg_ret, reg_vol, regime_final
    COPIADO EXATO DO V27.
    """
    df = df.copy()

    # 1. Features de retorno
    df["ret_1"] = df["close"].pct_change()
    df["ret_7"] = df["close"].pct_change(7)
    df["ret_14"] = df["close"].pct_change(14)

    # 2. Features de volatilidade
    df["vol_7"] = df["ret_1"].rolling(7).std()
    df["vol_14"] = df["ret_1"].rolling(14).std()
    if "range_pct" not in df.columns:
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # 3. Slope (regressão linear curta)
    janela_slope = 6

    def calc_slope(arr):
        if len(arr) < janela_slope:
            return np.nan
        x = np.arange(len(arr))
        y = arr
        coef = np.polyfit(x, y, 1)[0]
        return coef

    df["slope_6"] = df["close"].rolling(janela_slope).apply(calc_slope, raw=False)

    # Remover NaNs temporários
    regime_df = df[["ret_1", "ret_7", "ret_14", "vol_7", "vol_14", "range_pct", "slope_6"]].dropna()

    # 4. KMeans de retorno
    ret_features = regime_df[["ret_1", "ret_7", "ret_14"]].fillna(0)
    km_ret = KMeans(n_clusters=n_clusters_ret, n_init=10, random_state=42)
    ret_labels = km_ret.fit_predict(ret_features)

    # 5. KMeans de volatilidade
    vol_features = regime_df[["vol_7", "vol_14", "range_pct", "slope_6"]].fillna(0)
    km_vol = KMeans(n_clusters=n_clusters_vol, n_init=10, random_state=42)
    vol_labels = km_vol.fit_predict(vol_features)

    # 6. Reconstrução no df original
    regime_df = regime_df.assign(reg_ret=ret_labels, reg_vol=vol_labels)

    df["reg_ret"] = np.nan
    df["reg_vol"] = np.nan

    df.loc[regime_df.index, "reg_ret"] = regime_df["reg_ret"]
    df.loc[regime_df.index, "reg_vol"] = regime_df["reg_vol"]

    # 7. Regime final (cruzamento)
    df["regime_final"] = df["reg_ret"].astype(str) + "_" + df["reg_vol"].astype(str)

    print("    ✅ Regimes detectados (reg_ret, reg_vol, regime_final).", flush=True)
    return df

# =============================================================================
# CALCULAR PESO POR REGIME (COPIADO EXATO DO V27)
# =============================================================================
def calcular_peso_regime(df, target_col="target_K1"):
    """
    Calcula pesos por regime baseados na performance histórica.
    COPIADO EXATO DO V27.
    """
    df = df.copy()

    if "regime_final" not in df.columns:
        print("    ⚠ regime_final não encontrado. Usando peso_regime = 1.", flush=True)
        df["peso_regime"] = 1.0
        return df

    # Agrupar por regime e medir performance média
    if target_col in df.columns:
        grp = df.groupby("regime_final")[target_col].apply(
            lambda x: (x == 1).mean() if len(x) > 20 else np.nan
        ).dropna()

        if grp.empty:
            print("    ⚠ Sem regimes suficientes. Usando peso_regime = 1.", flush=True)
            df["peso_regime"] = 1.0
            return df

        # Normalização para intervalo 0.5 → 2.0
        min_v, max_v = grp.min(), grp.max()
        if max_v - min_v == 0:
            grp_norm = grp / grp
        else:
            grp_norm = 0.5 + 1.5 * (grp - min_v) / (max_v - min_v)

        df["peso_regime"] = df["regime_final"].map(grp_norm).fillna(1.0)
    else:
        df["peso_regime"] = 1.0

    print("    ✅ Peso por regime calculado.", flush=True)
    return df

# =============================================================================
# DETECTAR REGIMES MERCADO V25 (COPIADO EXATO DO V27)
# =============================================================================
def detectar_regimes_mercado_v25(df, n_regimes=4):
    """
    Detecta regimes de mercado usando KMeans e salva scaler/kmeans.
    COPIADO EXATO DO V27.
    """
    print(">>> Detectando regimes de mercado...", flush=True)
    
    # 1. Features para regime
    regime_features = [c for c in ['vol_realized', 'rsi_14', 'atr14', 'slope20'] if c in df.columns]
    
    if not regime_features:
        df['temp_ret'] = df['close'].pct_change(20)
        regime_features = ['temp_ret']
    
    print(f"    Features regime: {regime_features}", flush=True)
    
    # 2. Prepara matriz
    X_regime = df[regime_features].fillna(0).values
    
    # 3. Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    
    # 4. KMeans
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    # 5. SALVA SCALER E KMEANS
    try:
        scaler_path = os.path.join(MODELOS_DIR, 'scaler_regimes.pkl')
        joblib.dump(scaler, scaler_path)
        
        kmeans_path = os.path.join(MODELOS_DIR, 'kmeans_regimes.pkl')
        joblib.dump(kmeans, kmeans_path)
        
        # Cache global
        modelos_cache['scaler_regimes'] = scaler
        modelos_cache['kmeans_regimes'] = kmeans
        
        print(f"    ✅ scaler_regimes.pkl salvo", flush=True)
        print(f"    ✅ kmeans_regimes.pkl salvo", flush=True)
        print(f"    Distribuição regimes: {df['market_regime'].value_counts().to_dict()}", flush=True)
    except Exception as e:
        print(f"    ❌ Erro ao salvar scaler/kmeans: {e}", flush=True)
        raise
    
    return df, scaler, kmeans

# =============================================================================
# AUTO THRESHOLDS V25 (COPIADO EXATO DO V27)
# =============================================================================
def auto_thresholds_v25(df):
    """
    Calcula thresholds adaptativos baseados no ATR.
    COPIADO EXATO DO V27.
    """
    if "atr14" not in df.columns or "close" not in df.columns:
        raise RuntimeError("ATR14 ou close não encontrado.")

    # ATR em percentual (escala compatível com amp_fut)
    atr_pct = df["atr14"] / df["close"]

    atr_ref = (
        atr_pct
        .rolling(200, min_periods=200)
        .mean()
    )

    atr_global = atr_pct.mean()
    atr_ref = atr_ref.fillna(atr_global)

    # Thresholds agora em % (compatíveis com amp_fut)
    thrA = float((atr_ref * 1.2).iloc[-1])
    thrB = float((atr_ref * 1.6).iloc[-1])
    thrC = float((atr_ref * 2.0).iloc[-1])

    print(f"    Threshold A = {thrA:.4%}", flush=True)
    print(f"    Threshold B = {thrB:.4%}", flush=True)
    print(f"    Threshold C = {thrC:.4%}", flush=True)

    return thrA, thrB, thrC

# =============================================================================
# CRIAR TARGETS A, B, C (CORRIGIDO - IDÊNTICO AO V27)
# =============================================================================
def criar_targets_v25(df: pd.DataFrame, thrA: float, thrB: float, thrC: float) -> pd.DataFrame:
    """
    LÓGICA ORIGINAL V27 RESTAURADA (QUE FUNCIONAVA BEM)
    
    target_A: DIREÇÃO com 3 classes (-1, 0, 1)
              - Classe  1 = vai subir (COMPRA)
              - Classe -1 = vai cair (VENDA)
              - Classe  0 = neutro/lateral
              
    target_B: MAGNITUDE MÉDIA com 3 classes (-1, 0, 1)
              - Usa threshold B (maior que A)
              - Mesmo padrão de classes
              
    target_C: AMPLITUDE GRANDE
              - Classe 1 = movimento grande (volatilidade)
              - Classe 0 = mercado lateral
              
    target_A_bin: ATIVIDADE (GATE DE MERCADO) - BINÁRIO
              - Classe 1 = há movimento significativo (para cima OU para baixo)
              - Classe 0 = mercado lateral (sem movimento)
    """
    df = df.copy()
    sensibilidade_short = 0.95
    thrA_s = thrA * sensibilidade_short
    thrB_s = thrB * sensibilidade_short
    thrC_s = thrC * sensibilidade_short
    
    # =================================================================
    # TARGET_A: DIREÇÃO (3 classes: -1, 0, 1)
    # =================================================================
    df["target_A"] = 0
    df.loc[df["ret_fut"] >= thrA, "target_A"] = 1      # Compra
    df.loc[df["ret_fut"] <= -thrA_s, "target_A"] = -1  # Venda
    
    # =================================================================
    # TARGET_B: MAGNITUDE MÉDIA (3 classes: -1, 0, 1)
    # =================================================================
    df["target_B"] = 0
    df.loc[df["ret_fut"] >= thrB, "target_B"] = 1      # Compra forte
    df.loc[df["ret_fut"] <= -thrB_s, "target_B"] = -1  # Venda forte
    
    # =================================================================
    # TARGET_C: AMPLITUDE (2 classes: 0, 1)
    # =================================================================
    df["target_C"] = 0
    if "amp_fut" in df.columns:
        df.loc[df["amp_fut"] >= thrC, "target_C"] = 1
    
    # =================================================================
    # TARGET_A_BIN: ATIVIDADE/GATE - BINÁRIO (0 ou 1)
    # =================================================================
    # 1 = há movimento significativo (para cima OU para baixo)
    # 0 = mercado lateral
    df["target_A_bin"] = ((df["ret_fut"] >= thrA) | (df["ret_fut"] <= -thrA_s)).astype(int)
    
    # Log da distribuição
    print(f"    ✅ Targets criados (LÓGICA ORIGINAL V27):", flush=True)
    print(f"       target_A (-1/0/1): {dict(df['target_A'].value_counts().sort_index())}", flush=True)
    print(f"       target_B (-1/0/1): {dict(df['target_B'].value_counts().sort_index())}", flush=True)
    print(f"       target_C (0/1):    {dict(df['target_C'].value_counts().sort_index())}", flush=True)
    print(f"       target_A_bin:      {dict(df['target_A_bin'].value_counts().sort_index())}", flush=True)
    
    return df

# =============================================================================
# PREPARAR FUTUROS (COPIADO EXATO DO V27)
# =============================================================================
def preparar_futuros(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Calcula retorno futuro (ret_fut) e amplitude futura (amp_fut)
    sem leakage e remove linhas sem futuro.
    COPIADO EXATO DO V27.
    """
    df = df.copy()

    # preço futuro N barras à frente (sem leakage)
    df["close_future"] = df["close"].shift(-N)
    df["ret_fut"] = (df["close_future"] - df["close"]) / df["close"]

    # máxima e mínima futuras do candle N à frente
    df["high_fut"] = df["high"].shift(-N)
    df["low_fut"] = df["low"].shift(-N)
    df["amp_fut"] = (df["high_fut"] - df["low_fut"]) / (df["close"] + 1e-9)

    # remover linhas sem futuro
    df = df.iloc[:-N].reset_index(drop=True)
    
    return df
def montar_matriz(df: pd.DataFrame, alvo: str):
    """
    Remove colunas que não são features + retorna X, y e lista de colunas.

    PATCH CRÍTICO:
    - Remove TODAS as colunas target_* das features (A_bin, A/B/C, K1..K5, etc).
    - Evita leakage estrutural no treino e garante identidade com backtest.
    - GARANTE X 100% numérico (necessário para Quantile Regression).
    """

    non_feat = {
        # candles
        "open", "high", "low", "close", "volume",

        "ts","open","high","low","close","volume","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","close_time","ignore",
        "mark_price","index_price","fundingRate",
        "session","tp",
        "close_future","ret_fut","amp_fut",
        "high_fut","low_fut",
        "impulse_count",
        "ret_max", "ret_min", "ret_max_temp", "ret_min_temp", # 🔴 PATCH ANTI-LEAKAGE
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg" # Colunas auxiliares de micro (não são features diretas)
        # 🔴 FRACTAIS - VAZAMENTO (shift negativo)
        "fractal_high",
        "fractal_low",
        "pivot_high",
        "pivot_low",
        "last_pivot_high",
        "last_pivot_low",
        "swing_dir",
        "wave_amplitude",
        "wave_amplitude_abs",
        "correction_pct",
        "wave_strength",
        }

    # 🔴 PATCH: remover TODOS os targets do set de colunas não-feature
    target_cols = {
        c for c in df.columns
        if isinstance(c, str) and c.startswith("target_")
    }
    non_feat = non_feat.union(target_cols)

    if alvo not in df.columns:
        raise RuntimeError(f"[montar_matriz] alvo ausente no df: {alvo}")

    # -----------------------------
    # y (alvo)
    # -----------------------------
    y = df[alvo].values

    # -----------------------------
    # Seleção inicial de features
    # -----------------------------
    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].copy()

    # -----------------------------
    # 🔥 SANIDADE FINAL — APENAS NUMÉRICAS
    # -----------------------------
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise RuntimeError(
            f"[montar_matriz] Nenhuma feature numérica disponível para alvo={alvo}"
        )

    # Converte QUALQUER resíduo para numérico (object, datetime mascarado etc.)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Elimina NaN gerado por coerção
    if X.isna().any().any():
        X = X.fillna(0.0)

    feat_cols = list(X.columns)

    # Retorno FINAL — 100% float
    return X.values.astype(float), y, feat_cols


# ------------------------------------------------------------------------
# 5.2 — Split temporal 70/10/20
# ------------------------------------------------------------------------

def temporal_split(X, y, gap=15):
    n = len(X)
    tr = int(n * 0.70)
    va = int(n * 0.80)

    tr_end = tr
    va_start = tr + gap
    va_end = va
    te_start = va + gap
    return (
        X[:tr_end], y[:tr_end],
        X[va_start:va_end], y[va_start:va_end],
        X[te_start:], y[te_start:],
    )
def treinar_um_target(target_col, df, outdir):
    """Treina um target com LGBM + XGB e escolhe o melhor. COPIADO EXATO DO V27."""
    
    print(f"\n{'='*70}")
    print(f"TREINANDO TARGET {target_col}")
    print(f"{'='*70}")

    df_local = df.copy()
    classes_orig = np.unique(df_local[target_col].values)

    # Ajuste de classes
    if set(classes_orig) == {-1, 0, 1}:
        df_local[target_col] = df_local[target_col].map({-1: 0, 0: 1, 1: 2})
    elif len(classes_orig) > 0 and classes_orig.min() < 0:
        df_local[target_col] = df_local[target_col] - classes_orig.min()

    classes = np.unique(df_local[target_col].values)
    n_classes = len(classes)

    # Matriz + split
    X, y, feat_cols = montar_matriz(df_local, target_col)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"    Features: {len(feat_cols)}")

    # Sample weight
    if "sample_weight" in df_local.columns:
        sw = df_local["sample_weight"].values
        sw_train = sw[:len(X_train)]
    else:
        sw_train = None

    resultados = []

    # LGBM
    try:
        model_lgb = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            num_class=None if n_classes == 2 else n_classes,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1
        )

        model_lgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_lgb.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("LGBM", f1, model_lgb))
        print(f"    >>> LGBM F1={f1:.4f}")

    except Exception as e:
        print(f"    [LGBM] erro: {e}")

    # XGBoost
    try:
        counts = pd.Series(y_train).value_counts()
        spw = counts.min() / counts.max() if len(counts) == 2 else 1.0

        model_xgb = XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=spw if n_classes == 2 else None
        )

        model_xgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_xgb.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("XGB", f1, model_xgb))
        print(f"    >>> XGB F1={f1:.4f}")

    except Exception as e:
        print(f"    [XGB] erro: {e}")

    # CatBoost
    try:
        model_cat = CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=6,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42
        )

        model_cat.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_cat.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("CAT", f1, model_cat))
        print(f"    >>> CAT F1={f1:.4f}")

    except Exception as e:
        print(f"    [CAT] erro: {e}")

    # ------------------------------------------------------------
    # STACKING ENSEMBLE (LGBM + XGB + CAT + Meta-Learner)
    # ------------------------------------------------------------
    try:
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        print("    >>> Treinando STACK (pode demorar)...", flush=True)
        
        # Base estimators com menos iterações (já que são 3)
        base_estimators = [
            ('lgbm', LGBMClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=-1,
                n_jobs=-1, verbose=-1, class_weight="balanced"
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=6,
                tree_method="hist", eval_metric="logloss", verbosity=0,
                scale_pos_weight=spw if n_classes == 2 else None
            )),
            ('cat', CatBoostClassifier(
                iterations=200, learning_rate=0.03, depth=6,
                auto_class_weights="Balanced", verbose=0
            ))
        ]
        
        # Meta-learner simples
        meta_learner = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        
        model_stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,  # CV=3 para ser mais rápido
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Stacking não suporta sample_weight diretamente no fit
        model_stack.fit(X_train, y_train)
        preds = model_stack.predict(X_test)
        
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("STACK", f1, model_stack))
        
        print(f"    >>> STACK F1={f1:.4f}")
        
    except Exception as e:
        print(f"    [STACK] erro: {e}")

    if not resultados:
        raise RuntimeError(f"Nenhum modelo treinou para {target_col}")

    # Ordenar por F1 e mostrar ranking
    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)
    print(f"\n    📊 RANKING {target_col}:")
    for i, (nome, f1, _) in enumerate(resultados_ordenados, 1):
        marca = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"       {marca} {i}º {nome}: F1={f1:.4f}")

    melhor_nome, melhor_f1, melhor_modelo = resultados_ordenados[0]
    print(f"\n    ✅ VENCEDOR: {melhor_nome} (F1={melhor_f1:.4f})")

    # Salvar
    os.makedirs(outdir, exist_ok=True)
    nome_arquivo = f"{target_col}_{melhor_nome}.pkl"
    model_path = os.path.join(outdir, nome_arquivo)
    joblib.dump(melhor_modelo, model_path)

    # Registrar
    caminhos_modelos[target_col] = model_path
    features_por_target[target_col] = list(feat_cols)
    modelos_cache[target_col] = melhor_modelo

    print(f"    ✅ Modelo salvo: {model_path}")

    return nome_arquivo, melhor_f1, model_path

# =============================================================================
# DOWNLOAD E GERAÇÃO DE CSVs
# =============================================================================
def baixar_aggtrades():
    """Download aggTrades do Binance Data Vision."""
    print("\n>>> BAIXANDO aggTrades...", flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    print(f"    {total_dates} dias", flush=True)
    
    if os.path.exists(CSV_AGG_PATH):
        os.remove(CSV_AGG_PATH)
    
    session = requests.Session()
    success_count = 0
    first_write = True
    
    for i, date in enumerate(dates, 1):
        print(f"    [{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
        
        t_start = time.time()
        df = download_daily_file(SYMBOL, date, session, retry_count=5)
        elapsed = time.time() - t_start
        
        if df is not None:
            df_processed = process_binance_data(df)
            
            if df_processed is not None and not df_processed.empty:
                df_processed.to_csv(CSV_AGG_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                print(f"✓ {len(df_processed):,} trades ({elapsed:.1f}s)", flush=True)
                del df, df_processed
            else:
                print(f"⚠️ Vazio", flush=True)
        else:
            print(f"⚠️ Falhou", flush=True)
        
        time.sleep(random.uniform(0.3, 1.0))
    
    session.close()
    print(f"\n    {success_count}/{total_dates} dias OK", flush=True)
    
    if success_count == 0:
        raise Exception("NENHUM DADO baixado!")
    
    return success_count

def gerar_todos_timeframes():
    """Gera 15m, 30m, 1h, 4h, 8h, 1d."""
    print("\n>>> GERANDO TIMEFRAMES...", flush=True)
    
    timeframes = {
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "8h": 480,
        "1d": 1440,
    }
    
    csv_paths = {}
    
    for tf_name, tf_min in timeframes.items():
        csv_tf_path = os.path.join(OUT_DIR, f"{SYMBOL}_{tf_name}.csv")
        gerar_timeframe_tratado(CSV_AGG_PATH, csv_tf_path, timeframe_min=tf_min)
        csv_paths[tf_name] = csv_tf_path
        
        # fsync
        try:
            os.sync()
        except:
            pass
    
    return csv_paths

# =============================================================================
# TREINO COMPLETO K1-K6 (COPIADO DO V27)
# =============================================================================
def treinar_todos_targets_k(df_all):
    """
    Treina K1 até K6. COPIADO DO V27.
    
    IMPORTANTE PARA V27 CONFLUÊNCIA:
    ================================
    - K1, K2: Modelos ternários (-1/0/1) → K3 confluência derivada
    - K4, K5: Modelos ternários (-1/0/1) → K6 confluência derivada
    - K6: Modelo binário (0/1) com threshold elite 0.5%
    
    Confluência funciona assim:
    - target_K3: NÃO é treinado, é DERIVADO no backtest V27
      if K1==1 AND K2==1: trade LONG
      if K1==-1 AND K2==-1: trade SHORT
      
    - target_K6: Mesmo esquema usando K4 + K5
      if K4==1 AND K5==1: trade LONG
      if K4==-1 AND K5==-1: trade SHORT
    
    Targets A, B, C, A_bin também DEVEM ser treinados!
    (Veja PASSO 3.9 na função main() - V27 Confluência requer todos!)
    """
    
    print("\n" + "="*70)
    print("TREINO DOS TARGET_K (K1 até K6)")
    print("="*70)
    
    ks_treinados = []
    
    for k in range(1, HORIZONTE_BACKTEST + 1):
        alvo_k = f"target_K{k}"
        print(f"\n>>> Treinando {alvo_k} (horizonte={k} candles)")
        
        df_k = preparar_futuros(df_all.copy(), k)
        
        if "ret_fut" not in df_k.columns:
            raise RuntimeError(f"[TARGET_K{k}] ret_fut não encontrado")
        
        # K6 usa threshold especial (COPIADO DO V27)
        if k == 6:
            # K6 ELITE: Binário com threshold 0.5% (conforme V27 original)
            print(f"    🔥 K6 ELITE: Binário, threshold 0.5%")
            threshold_k6 = 0.005
            df_k[alvo_k] = (df_k["ret_fut"] > threshold_k6).astype(int)
        else:
            # K1-K5: TERNÁRIOS (-1, 0, 1) para confluência com shorts
            print(f"    K{k}: Ternário (-1/0/1) para confluência")
            df_k[alvo_k] = 0
            
            # Usa percentil 75 para threshold adaptativo (top/bottom 25%)
            p75 = df_k["ret_fut"].quantile(0.75)
            
            df_k.loc[df_k["ret_fut"] >= p75, alvo_k] = 1   # COMPRA (top 25%)
            df_k.loc[df_k["ret_fut"] <= -p75, alvo_k] = -1  # VENDA (bottom 25%)
        
        if df_k[alvo_k].nunique() < 2:
            print(f"    [AVISO] {alvo_k} degenerado — pulando")
            continue
        
        nome_modelo, f1, caminho_modelo = treinar_um_target(
            target_col=alvo_k,
            df=df_k,
            outdir=MODELOS_DIR
        )
        
        ks_treinados.append(k)
        print(f"    [OK] {alvo_k} treinado | F1={f1:.4f}")
    
    if not ks_treinados:
        raise RuntimeError("TARGET_K NÃO FOI TREINADO — ERRO FATAL")
    
    print(f"\n>>> TARGET_K treinados: {ks_treinados}")
    return ks_treinados

# =============================================================================
# CRIAR PACOTE FINAL (COPIADO DO V27)
# =============================================================================
def criar_pacote_final():
    """Cria SISTEMA_K6_COMPLETO.pkl com modelo + scaler + kmeans."""
    
    print("\n>>> Criando pacote final...", flush=True)
    
    scaler_path = os.path.join(MODELOS_DIR, 'scaler_regimes.pkl')
    kmeans_path = os.path.join(MODELOS_DIR, 'kmeans_regimes.pkl')
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    kmeans = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None
    modelo_k6 = modelos_cache.get('target_K6')
    
    if scaler and kmeans and modelo_k6:
        pacote = {
            'modelo': modelo_k6,
            'scaler': scaler,
            'kmeans': kmeans,
            'info': 'V27_SCALER_KMEANS_COMPLETO'
        }
        pacote_path = os.path.join(MODELOS_DIR, 'SISTEMA_K6_COMPLETO.pkl')
        joblib.dump(pacote, pacote_path)
        print(f"    ✅ SISTEMA_K6_COMPLETO.pkl salvo", flush=True)
        return pacote_path
    else:
        print(f"    ❌ Faltam componentes para o pacote", flush=True)
        return None

# =============================================================================
# SERVIDOR HTTP PARA DOWNLOAD (COPIADO DO V51)
# =============================================================================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download-csv' or self.path == '/download-csv/':
            if os.path.exists(ZIP_CSV_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(ZIP_CSV_PATH)}"')
                self.end_headers()
                with open(ZIP_CSV_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP CSVs ainda nao criado')
        elif self.path == '/download-pkl' or self.path == '/download-pkl/':
            if os.path.exists(ZIP_PKL_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(ZIP_PKL_PATH)}"')
                self.end_headers()
                with open(ZIP_PKL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP PKLs ainda nao criado')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            csv_status = '✅ PRONTO!' if os.path.exists(ZIP_CSV_PATH) else '⏳ Processando...'
            pkl_status = '✅ PRONTO!' if os.path.exists(ZIP_PKL_PATH) else '⏳ Processando...'
            
            csv_link = '<a href="/download-csv" style="font-size:20px;padding:15px 30px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;margin:10px;">📦 BAIXAR CSVs</a>' if os.path.exists(ZIP_CSV_PATH) else '<p>CSVs: Aguarde...</p>'
            pkl_link = '<a href="/download-pkl" style="font-size:20px;padding:15px 30px;background:#2196F3;color:white;text-decoration:none;border-radius:5px;margin:10px;">🤖 BAIXAR PKLs</a>' if os.path.exists(ZIP_PKL_PATH) else '<p>PKLs: Aguarde...</p>'
            
            html = f'''<html>
<body style="font-family:Arial;padding:50px;text-align:center;">
<h1>🚀 PENDLEUSDT DataManager V55</h1>
<hr>
<h2>CSVs (15m, 30m, 1h, 4h, 8h, 1d): {csv_status}</h2>
{csv_link}
<br><br>
<h2>PKLs (K1-K6, Scaler, KMeans): {pkl_status}</h2>
{pkl_link}
<hr>
<p style="color:gray;">Render Persistent Disk: {BASE_DISK}</p>
</body>
</html>'''
            self.wfile.write(html.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    print(f">>> Servidor HTTP na porta {port}", flush=True)
    server.serve_forever()

# =============================================================================
# CATBOX UPLOAD
# =============================================================================
def upload_catbox(filepath):
    url = "https://catbox.moe/user/api.php"
    with open(filepath, "rb") as f:
        r = requests.post(
            url,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
            timeout=300
        )
    r.raise_for_status()
    return r.text.strip()
def main():
    # Inicia servidor HTTP em background (COPIADO DO V51)
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    print("\n" + "="*70)
    print("🚀 DataManager V55 - PENDLE")
    print("="*70)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print(f"Diretório: {OUT_DIR}")
    print("="*70)
    
    # =========================================================================
    # PASSO 1: VERIFICAR CSVs EXISTENTES
    # =========================================================================
    print("\n>>> PASSO 1: Verificando CSVs...", flush=True)
    
    required_tfs = ['15m', '30m', '1h', '4h', '8h', '1d']
    csv_paths = {}
    all_exist = True
    
    for tf in required_tfs:
        csv_path = os.path.join(OUT_DIR, f"{SYMBOL}_{tf}.csv")
        if os.path.exists(csv_path):
            size_mb = os.path.getsize(csv_path) / (1024*1024)
            print(f"    ✅ {tf}: {size_mb:.2f} MB", flush=True)
            csv_paths[tf] = csv_path
        else:
            print(f"    ❌ {tf}: NÃO ENCONTRADO", flush=True)
            all_exist = False
    
    # =========================================================================
    # PASSO 2: DOWNLOAD E GERAÇÃO SE NECESSÁRIO
    # =========================================================================
    if not all_exist:
        print("\n>>> PASSO 2: Download e geração...", flush=True)
        baixar_aggtrades()
        csv_paths = gerar_todos_timeframes()
    
    # =========================================================================
    # PASSO 3: CARREGAR 15m E APLICAR FEATURES
    # =========================================================================
    print("\n>>> PASSO 3: Features + Regimes...", flush=True)
    
    df_15m = pd.read_csv(csv_paths['15m'])
    print(f"    15m carregado: {len(df_15m)} candles", flush=True)
    
    df_15m = feature_engine(df_15m)
    print(f"    Após feature_engine: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_features_avancadas(df_15m)
    print(f"    Após adicionar_features_avancadas: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_fractais_elliott(df_15m)
    print(f"    Após adicionar_fractais_elliott: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_vwap(df_15m)
    print(f"    Após adicionar_vwap: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_micro_squeeze(df_15m)
    print(f"    Após adicionar_micro_squeeze: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_inside_nr(df_15m)
    print(f"    Após adicionar_inside_nr: {len(df_15m.columns)} colunas", flush=True)
    df_15m = adicionar_zscore_intrabar(df_15m)
    print(f"    Após adicionar_zscore_intrabar: {len(df_15m.columns)} colunas", flush=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FEATURES INSTITUCIONAIS AVANÇADAS V2 (FLUXO DE BALEIAS >= $500)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n>>> Aplicando Features Institucionais V2...", flush=True)
    try:
        from features_institucionais_v2 import aplicar_features_institucionais_v2
        df_15m = aplicar_features_institucionais_v2(df_15m, verbose=True)
        print("OK Features Institucionais V2 aplicadas!", flush=True)
    except ImportError:
        print("WARN Modulo features_institucionais_v2 nao encontrado. Aplicando versao inline...", flush=True)
        if "buy_vol_agg" in df_15m.columns or "taker_buy_base" in df_15m.columns:
            if "buy_vol_agg" in df_15m.columns:
                buy_vol = df_15m["buy_vol_agg"].fillna(0)
                sell_vol = df_15m["sell_vol_agg"].fillna(0)
            else:
                buy_vol = df_15m["taker_buy_base"].fillna(0)
                sell_vol = (df_15m["volume"] - df_15m["taker_buy_base"]).fillna(0).clip(lower=0)
            total_vol = buy_vol + sell_vol
            delta = buy_vol - sell_vol
            df_15m["vpin_v2"] = (delta.abs().rolling(50, min_periods=10).sum() / (total_vol.rolling(50, min_periods=10).sum() + 1e-10)).clip(0, 1)
            for w in [5, 10, 20, 50]:
                df_15m[f"ofi_{w}"] = delta.rolling(w).sum() / (total_vol.rolling(w).sum() + 1e-10)
            df_15m["whale_ratio"] = total_vol / (df_15m["volume"] + 1e-10)
            df_15m["whale_delta"] = delta / (total_vol + 1e-10)
            print("OK Features inline aplicadas!", flush=True)
    except Exception as e:
        print(f"WARN Erro features institucionais: {e}", flush=True)
    # ═══════════════════════════════════════════════════════════════════════════════

    
    # Detectar regimes completos (V27)
    df_15m = detectar_regimes(df_15m)
    
    # Aplicar peso temporal (V27)
    df_15m = aplicar_peso_temporal(df_15m)
    
    print(f"    Features aplicadas: {len(df_15m.columns)} colunas", flush=True)
    
    df_15m, scaler, kmeans = detectar_regimes_mercado_v25(df_15m, n_regimes=4)
    
    # =========================================================================
    # PASSO 3.5: CRIAR TARGETS A, B, C (COPIADO DO V27)
    # =========================================================================
    print("\n>>> PASSO 3.5: Criar targets A, B, C...", flush=True)
    
    # Preparar futuros com horizonte padrão (N=3)
    H_FUT = 3
    df_15m = preparar_futuros(df_15m, H_FUT)
    print(f"    Horizonte futuro: {H_FUT} candles", flush=True)
    
    # Calcular thresholds adaptativos
    df_calib = df_15m.iloc[:int(len(df_15m) * 0.2)]
    thrA, thrB, thrC = auto_thresholds_v25(df_calib)
    
    # Criar targets A, B, C
    df_15m = criar_targets_v25(df_15m, thrA, thrB, thrC)
    
    print(f"    ✅ Targets A, B, C criados", flush=True)
    print(f"    target_A: {df_15m['target_A'].value_counts().to_dict()}", flush=True)
    print(f"    target_A_bin: {df_15m['target_A_bin'].value_counts().to_dict()}", flush=True)
    
    # Calcular peso por regime usando target_A
    df_15m = calcular_peso_regime(df_15m, target_col="target_A")
    
    # =========================================================================
    # PASSO 3.9: TREINO TARGETS A, B, C, A_BIN (V27 CONFLUÊNCIA REQUER)
    # =========================================================================
    print("\n>>> PASSO 3.9: Treino targets A, B, C, A_bin...", flush=True)
    
    targets_abc = ['target_A_bin', 'target_A', 'target_B', 'target_C']
    abc_treinados = []
    
    for target_col in targets_abc:
        if target_col not in df_15m.columns:
            print(f"    [AVISO] {target_col} não existe — pulando", flush=True)
            continue
        
        print(f"    >>> Treinando {target_col}", flush=True)
        
        try:
            nome_modelo, f1, caminho_modelo = treinar_um_target(
                target_col=target_col,
                df=df_15m,
                outdir=MODELOS_DIR
            )
            abc_treinados.append(target_col)
            print(f"        ✅ {target_col} treinado | F1={f1:.4f}", flush=True)
        except Exception as e:
            print(f"        ❌ {target_col}: {str(e)[:100]}", flush=True)
    
    if abc_treinados:
        print(f"    ✅ Targets A/B/C treinados: {abc_treinados}", flush=True)
    else:
        print("    ⚠️ AVISO: Nenhum target A/B/C foi treinado!")
        print("    V27 Confluência pode não funcionar completamente!")

        # =========================================================================
    # PASSO 4: TREINO K1-K6
    # =========================================================================
    print("\n>>> PASSO 4: Treino K1-K6...", flush=True)
    ks_treinados = treinar_todos_targets_k(df_15m)
    
    # =========================================================================
    # PASSO 5: CRIAR PACOTE FINAL
    # =========================================================================
    print("\n>>> PASSO 5: Pacote final...", flush=True)
    pacote_path = criar_pacote_final()
    
    # =========================================================================
    # PASSO 6: CRIAR ZIPs
    # =========================================================================
    print("\n>>> PASSO 6: Criando ZIPs...", flush=True)
    
    # ZIP CSVs
    with zipfile.ZipFile(ZIP_CSV_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for tf, path in csv_paths.items():
            z.write(path, arcname=os.path.basename(path))
    print(f"    ✅ {ZIP_CSV_PATH}", flush=True)
    
    # ZIP PKLs
    with zipfile.ZipFile(ZIP_PKL_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(MODELOS_DIR):
            if f.endswith('.pkl'):
                z.write(os.path.join(MODELOS_DIR, f), arcname=f)
    print(f"    ✅ {ZIP_PKL_PATH}", flush=True)
    
    # =========================================================================
    # PASSO 7: UPLOAD CATBOX
    # =========================================================================
    print("\n>>> PASSO 7: Upload CatBox...", flush=True)
    
    try:
        link_csvs = upload_catbox(ZIP_CSV_PATH)
        print(f"    ✅ CSVs: {link_csvs}", flush=True)
    except Exception as e:
        print(f"    ❌ Erro CSVs: {e}", flush=True)
        link_csvs = "ERRO"
    
    try:
        link_pkls = upload_catbox(ZIP_PKL_PATH)
        print(f"    ✅ PKLs: {link_pkls}", flush=True)
    except Exception as e:
        print(f"    ❌ Erro PKLs: {e}", flush=True)
        link_pkls = "ERRO"
    
    # =========================================================================
    # FINALIZADO
    # =========================================================================
    print("\n" + "="*70)
    print("✅ COMPLETO!")
    print("="*70)
    print(f"📦 CSVs: {link_csvs}")
    print(f"📦 PKLs: {link_pkls}")
    print("="*70)
    
    # Manter vivo
    print("\n>>> Servidor mantido ativo...", flush=True)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
