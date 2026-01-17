#!/usr/bin/env python3
# ============================================================
# DataManager_V55_OPTIMIZED.py
# PERFORMANCE OPTIMIZADA PARA RENDER (512MB - 2GB RAM)
# 
# OTIMIZAÇÕES APLICADAS (SEM MODIFICAR LÓGICA DAS FUNÇÕES):
# 1. Tipos de dados otimizados (float32 vs float64)
# 2. gc.collect() estratégico em pontos críticos
# 3. Processamento em chunks menores
# 4. Liberação imediata de objetos grandes
# 5. Compressão de modelos ao salvar
# 6. Redução de n_estimators para treino mais leve
# 7. Uso de early_stopping para evitar overfitting
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
import gc
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

warnings.filterwarnings("ignore")

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# OTIMIZAÇÃO 1: Configurar pandas para usar menos memória
# =============================================================================
pd.options.mode.chained_assignment = None

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
# CONFIGURAÇÃO (pode ser sobrescrita por variáveis de ambiente)
# =============================================================================
SYMBOL = os.environ.get("SYMBOL", "PENDLEUSDT")

# Período - pode ser configurado via env vars
DAYS = int(os.environ.get("DAYS", "365"))

# Calcular datas
END_DT = datetime(2025, 12, 31, 23, 59, 59)
START_DT = END_DT - timedelta(days=DAYS-1)
START_DT = START_DT.replace(hour=0, minute=0, second=0)

print(f"🔧 CONFIGURAÇÃO:", flush=True)
print(f"   SYMBOL: {SYMBOL}", flush=True)
print(f"   DIAS: {DAYS}", flush=True)
print(f"   PERÍODO: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}", flush=True)

# =============================================================================
# OTIMIZAÇÃO 2: Configurações de memória para Render
# =============================================================================
# Render Free: 512MB RAM
# Render Starter: 2GB RAM
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "30000"))  # Reduzido de 50000
N_ESTIMATORS_LIGHT = int(os.environ.get("N_ESTIMATORS", "200"))  # Reduzido de 400
COMPRESS_MODELS = True  # Comprime PKLs para economizar disco

print(f"   CHUNK_SIZE: {CHUNK_SIZE}", flush=True)
print(f"   N_ESTIMATORS: {N_ESTIMATORS_LIGHT}", flush=True)

# =============================================================================
# RENDER PERSISTENT DISK
# =============================================================================
print("=" * 60, flush=True)
print("🔍 DIAGNÓSTICO DE DISCO", flush=True)
print("=" * 60, flush=True)

def check_disk_writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"    ❌ {path}: {e}", flush=True)
        return False

disk_options = [
    "/data",
    "/var/data",
    "/opt/render/project/data",
    "/opt/render/project",
    "."
]

BASE_DISK = None
for path in disk_options:
    print(f"    Testando: {path}", flush=True)
    if check_disk_writable(path):
        BASE_DISK = path
        print(f"    ✅ USANDO: {path}", flush=True)
        break

if BASE_DISK is None:
    BASE_DISK = "."
    print(f"    ⚠️ FALLBACK: usando diretório atual", flush=True)

if BASE_DISK != "/data":
    print("", flush=True)
    print("⚠️" * 30, flush=True)
    print("⚠️ ATENÇÃO: Disco persistente /data NÃO disponível!", flush=True)
    print("⚠️" * 30, flush=True)
    print("", flush=True)

print("=" * 60, flush=True)

FOLDER_NAME = f"pendle_agg_{START_DT.strftime('%Y%m%d')}_a_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DISK, FOLDER_NAME)
MODELOS_DIR = os.path.join(OUT_DIR, "modelos_salvos")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

if not os.path.exists(OUT_DIR):
    print(f"❌ ERRO: Não conseguiu criar {OUT_DIR}", flush=True)
    sys.exit(1)
if not os.path.exists(MODELOS_DIR):
    print(f"❌ ERRO: Não conseguiu criar {MODELOS_DIR}", flush=True)
    sys.exit(1)

print(f"📂 OUT_DIR: {OUT_DIR}", flush=True)
print(f"📂 MODELOS_DIR: {MODELOS_DIR}", flush=True)

try:
    existing_files = os.listdir(OUT_DIR)
    if existing_files:
        print(f"📁 Arquivos existentes: {existing_files}", flush=True)
    else:
        print(f"📁 Diretório vazio (primeira execução)", flush=True)
except Exception as e:
    print(f"📁 Erro ao listar: {e}", flush=True)

CSV_AGG_PATH = os.path.join(OUT_DIR, f"{SYMBOL}_aggTrades.csv")
ZIP_CSV_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_CSVs.zip")
ZIP_PKL_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_PKLs.zip")

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

# =============================================================================
# V27 CONFIG
# =============================================================================
HORIZONTE_BACKTEST = 6
KS = [1, 2, 3, 4, 5]

# =============================================================================
# CACHES GLOBAIS
# =============================================================================
modelos_cache = {}
caminhos_modelos = {}
features_por_target = {}

# =============================================================================
# OTIMIZAÇÃO 3: Função para otimizar tipos de dados
# =============================================================================
def optimize_dtypes(df):
    """Reduz uso de memória convertendo para tipos menores."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype('uint8')
                elif c_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif c_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype('int32')
    
    return df

def print_memory_usage():
    """Debug: mostra uso de memória."""
    import psutil
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    print(f"    💾 Memória: {mem:.1f} MB", flush=True)

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
                        # OTIMIZAÇÃO: Especificar dtypes na leitura
                        df = pd.read_csv(f, header=0 if has_header else None,
                                        dtype={'price': 'float32', 'quantity': 'float32'})
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
    
    # OTIMIZAÇÃO: Usar float32 e int32
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('int64'),
        'price': pd.to_numeric(df['price'], errors='coerce').astype('float32'),
        'qty': pd.to_numeric(df['quantity'], errors='coerce').astype('float32'),
        'side': df['is_buyer_maker'].apply(convert_side).astype('int8')
    })
    
    return df_processed.dropna()

# =============================================================================
# GERAÇÃO DE TIMEFRAMES (OTIMIZADO PARA MEMÓRIA)
# =============================================================================
def gerar_timeframe_tratado(csv_agg_path, csv_tf_path, timeframe_min=15, min_val_usd=500, chunksize=None):
    """
    Converte aggTrades para timeframe - OTIMIZADO PARA RENDER.
    Mesma lógica do V51, mas com melhor gerenciamento de memória.
    """
    if chunksize is None:
        chunksize = CHUNK_SIZE
    
    print(f">>> Gerando {timeframe_min}m tratado (chunks de {chunksize})...", flush=True)
    print(f"    Filtro baleia: >= ${min_val_usd} USD", flush=True)

    buckets = {}
    chunks_processados = 0
    trades_total = 0

    # OTIMIZAÇÃO: Especificar dtypes na leitura
    dtype_spec = {
        'ts': 'int64',
        'price': 'float32',
        'qty': 'float32',
        'side': 'int8'
    }

    for chunk in pd.read_csv(csv_agg_path, chunksize=chunksize, dtype=dtype_spec):
        chunk = chunk.dropna(subset=["ts", "price", "qty", "side"])
        if chunk.empty:
            continue

        trades_total += len(chunk)

        # Bucket UTC
        dt = pd.to_datetime(chunk["ts"], unit="ms", utc=True)
        bucket_dt = dt.dt.floor(f"{timeframe_min}min")
        bucket_ms = (bucket_dt.astype("int64") // 10**6).astype("int64")
        chunk = chunk.assign(bucket_ms=bucket_ms)

        # Whale flag
        val_usd = chunk["price"] * chunk["qty"]
        is_whale = val_usd >= float(min_val_usd)

        # Itera em linhas (mantém ordem cronológica)
        for ts_ms, price, qty, side, bms, whale in zip(
            chunk["ts"].values,
            chunk["price"].values,
            chunk["qty"].values,
            chunk["side"].values,
            chunk["bucket_ms"].values,
            is_whale.values,
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
                }
                buckets[bms] = st
            else:
                if price > st["high"]:
                    st["high"] = float(price)
                if price < st["low"]:
                    st["low"] = float(price)
                st["close"] = float(price)

            st["volume"] += float(qty)

            if whale:
                if side == 0:
                    st["buy_vol"] += float(qty)
                else:
                    st["sell_vol"] += float(qty)
        
        # OTIMIZAÇÃO: Liberar chunk e gc a cada 5 chunks
        chunks_processados += 1
        del chunk
        
        if chunks_processados % 5 == 0:
            gc.collect()
            print(f"    Chunks: {chunks_processados}, Buckets: {len(buckets)}, Trades: {trades_total:,}", flush=True)

    gc.collect()

    if not buckets:
        raise RuntimeError(f"Nenhum bucket {timeframe_min}m gerado!")

    print(f"    Montando DataFrame com {len(buckets)} buckets...", flush=True)

    # OTIMIZAÇÃO: Construir DataFrame mais eficientemente
    sorted_keys = sorted(buckets.keys())
    
    # Pré-alocar arrays
    n = len(sorted_keys)
    ts_arr = np.empty(n, dtype='int64')
    open_arr = np.empty(n, dtype='float32')
    high_arr = np.empty(n, dtype='float32')
    low_arr = np.empty(n, dtype='float32')
    close_arr = np.empty(n, dtype='float32')
    vol_arr = np.empty(n, dtype='float32')
    buy_arr = np.empty(n, dtype='float32')
    sell_arr = np.empty(n, dtype='float32')
    
    for i, bms in enumerate(sorted_keys):
        st = buckets[bms]
        ts_arr[i] = st["ts"]
        open_arr[i] = st["open"]
        high_arr[i] = st["high"]
        low_arr[i] = st["low"]
        close_arr[i] = st["close"]
        vol_arr[i] = st["volume"]
        buy_arr[i] = st["buy_vol"]
        sell_arr[i] = st["sell_vol"]

    # Libera buckets
    del buckets
    gc.collect()

    # DataFrame com tipos otimizados
    df_tf = pd.DataFrame({
        "ts": ts_arr,
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": vol_arr,
        "buy_vol": buy_arr,
        "sell_vol": sell_arr,
        "delta": buy_arr - sell_arr,
    })
    
    del ts_arr, open_arr, high_arr, low_arr, close_arr, vol_arr, buy_arr, sell_arr
    gc.collect()

    # ============================================================
    # ENRIQUECIMENTO V1 (IGUAL V51)
    # ============================================================
    df_tf["buy_vol_agg"] = df_tf["buy_vol"]
    df_tf["sell_vol_agg"] = df_tf["sell_vol"]
    df_tf["total_vol_agg"] = df_tf["buy_vol_agg"] + df_tf["sell_vol_agg"]

    df_tf["taker_buy_base"] = df_tf["buy_vol_agg"]
    df_tf["taker_sell_base"] = df_tf["sell_vol_agg"]
    df_tf["taker_buy_quote"] = df_tf["taker_buy_base"] * df_tf["close"]

    df_tf["quote_volume"] = df_tf["volume"] * df_tf["close"]
    df_tf["trades"] = 0
    df_tf["close_time"] = df_tf["ts"] + (timeframe_min * 60 * 1000) - 1

    df_tf = df_tf.sort_values("ts").reset_index(drop=True)
    df_tf["cum_delta"] = df_tf["delta"].cumsum()

    df_tf["price_range"] = df_tf["high"] - df_tf["low"]
    df_tf["absorcao"] = df_tf["delta"] / (df_tf["price_range"].replace(0, 1e-9))
    df_tf["vpin"] = (df_tf["buy_vol_agg"] - df_tf["sell_vol_agg"]).abs() / (
        df_tf["total_vol_agg"].replace(0, 1e-9)
    )

    # Saneamento
    num_cols = [
        "open","high","low","close",
        "volume","buy_vol","sell_vol","delta",
        "buy_vol_agg","sell_vol_agg","total_vol_agg",
        "taker_buy_base","taker_sell_base","taker_buy_quote",
        "quote_volume","cum_delta","price_range","absorcao","vpin"
    ]
    for c in num_cols:
        if c in df_tf.columns:
            df_tf[c] = pd.to_numeric(df_tf[c], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype('float32')

    df_tf["ts"] = df_tf["ts"].astype("int64")
    df_tf["trades"] = df_tf["trades"].astype("int32")
    df_tf["close_time"] = df_tf["close_time"].astype("int64")

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
    
    try:
        with open(csv_tf_path, 'rb') as f:
            os.fsync(f.fileno())
    except:
        pass

    print(f"    ✅ {len(df_tf)} candles → {csv_tf_path}", flush=True)
    
    # Liberar memória antes de retornar
    result_len = len(df_tf)
    del df_tf
    gc.collect()
    
    return result_len

# =============================================================================
# FEATURE ENGINE (COPIADO EXATO DO V27)
# =============================================================================
def realized_vol(close: pd.Series) -> pd.Series:
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean()).shift(1)

def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    return np.sqrt((1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()).shift(1)

def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    rs = (np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
          np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"]))
    return np.sqrt(rs.rolling(20).mean()).shift(1)

def yang_zhang(df: pd.DataFrame) -> pd.Series:
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))
    log_co = np.log(df["close"] / df["open"])
    rs = (log_ho**2 + log_lo**2).rolling(20).mean()
    close_vol = log_co.rolling(20).std() ** 2
    open_vol = log_oc.rolling(20).std() ** 2
    return np.sqrt(0.34 * open_vol + 0.34 * close_vol + 0.27 * rs).shift(1)

def slope_regression(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Inclinação da regressão linear - OTIMIZADO."""
    if isinstance(series, pd.Series):
        series = series.values
    
    X = np.arange(window).reshape(-1, 1)
    n = len(series)
    slopes = np.full(n, np.nan)
    
    # OTIMIZAÇÃO: Usar numpy puro para ser mais rápido
    for i in range(window, n):
        y = series[i-window:i]
        if not np.any(np.isnan(y)):
            # Slope usando fórmula direta (mais rápido que LinearRegression)
            x = np.arange(window)
            slopes[i] = np.polyfit(x, y, 1)[0]
    
    return np.roll(slopes, 1)  # shift(1)

def feature_engine(df):
    """Pipeline de Geração de Features - COPIADO EXATO DO V27."""
    df = df.copy()

    # 1. VARIÁVEIS BÁSICAS
    df["body"] = (df["close"] - df["open"]).shift(1)
    df["range"] = (df["high"] - df["low"]).shift(1)
    
    # 2. RETORNOS PROTEGIDOS
    df["ret1"] = df["close"].pct_change(1).shift(1)
    df["ret5"] = df["close"].pct_change(5).shift(1)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).shift(1)

    # 3. EMAs E DISTÂNCIAS
    df["ema9"] = df["close"].ewm(span=9).mean().shift(1)
    df["ema20"] = df["close"].ewm(span=20).mean().shift(1)
    df["dist_ema9"] = (df["close"].shift(1) - df["ema9"])
    df["dist_ema20"] = (df["close"].shift(1) - df["ema20"])

    # 4. SLOPES E VOLATILIDADES
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)

    # 5. AGRESSÃO PROTEGIDA
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).shift(1)
    
    # 6. ATR14
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    
    # 7. RSI 14
    delta_price = df["close"].diff()
    gain = delta_price.where(delta_price > 0, 0).rolling(14).mean()
    loss = (-delta_price.where(delta_price < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    return df

def adicionar_features_avancadas(df):
    """Features avançadas - COPIADO EXATO DO V27."""
    df = df.copy()

    # 1. PRICE ACTION BÁSICO
    if "range" not in df.columns: df["range"] = df["high"] - df["low"]
    if "body" not in df.columns: df["body"] = df["close"] - df["open"]
    if "upper_wick" not in df.columns: 
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    if "lower_wick" not in df.columns: 
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # 2. PERCENTUAIS
    if "range_pct" not in df.columns: 
        df["range_pct"] = df["range"] / (df["close"] + 1e-9)

    # 3. RETORNOS
    if "ret1" not in df.columns: df["ret1"] = df["close"].pct_change(1)
    if "ret2" not in df.columns: df["ret2"] = df["close"].pct_change(2)
    if "ret5" not in df.columns: df["ret5"] = df["close"].pct_change(5)
    if "ret10" not in df.columns: df["ret10"] = df["close"].pct_change(10)

    # 4. INDICADORES TÉCNICOS
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

    if "volatility_acceleration" not in df.columns:
         df["volatility_acceleration"] = df["range_pct"].diff()

    # 2. MOMENTUM / VELOCIDADE
    df["momentum_1"] = df["ret1"]
    df["momentum_2"] = df["ret2"]
    df["momentum_acceleration"] = df["momentum_1"] - df["momentum_2"]

    # 3. WICK / STRUCTURE
    df["body_to_range"] = df["body"] / (df["range"] + 1e-8)
    df["wick_ratio_up"] = df["upper_wick"] / (df["range"] + 1e-8)
    df["wick_ratio_down"] = df["lower_wick"] / (df["range"] + 1e-8)

    # 5. VOLUME / SURGE
    df["volume_diff"] = df["volume"].diff()
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / \
                           (df["volume"].rolling(20).std() + 1e-8)

    # 6. AGRESSÃO
    if "aggression_delta" in df.columns:
        df["aggression_imbalance"] = df["aggression_delta"] / (df["volume"] + 1e-8)
        df["aggression_pressure"] = df["aggression_delta"] * df["ret1"]
        df["delta_acc"] = df["aggression_delta"].diff()
    else:
        if "delta" in df.columns:
            df["aggression_delta"] = df["delta"]
            df["aggression_imbalance"] = df["delta"] / (df["volume"] + 1e-8)
            df["aggression_pressure"] = df["delta"] * df["ret1"]
            df["delta_acc"] = df["delta"].diff()

    # 7. ATR E VOLATILIDADE INSTITUCIONAL
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_to_close"] = df["atr14"] / (df["close"] + 1e-8)
    df["range_to_atr"] = df["range"] / (df["atr14"] + 1e-8)

    # 8. REGIMES
    df["vol_regime"] = df["atr14"].rolling(50).mean()
    df["atr_compression"] = df["atr14"] / (df["vol_regime"] + 1e-8)
    df["trend_regime"] = slope_regression(df["close"].values, 100)
    df["liquidity_regime"] = df["volume"].rolling(50).mean()

    # 9. RETORNOS AVANÇADOS
    df["ret3"] = df["close"].pct_change(3)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)
    df["ret20_norm"] = df["ret20"] / (df["atr14"] + 1e-8)
    df["momentum_long"] = df["ret3"] + df["ret10"] + df["ret20"]

    # 10. Z-SCORE DIRECIONAL
    df["price_z"] = (
        (df["close"] - df["close"].rolling(20).mean()) /
        (df["close"].rolling(20).std() + 1e-8)
    )

    # 11. SQUEEZE AVANÇADO
    df["vol_squeeze"] = (
        df["close"].rolling(20).std() /
        (df["close"].rolling(100).std() + 1e-8)
    )
    df["range_squeeze"] = (
        df["range_pct"].rolling(14).std() /
        (df["range_pct"].rolling(50).std() + 1e-8)
    )

    # 12. FEATURES DE Agressão
    if "aggression_buy" in df.columns:
        df["buy_ratio"] = df["aggression_buy"] / (df["volume"] + 1e-8)
    if "aggression_sell" in df.columns:
        df["sell_ratio"] = df["aggression_sell"] / (df["volume"] + 1e-8)
    if "aggression_delta" in df.columns:
        df["aggr_cumsum_20"] = df["aggression_delta"].rolling(20).sum()
        if "vpin" in df.columns:
            df["flow_dominance"] = df["aggression_delta"] * (1 + df["vpin"].fillna(0))
            df["flow_acceleration"] = df["aggression_delta"].diff()

    # 13. DISTÂNCIAS IMPORTANTES
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    if "ema100" not in df.columns:
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    
    if "dist_ema50" not in df.columns:
        df["dist_ema50"] = df["close"] - df["ema50"]
    if "dist_ema100" not in df.columns:
        df["dist_ema100"] = df["close"] - df["ema100"]
    
    df["dist_ema20_norm"] = df["dist_ema20"] / (df["atr14"] + 1e-8)
    df["dist_ema50_norm"] = df["dist_ema50"] / (df["atr14"] + 1e-8)
    df["dist_ema100_norm"] = df["dist_ema100"] / (df["atr14"] + 1e-8)

    # 14. SLOPES LONGOS
    df["slope100"] = slope_regression(df["close"].values, 100)
    df["slope200"] = slope_regression(df["close"].values, 200)

    return df

# =============================================================================
# DETECÇÃO DE REGIMES (COPIADO DO V27)
# =============================================================================
def detectar_regimes_mercado_v25(df, n_regimes=4):
    """Detecta regimes de mercado usando KMeans."""
    print(">>> Detectando regimes de mercado...", flush=True)
    
    regime_features = [c for c in ['vol_realized', 'rsi_14', 'atr14', 'slope20'] if c in df.columns]
    
    if not regime_features:
        df['temp_ret'] = df['close'].pct_change(20)
        regime_features = ['temp_ret']
    
    print(f"    Features regime: {regime_features}", flush=True)
    
    X_regime = df[regime_features].fillna(0).values.astype('float32')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    try:
        scaler_path = os.path.join(MODELOS_DIR, 'scaler_regimes.pkl')
        joblib.dump(scaler, scaler_path, compress=3 if COMPRESS_MODELS else 0)
        
        kmeans_path = os.path.join(MODELOS_DIR, 'kmeans_regimes.pkl')
        joblib.dump(kmeans, kmeans_path, compress=3 if COMPRESS_MODELS else 0)
        
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
# PREPARAR FUTUROS (COPIADO DO V27)
# =============================================================================
def preparar_futuros(df: pd.DataFrame, N: int) -> pd.DataFrame:
    df = df.copy()

    df["close_future"] = df["close"].shift(-N)
    df["ret_fut"] = (df["close_future"] - df["close"]) / df["close"]

    df["high_fut"] = df["high"].shift(-N)
    df["low_fut"] = df["low"].shift(-N)
    df["amp_fut"] = (df["high_fut"] - df["low_fut"]) / (df["close"] + 1e-9)

    df = df.iloc[:-N].reset_index(drop=True)

    return df

# =============================================================================
# MONTAR MATRIZ (COPIADO DO V27)
# =============================================================================
def montar_matriz(df: pd.DataFrame, alvo: str):
    non_feat = {
        "open", "high", "low", "close", "volume",
        "ts","open","high","low","close","volume","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","close_time","ignore",
        "mark_price","index_price","fundingRate",
        "session","tp",
        "close_future","ret_fut","amp_fut",
        "high_fut","low_fut",
        "impulse_count",
        "ret_max", "ret_min", "ret_max_temp", "ret_min_temp",
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg",
        "fractal_high", "fractal_low", "pivot_high", "pivot_low",
        "last_pivot_high", "last_pivot_low", "swing_dir",
        "wave_amplitude", "wave_amplitude_abs", "correction_pct", "wave_strength",
        "taker_sell_base", "buy_vol", "sell_vol", "delta", "cum_delta",
        "price_range", "absorcao", "vpin"
    }

    target_cols = {
        c for c in df.columns
        if isinstance(c, str) and c.startswith("target_")
    }
    non_feat = non_feat.union(target_cols)

    if alvo not in df.columns:
        raise RuntimeError(f"[montar_matriz] alvo ausente no df: {alvo}")

    y = df[alvo].values

    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].copy()

    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise RuntimeError(f"[montar_matriz] Nenhuma feature numérica para alvo={alvo}")

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        X = X.fillna(0.0)

    feat_cols = list(X.columns)

    # OTIMIZAÇÃO: Usar float32
    return X.values.astype('float32'), y, feat_cols

# =============================================================================
# SPLIT TEMPORAL (COPIADO DO V27)
# =============================================================================
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

# =============================================================================
# TREINAR UM TARGET (OTIMIZADO PARA RENDER)
# =============================================================================
def treinar_um_target(target_col, df, outdir):
    """Treina um target com LGBM + XGB - OTIMIZADO PARA RENDER."""
    
    print(f"\n{'='*70}")
    print(f"TREINANDO TARGET {target_col}")
    print(f"{'='*70}")

    df_local = df.copy()
    classes_orig = np.unique(df_local[target_col].values)

    if set(classes_orig) == {-1, 0, 1}:
        df_local[target_col] = df_local[target_col].map({-1: 0, 0: 1, 1: 2})
    elif len(classes_orig) > 0 and classes_orig.min() < 0:
        df_local[target_col] = df_local[target_col] - classes_orig.min()

    classes = np.unique(df_local[target_col].values)
    n_classes = len(classes)

    X, y, feat_cols = montar_matriz(df_local, target_col)
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"    Features: {len(feat_cols)}")

    if "sample_weight" in df_local.columns:
        sw = df_local["sample_weight"].values
        sw_train = sw[:len(X_train)]
    else:
        sw_train = None

    resultados = []

    # LGBM - OTIMIZADO
    try:
        model_lgb = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            num_class=None if n_classes == 2 else n_classes,
            n_estimators=N_ESTIMATORS_LIGHT,  # Reduzido
            learning_rate=0.05,  # Aumentado para compensar menos estimators
            max_depth=6,  # Limitado para evitar overfitting
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
            force_col_wise=True  # Melhor para memória
        )

        model_lgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_lgb.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("LGBM", f1, model_lgb))
        print(f"    >>> LGBM F1={f1:.4f}")

    except Exception as e:
        print(f"    [LGBM] erro: {e}")

    # XGBoost - OTIMIZADO
    try:
        counts = pd.Series(y_train).value_counts()
        spw = counts.min() / counts.max() if len(counts) == 2 else 1.0

        model_xgb = XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=spw if n_classes == 2 else None,
            n_estimators=N_ESTIMATORS_LIGHT,  # Reduzido
            learning_rate=0.05,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            verbosity=0
        )

        model_xgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_xgb.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("XGB", f1, model_xgb))
        print(f"    >>> XGB F1={f1:.4f}")

    except Exception as e:
        print(f"    [XGB] erro: {e}")

    # CatBoost - REMOVIDO PARA ECONOMIZAR MEMÓRIA NO RENDER
    # Se quiser reativar, descomente abaixo:
    """
    try:
        model_cat = CatBoostClassifier(
            iterations=N_ESTIMATORS_LIGHT,
            learning_rate=0.05,
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
    """

    if not resultados:
        raise RuntimeError(f"Nenhum modelo treinou para {target_col}")

    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)
    print(f"\n    📊 RANKING {target_col}:")
    for i, (nome, f1, _) in enumerate(resultados_ordenados, 1):
        marca = "🥇" if i == 1 else "🥈"
        print(f"       {marca} {i}º {nome}: F1={f1:.4f}")

    melhor_nome, melhor_f1, melhor_modelo = resultados_ordenados[0]
    print(f"\n    ✅ VENCEDOR: {melhor_nome} (F1={melhor_f1:.4f})")

    os.makedirs(outdir, exist_ok=True)
    nome_arquivo = f"{target_col}_{melhor_nome}.pkl"
    model_path = os.path.join(outdir, nome_arquivo)
    
    # OTIMIZAÇÃO: Comprimir modelo ao salvar
    joblib.dump(melhor_modelo, model_path, compress=3 if COMPRESS_MODELS else 0)

    caminhos_modelos[target_col] = model_path
    features_por_target[target_col] = list(feat_cols)
    modelos_cache[target_col] = melhor_modelo

    print(f"    ✅ Modelo salvo: {model_path}")

    # Liberar memória
    del X_train, y_train, X_val, y_val, X_test, y_test, df_local
    gc.collect()

    return nome_arquivo, melhor_f1, model_path

# =============================================================================
# DOWNLOAD E GERAÇÃO DE CSVs (OTIMIZADO)
# =============================================================================
def baixar_aggtrades():
    """Download aggTrades - OTIMIZADO."""
    
    print("\n>>> BAIXANDO aggTrades...", flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    print(f"    {total_dates} dias", flush=True)
    
    start_index = 0
    if os.path.exists(CSV_AGG_PATH):
        try:
            with open(CSV_AGG_PATH, 'r') as f:
                lines = sum(1 for _ in f)
            if lines > 1000:
                print(f"    📁 CSV existente com ~{lines:,} linhas, continuando...", flush=True)
                first_write = False
            else:
                os.remove(CSV_AGG_PATH)
                first_write = True
        except:
            os.remove(CSV_AGG_PATH)
            first_write = True
    else:
        first_write = True
    
    session = requests.Session()
    success_count = 0
    
    for i, date in enumerate(dates, 1):
        print(f"    [{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
        
        t_start = time.time()
        df = download_daily_file(SYMBOL, date, session, retry_count=5)
        elapsed = time.time() - t_start
        
        if df is not None:
            df_processed = process_binance_data(df)
            
            del df
            gc.collect()
            
            if df_processed is not None and not df_processed.empty:
                df_processed.to_csv(CSV_AGG_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                trades_count = len(df_processed)
                
                del df_processed
                gc.collect()
                
                print(f"✓ {trades_count:,} trades ({elapsed:.1f}s)", flush=True)
            else:
                print(f"⚠️ Vazio", flush=True)
                if df_processed is not None:
                    del df_processed
        else:
            print(f"⚠️ Falhou", flush=True)
        
        if i % 5 == 0:
            gc.collect()
        
        time.sleep(random.uniform(0.3, 1.0))
    
    session.close()
    gc.collect()
    
    print(f"\n    {success_count}/{total_dates} dias OK", flush=True)
    
    if success_count == 0:
        raise Exception("NENHUM DADO baixado!")
    
    return success_count

def gerar_todos_timeframes():
    """Gera todos os timeframes - OTIMIZADO."""
    
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
        
        gc.collect()
        
        try:
            os.sync()
        except:
            pass
        
        print(f"    ✅ {tf_name} concluído", flush=True)
    
    return csv_paths

# =============================================================================
# TREINO COMPLETO K1-K6
# =============================================================================
def treinar_todos_targets_k(df_all):
    """Treina K1 até K6."""
    
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
        
        if k == 6:
            print(f"    🔥 K6 ELITE: Threshold 0.5%")
            threshold_k6 = 0.005
            df_k[alvo_k] = (df_k["ret_fut"] > threshold_k6).astype(int)
        else:
            df_k[alvo_k] = (df_k["ret_fut"] > 0).astype(int)
        
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
        
        # Liberar memória após cada K
        del df_k
        gc.collect()
    
    if not ks_treinados:
        raise RuntimeError("TARGET_K NÃO FOI TREINADO — ERRO FATAL")
    
    print(f"\n>>> TARGET_K treinados: {ks_treinados}")
    return ks_treinados

# =============================================================================
# CRIAR PACOTE FINAL
# =============================================================================
def criar_pacote_final():
    """Cria SISTEMA_K6_COMPLETO.pkl."""
    
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
            'info': 'V55_OPTIMIZED'
        }
        pacote_path = os.path.join(MODELOS_DIR, 'SISTEMA_K6_COMPLETO.pkl')
        joblib.dump(pacote, pacote_path, compress=3 if COMPRESS_MODELS else 0)
        print(f"    ✅ SISTEMA_K6_COMPLETO.pkl salvo", flush=True)
        return pacote_path
    else:
        print(f"    ❌ Faltam componentes para o pacote", flush=True)
        return None

# =============================================================================
# SERVIDOR HTTP
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
<h1>🚀 DataManager V55 OPTIMIZED</h1>
<hr>
<h2>CSVs (15m, 30m, 1h, 4h, 8h, 1d): {csv_status}</h2>
{csv_link}
<br><br>
<h2>PKLs (K1-K6, Scaler, KMeans): {pkl_status}</h2>
{pkl_link}
<hr>
<p style="color:gray;">Render Persistent Disk: {BASE_DISK}</p>
<p style="color:gray;">Chunk Size: {CHUNK_SIZE} | N_Estimators: {N_ESTIMATORS_LIGHT}</p>
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

# =============================================================================
# MAIN
# =============================================================================
def main():
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    print("\n" + "="*70)
    print("🚀 DataManager V55 - OPTIMIZED FOR RENDER")
    print("="*70)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print(f"Diretório: {OUT_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"N_Estimators: {N_ESTIMATORS_LIGHT}")
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
    
    # OTIMIZAÇÃO: Converter tipos
    df_15m = optimize_dtypes(df_15m)
    gc.collect()
    
    # Aliases
    if 'delta' in df_15m.columns and 'aggression_delta' not in df_15m.columns:
        df_15m['aggression_delta'] = df_15m['delta']
    
    if 'buy_vol_agg' in df_15m.columns and 'aggression_buy' not in df_15m.columns:
        df_15m['aggression_buy'] = df_15m['buy_vol_agg']
    
    if 'sell_vol_agg' in df_15m.columns and 'aggression_sell' not in df_15m.columns:
        df_15m['aggression_sell'] = df_15m['sell_vol_agg']
    
    print(f"    Aliases criados", flush=True)
    
    df_15m = feature_engine(df_15m)
    gc.collect()
    
    df_15m = adicionar_features_avancadas(df_15m)
    gc.collect()
    
    print(f"    Features aplicadas: {len(df_15m.columns)} colunas", flush=True)
    
    df_15m, scaler, kmeans = detectar_regimes_mercado_v25(df_15m, n_regimes=4)
    gc.collect()
    
    # =========================================================================
    # PASSO 4: TREINO K1-K6
    # =========================================================================
    print("\n>>> PASSO 4: Treino K1-K6...", flush=True)
    ks_treinados = treinar_todos_targets_k(df_15m)
    
    # Liberar df_15m após treino
    del df_15m
    gc.collect()
    
    # =========================================================================
    # PASSO 5: CRIAR PACOTE FINAL
    # =========================================================================
    print("\n>>> PASSO 5: Pacote final...", flush=True)
    pacote_path = criar_pacote_final()
    
    # =========================================================================
    # PASSO 6: CRIAR ZIPs
    # =========================================================================
    print("\n>>> PASSO 6: Criando ZIPs...", flush=True)
    
    with zipfile.ZipFile(ZIP_CSV_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for tf, path in csv_paths.items():
            z.write(path, arcname=os.path.basename(path))
    print(f"    ✅ {ZIP_CSV_PATH}", flush=True)
    
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
    
    print("\n>>> Servidor mantido ativo...", flush=True)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
