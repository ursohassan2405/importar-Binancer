# ============================================================
# DataManager_V54_FINAL_CORRIGIDO.py
# PENDLEUSDT – Binance Data Vision + Treino XGBoost
# OBJETIVO: Gerar dados IDÊNTICOS ao ANT (18 colunas) + treinar modelo
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import joblib
import warnings
warnings.filterwarnings("ignore")

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =========================
# CONFIGURAÇÃO
# =========================
SYMBOL = "PENDLEUSDT"

# TESTE: 10 dias
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 1, 10, 23, 59, 59)

# PRODUÇÃO: 1 ano (descomentar para produção)
# START_DT = datetime(2025, 1, 1, 0, 0, 0)
# END_DT = datetime(2025, 12, 31, 23, 59, 59)

# Detectar ambiente (Render vs Local)
if os.path.exists("/opt/render/project"):
    BASE_DIR = "/opt/render/project"
    print(f"⚠️ Usando {BASE_DIR} (pode ser apagado!)")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório de output
date_str = f"{START_DT.strftime('%Y%m%d')}_a_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DIR, f"pendle_agg_{date_str}")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_AGG_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
CSV_15M_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_15m.csv")
ZIP_PATH = OUT_DIR + ".zip"

print(f"📂 OUT_DIR: {OUT_DIR}")

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

# =========================
# FUNÇÕES DE DOWNLOAD
# =========================
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

# =========================
# GERAÇÃO DO CSV 15M - 18 COLUNAS (IGUAL ANT)
# =========================
def gerar_15m_tratado(csv_agg_path, csv_15m_path, timeframe_min=15, min_val_usd=500, chunksize=200_000):
    """
    Gera CSV 15m com EXATAMENTE 18 colunas igual ao ANT:
    ts, open, high, low, close, volume, quote_volume, trades,
    taker_buy_base, taker_buy_quote, close_time, cum_delta,
    total_vol_agg, buy_vol_agg, sell_vol_agg, vpin, price_range, absorcao
    """
    print(">>> Gerando dataset 15m (18 colunas igual ANT)...", flush=True)

    buckets = {}

    for chunk in pd.read_csv(csv_agg_path, chunksize=chunksize):
        chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")
        chunk["side"] = pd.to_numeric(chunk["side"], errors="coerce")

        chunk = chunk.dropna(subset=["ts", "price", "qty", "side"])
        if chunk.empty:
            continue

        dt = pd.to_datetime(chunk["ts"].astype("int64"), unit="ms", utc=True)
        bucket_dt = dt.dt.floor(f"{timeframe_min}min")
        bucket_ms = (bucket_dt.astype("int64") // 10**6).astype("int64")
        chunk = chunk.assign(bucket_ms=bucket_ms)

        val_usd = chunk["price"] * chunk["qty"]
        is_whale = val_usd >= float(min_val_usd)

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
                    "taker_buy_base": 0.0,  # TODO volume buy (não filtrado)
                    "buy_vol": 0.0,         # SÓ baleias
                    "sell_vol": 0.0,        # SÓ baleias
                    "trades": 0,
                }
                buckets[bms] = st
            else:
                if price > st["high"]:
                    st["high"] = float(price)
                if price < st["low"]:
                    st["low"] = float(price)
                st["close"] = float(price)

            st["volume"] += float(qty)
            st["trades"] += 1
            
            # taker_buy_base = TODO volume buy (side=0 significa buyer maker = comprador agressivo)
            if side == 0:
                st["taker_buy_base"] += float(qty)

            # Baleias (>$500)
            if whale:
                if side == 0:
                    st["buy_vol"] += float(qty)
                else:
                    st["sell_vol"] += float(qty)

    if not buckets:
        raise RuntimeError("Nenhum bucket 15m gerado!")

    # Monta DataFrame
    rows = []
    for bms in sorted(buckets.keys()):
        st = buckets[bms]
        rows.append(st)

    df_15m = pd.DataFrame(rows)
    
    # Calcular campos derivados
    df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
    df_15m["taker_buy_quote"] = df_15m["taker_buy_base"] * df_15m["close"]
    df_15m["close_time"] = df_15m["ts"] + (timeframe_min * 60 * 1000) - 1
    
    # Delta e cum_delta
    df_15m["delta"] = df_15m["buy_vol"] - df_15m["sell_vol"]
    df_15m["cum_delta"] = df_15m["delta"].cumsum()
    
    # Métricas de fluxo
    df_15m["total_vol_agg"] = df_15m["buy_vol"] + df_15m["sell_vol"]
    df_15m["buy_vol_agg"] = df_15m["buy_vol"]
    df_15m["sell_vol_agg"] = df_15m["sell_vol"]
    
    df_15m["price_range"] = df_15m["high"] - df_15m["low"]
    df_15m["vpin"] = (df_15m["buy_vol_agg"] - df_15m["sell_vol_agg"]).abs() / (
        df_15m["total_vol_agg"].replace(0, 1e-9)
    )
    df_15m["absorcao"] = df_15m["delta"] / (df_15m["price_range"].replace(0, 1e-9))

    # ORDEM DAS 18 COLUNAS - IGUAL ANT
    cols_18 = [
        "ts", "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "close_time", "cum_delta",
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg", "vpin", "price_range", "absorcao"
    ]
    
    df_15m = df_15m[cols_18]
    
    # Saneamento
    for c in cols_18:
        if c in ["ts", "trades", "close_time"]:
            df_15m[c] = pd.to_numeric(df_15m[c], errors="coerce").fillna(0).astype("int64")
        else:
            df_15m[c] = pd.to_numeric(df_15m[c], errors="coerce").replace(
                [float("inf"), float("-inf")], 0.0
            ).fillna(0.0)

    df_15m.to_csv(csv_15m_path, index=False)
    
    print(f"    ✅ {len(df_15m)} candles", flush=True)
    print(f"    🐋 Candles com baleias: {(df_15m['buy_vol_agg'] > 0).sum()}/{len(df_15m)}", flush=True)
    print(f"    📊 Colunas: {len(df_15m.columns)} (esperado: 18)", flush=True)
    
    return df_15m

# =========================
# FEATURE ENGINE (DO V27)
# =========================
from sklearn.linear_model import LinearRegression

def slope_regression(series_values, window=20):
    """Inclinação da regressão linear."""
    X = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * window
    for i in range(window, len(series_values)):
        y = series_values[i-window:i]
        if len(y) == window and not np.any(np.isnan(y)):
            slopes.append(LinearRegression().fit(X, y).coef_[0])
        else:
            slopes.append(np.nan)
    return np.array(slopes)

def realized_vol(close):
    """Volatilidade realizada."""
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean())

def yang_zhang(df):
    """Volatilidade Yang-Zhang."""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))
    log_co = np.log(df["close"] / df["open"])
    rs = (log_ho**2 + log_lo**2).rolling(20).mean()
    close_vol = log_co.rolling(20).std() ** 2
    open_vol = log_oc.rolling(20).std() ** 2
    return np.sqrt(0.34 * open_vol + 0.34 * close_vol + 0.27 * rs)

def feature_engine(df):
    """Pipeline de Features - Versão V27."""
    df = df.copy()

    # Price Action
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    
    # Retornos
    df["ret1"] = df["close"].pct_change(1)
    df["ret2"] = df["close"].pct_change(2)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # EMAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["dist_ema9"] = df["close"] - df["ema9"]
    df["dist_ema20"] = df["close"] - df["ema20"]

    # Slopes
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    
    # Volatilidades
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)
    
    # ATR
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Agressão (se disponível)
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"]
        df["aggression_delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
        df["aggression_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
    
    # Volume features
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1e-9)
    df["volume_zscore"] = (df["volume"] - df["volume_ma20"]) / (df["volume"].rolling(20).std() + 1e-9)
    
    # Percentuais
    df["range_pct"] = df["range"] / (df["close"] + 1e-9)
    df["body_pct"] = df["body"].abs() / (df["close"] + 1e-9)
    df["body_to_range"] = df["body"] / (df["range"] + 1e-9)
    
    # Wicks
    df["wick_ratio_up"] = df["upper_wick"] / (df["range"] + 1e-9)
    df["wick_ratio_down"] = df["lower_wick"] / (df["range"] + 1e-9)
    
    # Close position
    df["close_pos"] = (df["close"] - df["low"]) / (df["range"] + 1e-9)
    
    # Momentum
    df["momentum_1"] = df["ret1"]
    df["momentum_2"] = df["ret2"]
    df["momentum_accel"] = df["momentum_1"] - df["momentum_2"]
    
    # Volatility acceleration
    df["volatility_accel"] = df["range_pct"].diff()
    
    # Z-scores
    df["body_z"] = (df["body"] - df["body"].rolling(10).mean()) / (df["body"].rolling(10).std() + 1e-9)
    df["range_z"] = (df["range"] - df["range"].rolling(10).mean()) / (df["range"].rolling(10).std() + 1e-9)
    
    # Bollinger
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    
    # Keltner
    df["kc_mid"] = df["close"].ewm(span=20).mean()
    df["kc_upper"] = df["kc_mid"] + 1.5 * df["atr14"]
    df["kc_lower"] = df["kc_mid"] - 1.5 * df["atr14"]
    
    # Squeeze
    df["squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)
    
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_ma"] = pd.Series(obv).rolling(20).mean().values
    
    # ADX (simplificado)
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr_14 = df["atr14"]
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(14).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    
    # Williams %R
    df["willr"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-9)
    
    # CCI
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_ma = tp.rolling(20).mean()
    tp_std = tp.rolling(20).std()
    df["cci"] = (tp - tp_ma) / (0.015 * tp_std + 1e-9)
    
    # MFI
    if "taker_buy_base" in df.columns:
        mf_pos = (tp * df["taker_buy_base"]).rolling(14).sum()
        mf_neg = (tp * (df["volume"] - df["taker_buy_base"])).rolling(14).sum()
        df["mfi"] = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-9)))
    
    return df

# =========================
# TARGET K (DO V27)
# =========================
def criar_targets_k(df, lookforward=8):
    """Cria targets K1-K5 baseados em movimento futuro."""
    df = df.copy()
    
    # Retorno futuro
    df["close_future"] = df["close"].shift(-lookforward)
    df["ret_fut"] = (df["close_future"] - df["close"]) / df["close"]
    
    # High/Low futuros para movimento máximo
    df["high_fut"] = df["high"].rolling(lookforward).max().shift(-lookforward)
    df["low_fut"] = df["low"].rolling(lookforward).min().shift(-lookforward)
    
    # Amplitude do movimento
    df["amp_fut"] = df["ret_fut"].abs()
    
    # Thresholds baseados em percentis do movimento
    p50 = df["amp_fut"].quantile(0.50)
    p60 = df["amp_fut"].quantile(0.60)
    p70 = df["amp_fut"].quantile(0.70)
    p80 = df["amp_fut"].quantile(0.80)
    p90 = df["amp_fut"].quantile(0.90)
    
    print(f">>> Percentis de movimento:")
    print(f"    P50: {p50*100:.3f}%")
    print(f"    P60: {p60*100:.3f}%")
    print(f"    P70: {p70*100:.3f}%")
    print(f"    P80: {p80*100:.3f}%")
    print(f"    P90: {p90*100:.3f}%")
    
    # Target K1: Movimento > P50 (mais fácil)
    df["target_K1"] = 0
    df.loc[df["ret_fut"] > p50, "target_K1"] = 1
    df.loc[df["ret_fut"] < -p50, "target_K1"] = -1
    
    # Target K2: Movimento > P60
    df["target_K2"] = 0
    df.loc[df["ret_fut"] > p60, "target_K2"] = 1
    df.loc[df["ret_fut"] < -p60, "target_K2"] = -1
    
    # Target K3: Movimento > P70
    df["target_K3"] = 0
    df.loc[df["ret_fut"] > p70, "target_K3"] = 1
    df.loc[df["ret_fut"] < -p70, "target_K3"] = -1
    
    # Target K4: Movimento > P80
    df["target_K4"] = 0
    df.loc[df["ret_fut"] > p80, "target_K4"] = 1
    df.loc[df["ret_fut"] < -p80, "target_K4"] = -1
    
    # Target K5: Movimento > P90 (mais difícil)
    df["target_K5"] = 0
    df.loc[df["ret_fut"] > p90, "target_K5"] = 1
    df.loc[df["ret_fut"] < -p90, "target_K5"] = -1
    
    # Target binário simples (para teste rápido)
    df["target_A_bin"] = (df["ret_fut"] > 0).astype(int)
    
    return df

# =========================
# TREINO DO MODELO (DO V27)
# =========================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def treinar_modelo_v27(df, out_dir):
    """
    Treina modelo XGBoost no estilo V27.
    """
    print("\n" + "="*60)
    print("🚀 TREINO DO MODELO - ESTILO V27")
    print("="*60)
    
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("❌ XGBoost não instalado. Instale com: pip install xgboost")
        return None, None, None, None
    
    # Aplicar features
    print(">>> Aplicando Feature Engine...")
    df = feature_engine(df)
    
    # Criar targets
    print(">>> Criando targets K...")
    df = criar_targets_k(df, lookforward=8)
    
    # Remover NaN iniciais
    df = df.dropna().reset_index(drop=True)
    print(f">>> Dados após limpeza: {len(df)} linhas")
    
    if len(df) < 100:
        print("❌ Dados insuficientes para treino!")
        return None, None, None, None
    
    # Selecionar features (excluir targets e colunas não-feature)
    non_feat = {
        "ts", "open", "high", "low", "close", "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "close_time", "cum_delta",
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg", "vpin", "price_range", "absorcao",
        "close_future", "ret_fut", "high_fut", "low_fut", "amp_fut",
        "target_K1", "target_K2", "target_K3", "target_K4", "target_K5", "target_A_bin"
    }
    
    feat_cols = [c for c in df.columns if c not in non_feat and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    print(f">>> Features: {len(feat_cols)}")
    
    # Preparar X e y
    target_col = "target_A_bin"  # Começar com target binário simples
    
    X = df[feat_cols].values
    y = df[target_col].values
    
    # Verificar se y tem valores válidos
    unique_y = np.unique(y)
    print(f">>> Classes no target: {unique_y}")
    
    if len(unique_y) < 2:
        print("❌ Target não tem variação suficiente!")
        return None, None, None, None
    
    # Split temporal (70/30)
    n = len(X)
    train_end = int(n * 0.7)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]
    
    print(f">>> Split temporal...")
    print(f"    Train: {len(X_train)}")
    print(f"    Test: {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ Split resultou em conjuntos vazios!")
        return None, None, None, None
    
    # Calcular peso temporal (dados mais recentes = mais peso)
    print(">>> Calculando peso temporal...")
    ts = df["ts"].values[:train_end]
    ts_max = ts.max()
    ts_min = ts.min()
    
    if ts_max > ts_min:
        idade_norm = (ts - ts_min) / (ts_max - ts_min)
        sample_weight_train = 0.5 + 0.5 * idade_norm  # Peso entre 0.5 e 1.0
    else:
        sample_weight_train = np.ones(len(X_train))
    
    print(f"    ✅ Pesos: min={sample_weight_train.min():.2f}, max={sample_weight_train.max():.2f}")
    
    # Treinar XGBoost (configuração do V27)
    print(">>> Treinando XGBoost...")
    
    modelo = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    modelo.fit(
        X_train, y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Avaliar
    from sklearn.metrics import accuracy_score, f1_score
    
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n>>> RESULTADOS:")
    print(f"    Accuracy: {acc*100:.2f}%")
    print(f"    F1-Score: {f1:.4f}")
    
    # Criar scaler e kmeans para regimes (como no V27)
    print("\n>>> Criando detector de regimes...")
    
    regime_features = ['vol_realized', 'atr14', 'slope20']
    regime_cols_exist = [c for c in regime_features if c in df.columns]
    
    if len(regime_cols_exist) >= 3:
        X_regime = df[regime_cols_exist].fillna(0).values
        
        scaler = StandardScaler()
        X_regime_scaled = scaler.fit_transform(X_regime)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_regime_scaled)
        
        df["market_regime"] = kmeans.predict(X_regime_scaled)
        print(f"    ✅ Regimes: {df['market_regime'].value_counts().to_dict()}")
    else:
        scaler = None
        kmeans = None
        print("    ⚠️ Features de regime insuficientes")
    
    # Salvar modelo
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, "modelo_xgb.pkl")
    joblib.dump(modelo, model_path)
    print(f"\n>>> Modelo salvo: {model_path}")
    
    if scaler is not None:
        scaler_path = os.path.join(out_dir, "scaler_regimes.pkl")
        joblib.dump(scaler, scaler_path)
        print(f">>> Scaler salvo: {scaler_path}")
    
    if kmeans is not None:
        kmeans_path = os.path.join(out_dir, "kmeans_regimes.pkl")
        joblib.dump(kmeans, kmeans_path)
        print(f">>> KMeans salvo: {kmeans_path}")
    
    # Salvar features usadas
    features_path = os.path.join(out_dir, "features.json")
    import json
    with open(features_path, "w") as f:
        json.dump({"features": feat_cols}, f, indent=2)
    print(f">>> Features salvas: {features_path}")
    
    return modelo, scaler, kmeans, feat_cols

# =========================
# UPLOAD CATBOX
# =========================
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

# =========================
# SERVIDOR HTTP
# =========================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download':
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', 'attachment; filename="PENDLEUSDT_data.zip"')
                self.end_headers()
                with open(ZIP_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP nao criado ainda')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            status = 'PRONTO!' if os.path.exists(ZIP_PATH) else 'Processando...'
            html = f'<html><body style="font-family:Arial;padding:50px;text-align:center;"><h1>PENDLEUSDT</h1><p>{status}</p><a href="/download">BAIXAR ZIP</a></body></html>'
            self.wfile.write(html.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    print(f">>> Servidor HTTP na porta {port}", flush=True)
    server.serve_forever()

# =========================
# MAIN
# =========================
def main():
    print("="*70)
    print("🚀 DataManager V54 FINAL CORRIGIDO")
    print("="*70)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print(f"Diretório: {OUT_DIR}")
    print("="*70)
    
    # Inicia servidor HTTP em background
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    # PASSO 1: Verificar se CSV 15m já existe
    print("\n>>> PASSO 1: Verificando CSVs existentes...")
    
    if os.path.exists(CSV_15M_PATH):
        df_15m = pd.read_csv(CSV_15M_PATH)
        print(f"    ✅ 15m já existe: {len(df_15m)} linhas, {len(df_15m.columns)} colunas")
        
        if len(df_15m.columns) == 18:
            print("    ✅ Estrutura correta (18 colunas)")
        else:
            print(f"    ⚠️ Estrutura diferente ({len(df_15m.columns)} colunas)")
            os.remove(CSV_15M_PATH)
            df_15m = None
    else:
        df_15m = None
        print("    ❌ 15m não encontrado, será gerado")
    
    # PASSO 2: Download de aggTrades se necessário
    if df_15m is None:
        print("\n>>> PASSO 2: Download de aggTrades...")
        
        dates = generate_date_range(START_DT, END_DT)
        total_dates = len(dates)
        print(f"    {total_dates} dias")
        
        if os.path.exists(CSV_AGG_PATH):
            os.remove(CSV_AGG_PATH)
        
        session = requests.Session()
        success_count = 0
        first_write = True
        
        for i, date in enumerate(dates, 1):
            print(f"    [{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
            
            df = download_daily_file(SYMBOL, date, session)
            
            if df is not None:
                df_processed = process_binance_data(df)
                
                if df_processed is not None and not df_processed.empty:
                    df_processed.to_csv(CSV_AGG_PATH, mode='a', header=first_write, index=False)
                    first_write = False
                    success_count += 1
                    print(f"✓ {len(df_processed):,} trades", flush=True)
                    del df, df_processed
                else:
                    print("⚠️", flush=True)
            else:
                print("⚠️", flush=True)
            
            time.sleep(random.uniform(0.3, 1.0))
        
        session.close()
        
        print(f"\n>>> Download: {success_count}/{total_dates} dias")
        
        if success_count == 0:
            print("❌ Nenhum dado baixado!")
            return
        
        # PASSO 3: Gerar CSV 15m
        print("\n>>> PASSO 3: Gerando CSV 15m...")
        df_15m = gerar_15m_tratado(CSV_AGG_PATH, CSV_15M_PATH)
    
    # PASSO 4: Treinar modelo
    print("\n>>> PASSO 4: Treinando modelo...")
    modelo, scaler, kmeans, feat_cols = treinar_modelo_v27(df_15m, OUT_DIR)
    
    if modelo is None:
        print("❌ Treino falhou!")
    else:
        print("✅ Treino concluído!")
    
    # PASSO 5: Criar ZIP
    print("\n>>> PASSO 5: Criando ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_15M_PATH, arcname="PENDLEUSDT_15m.csv")
        
        model_path = os.path.join(OUT_DIR, "modelo_xgb.pkl")
        if os.path.exists(model_path):
            z.write(model_path, arcname="modelo_xgb.pkl")
        
        scaler_path = os.path.join(OUT_DIR, "scaler_regimes.pkl")
        if os.path.exists(scaler_path):
            z.write(scaler_path, arcname="scaler_regimes.pkl")
        
        kmeans_path = os.path.join(OUT_DIR, "kmeans_regimes.pkl")
        if os.path.exists(kmeans_path):
            z.write(kmeans_path, arcname="kmeans_regimes.pkl")
    
    zip_size = os.path.getsize(ZIP_PATH) / (1024*1024)
    print(f">>> ZIP criado: {zip_size:.2f} MB")
    
    # PASSO 6: Upload para CatBox
    print("\n>>> PASSO 6: Upload para CatBox...")
    try:
        link = upload_catbox(ZIP_PATH)
        print("="*70)
        print(f"🔗 LINK PARA DOWNLOAD: {link}")
        print("="*70)
    except Exception as e:
        print(f"❌ Erro no upload: {e}")
        print(">>> Servidor HTTP mantido ativo para download local")
    
    # Manter servidor ativo
    print("\n>>> Servidor mantido ativo...")
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
