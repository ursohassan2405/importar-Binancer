#!/usr/bin/env python3
# ============================================================
# DataManager_V53_FINAL.py
# PENDLEUSDT – Binance Data Vision + TREINO V27
# 
# COMPONENTES:
# 1. Download aggTrades (V51) - SE NECESSÁRIO
# 2. Geração de timeframes (V51) - SE NECESSÁRIO  
# 3. Treino V27 (V52 revisado)
# 4. Gravação permanente com fsync
# 5. Download via HTTP
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =========================
# CONFIGURAÇÃO IMPECÁVEL (IDENTICA AO ANT)
# =========================
SYMBOL = "PENDLEUSDT"

# ⚠️ TESTE: 10 DIAS (comentar depois de testar)
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 1, 10, 23, 59, 59)  # ⭐ 10 DIAS PARA TESTE

# 🔴 PRODUÇÃO: 1 ANO (descomentar para rodar completo)
# START_DT = datetime(2025, 1, 1, 0, 0, 0)
# END_DT = datetime(2025, 12, 31, 23, 59, 59)

# Render persistent disk
# PRIORIDADE: /data > /opt/render/project > .
if os.path.exists("/data"):
    BASE_DISK = "/data"
    print("✅ Usando disco persistente: /data", flush=True)
elif os.path.exists("/opt/render/project"):
    BASE_DISK = "/opt/render/project"
    print("⚠️ Usando /opt/render/project (pode ser apagado!)", flush=True)
else:
    BASE_DISK = "."
    print("⚠️ Usando diretório local (SERÁ APAGADO!)", flush=True)

FOLDER_NAME = f"pendle_agg_{START_DT.strftime('%Y%m%d')}_a_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DISK, FOLDER_NAME)

print(f"📂 OUT_DIR: {OUT_DIR}", flush=True)

CSV_PATH = os.path.join(OUT_DIR, f"{SYMBOL}_aggTrades_full.csv")
ZIP_CSV_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_CSVs.zip")
ZIP_PKL_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_PKLs.zip")

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

# Parâmetros do treino V27
CANDLES_FUTURO = 5

# =========================
# FUNÇÕES AUXILIARES (MATEMÁTICA)
# =========================
def slope_regression(series, window):
    """Calcula slope via regressão linear"""
    # Aceita Series ou array
    if isinstance(series, pd.Series):
        values = series.values
        index = series.index
    else:
        values = series
        index = None
    
    slopes = []
    for i in range(len(values)):
        if i < window - 1:
            slopes.append(np.nan)
        else:
            y = values[i - window + 1:i + 1]
            x = np.arange(window)
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                slopes.append(np.nan)
            else:
                lr = LinearRegression()
                lr.fit(x[valid].reshape(-1, 1), y[valid])
                slopes.append(lr.coef_[0])
    
    if index is not None:
        return pd.Series(slopes, index=index)
    else:
        return np.array(slopes)

def realized_vol(close, window=20):
    """Volatilidade realizada"""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(window)

def yang_zhang(df, window=20):
    """Yang-Zhang volatility"""
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    
    oc = np.log(o / c.shift(1))
    cc = np.log(c / c.shift(1))
    hl = np.log(h / l)
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    rs_var = oc.rolling(window).var()
    cc_var = cc.rolling(window).var()
    hl_var = hl.rolling(window).var()
    
    yz_var = rs_var + k * cc_var + (1 - k) * hl_var
    return np.sqrt(yz_var)

# =========================
# FEATURE ENGINE (16 FEATURES!)
# =========================
def feature_engine(df):
    """Pipeline de features - 16 features base"""
    df = df.copy()
    
    # 1. Básicas
    df["body"] = (df["close"] - df["open"]).shift(1)
    df["range"] = (df["high"] - df["low"]).shift(1)
    
    # 2. Retornos
    df["ret1"] = df["close"].pct_change(1).shift(1)
    df["ret5"] = df["close"].pct_change(5).shift(1)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).shift(1)
    
    # 3. EMAs
    df["ema9"] = df["close"].ewm(span=9).mean().shift(1)
    df["ema20"] = df["close"].ewm(span=20).mean().shift(1)
    df["dist_ema9"] = (df["close"].shift(1) - df["ema9"])
    df["dist_ema20"] = (df["close"].shift(1) - df["ema20"])
    
    # 4. Slopes, Vol, ATR
    df["slope20"] = slope_regression(df["close"], 20)
    df["slope50"] = slope_regression(df["close"], 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)
    
    # ⚠️ ATR14 (OBRIGATÓRIO PARA REGIMES!)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().shift(1)
    
    # 5. Agressão
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["taker_buy_base"]).shift(1)
        df["aggression_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).shift(1)
    else:
        df["aggression_buy"] = 0
        df["aggression_sell"] = 0
        df["aggression_delta"] = 0
    
    return df

def adicionar_features_avancadas(df):
    """DESABILITADO - retorna df sem modificações"""
    return df

# =========================
# DETECÇÃO DE REGIMES
# =========================
def detectar_regimes_mercado_v25(df, n_regimes=4, out_dir=OUT_DIR):
    """Detecta regimes usando 3 features"""
    print(">>> Detectando regimes de mercado...", flush=True)
    
    REGIME_FEATURES = ['vol_realized', 'atr14', 'slope20']
    
    regime_features = [c for c in REGIME_FEATURES if c in df.columns]
    
    if len(regime_features) != 3:
        print(f"    ⚠️ AVISO: Apenas {len(regime_features)}/3 features!", flush=True)
        print(f"    Faltando: {set(REGIME_FEATURES) - set(regime_features)}", flush=True)
    
    X_regime = df[regime_features].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    
    print(f"    📊 MÉDIAS: {scaler.mean_}", flush=True)
    print(f"    📊 DESVIOS: {scaler.scale_}", flush=True)
    
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    print(f"    ✅ {n_regimes} regimes detectados", flush=True)
    print(f"    Distribuição: {df['market_regime'].value_counts().to_dict()}", flush=True)
    
    return df, scaler, kmeans

# =========================
# TREINO V27
# =========================
def treinar_modelo_v27(df_15m, out_dir=OUT_DIR):
    """Treina modelo K6 EXATAMENTE como V27"""
    print("\n" + "="*70, flush=True)
    print("⭐ TREINO V27", flush=True)
    print("="*70, flush=True)
    
    # 0. Normalizar timestamp
    if 'open_time' in df_15m.columns and 'ts' not in df_15m.columns:
        df_15m.rename(columns={'open_time': 'ts'}, inplace=True)
    
    # 1. Target K6
    print(">>> Criando target K6...", flush=True)
    df_15m['target_K6'] = df_15m['close'].shift(-CANDLES_FUTURO) / df_15m['close'] - 1
    df_15m['target_K6_bin'] = (df_15m['target_K6'] > 0).astype(int)
    print(f"    Distribuição: {df_15m['target_K6_bin'].value_counts().to_dict()}", flush=True)
    
    # 2. Features
    print(">>> Calculando features...", flush=True)
    t_start = time.time()
    df_15m = feature_engine(df_15m)
    df_15m = adicionar_features_avancadas(df_15m)
    print(f"    ✅ {len(df_15m.columns)} colunas ({time.time()-t_start:.1f}s)", flush=True)
    
    # 3. Regimes
    t_start = time.time()
    df_15m, scaler, kmeans = detectar_regimes_mercado_v25(df_15m, n_regimes=4, out_dir=out_dir)
    print(f"    ✅ Regimes ({time.time()-t_start:.1f}s)", flush=True)
    
    # 4. Preparar X, y
    print(">>> Preparando matriz...", flush=True)
    
    non_feat = {
        "open", "high", "low", "close", "volume",
        "ts", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "taker_sell_base", "close_time", "ignore", "open_time",
        "mark_price", "index_price", "fundingRate",
        "target_K6", "target_K6_bin",
        "buy_vol", "sell_vol", "delta", "buy_vol_agg", "sell_vol_agg",
        "total_vol_agg", "cum_delta", "price_range", "absorcao", "vpin"
    }
    
    target_cols = {c for c in df_15m.columns if isinstance(c, str) and c.startswith("target_")}
    non_feat = non_feat.union(target_cols)
    
    feat_cols = [c for c in df_15m.columns if c not in non_feat]
    X = df_15m[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(X.columns)
    y = df_15m['target_K6_bin'].values
    
    print(f"    ANTES dropna: X={X.shape}, y={y.shape}", flush=True)
    valid_mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"    DEPOIS dropna: X={X.shape}, y={y.shape}", flush=True)
    print(f"    ✅ Features: {len(feat_cols)}", flush=True)
    
    # 5. Split
    print(">>> Split temporal...", flush=True)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]
    print(f"    Train: {len(X_train)}", flush=True)
    print(f"    Test: {len(X_test)}", flush=True)
    
    # 6. Peso temporal
    print(">>> Calculando peso temporal...", flush=True)
    regime_col = 'market_regime'
    if regime_col not in df_15m.columns:
        print("    ⚠️ Sem regime! Peso uniforme.", flush=True)
        sample_weight_train = None
    else:
        regime_train = df_15m.loc[X_train.index, regime_col].values
        regime_counts = pd.Series(regime_train).value_counts()
        total_samples = len(regime_train)
        
        regime_weights = {}
        for regime_id, count in regime_counts.items():
            regime_weights[regime_id] = total_samples / (len(regime_counts) * count)
        
        sample_weight_train = np.array([regime_weights[r] for r in regime_train])
        print(f"    ✅ Pesos: {regime_weights}", flush=True)
    
    # 7. Treino
    print(">>> Treinando XGBoost...", flush=True)
    t_start = time.time()
    
    modelo = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    if sample_weight_train is not None:
        print("    ⚠️ Peso temporal ATIVADO", flush=True)
        modelo.fit(X_train, y_train, sample_weight=sample_weight_train,
                   eval_set=[(X_test, y_test)], verbose=False)
    else:
        print("    ⚠️ Peso temporal DESATIVADO", flush=True)
        modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    print(f"    ✅ Treinado em {time.time()-t_start:.1f}s", flush=True)
    
    # 8. Avaliar
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Acurácia: {acc:.2%}", flush=True)
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']), flush=True)
    print("="*70, flush=True)
    
    return modelo, scaler, kmeans, feat_cols

# =========================
# DOWNLOAD BINANCE (V51)
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
                print(f"      Retry {attempt}/5 - Aguardando {wait}s...", flush=True)
                time.sleep(wait)
            
            response = session.get(url, headers=get_headers(), timeout=180)  # ⭐ 3min
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    files = z.namelist()
                    if not files:
                        print(f"      ⚠️ ZIP vazio!", flush=True)
                        return None
                    
                    csv_filename = files[0]
                    
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any('transact_time' in str(val) for val in df_test.iloc[0])
                    
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=0 if has_header else None)
                        return df
            
            elif response.status_code == 404:
                print(f"      404 Not Found", flush=True)
                return None
            elif response.status_code in [418, 429]:
                print(f"      {response.status_code} Rate Limit", flush=True)
                continue
            else:
                print(f"      HTTP {response.status_code}", flush=True)
                continue
        except Exception as e:
            print(f"      ❌ Erro: {type(e).__name__}: {str(e)[:100]}", flush=True)
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

def gerar_15m_tratado_incremental(csv_agg_path, csv_15m_path, timeframe_min=15, min_val_usd=500, chunksize=200_000):
    """Gera timeframe a partir de aggTrades (V51)"""
    print(f">>> Gerando {timeframe_min}m...", flush=True)
    
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
                    "buy_vol": 0.0,
                    "sell_vol": 0.0,
                    "total_taker_buy_qty": 0.0,  # ⭐ TOTAL (não só baleias)
                    "total_taker_buy_quote": 0.0,  # ⭐ TOTAL (não só baleias)
                }
                buckets[bms] = st
            else:
                if price > st["high"]:
                    st["high"] = float(price)
                if price < st["low"]:
                    st["low"] = float(price)
                st["close"] = float(price)
            
            st["volume"] += float(qty)
            
            # ⚠️ CRÍTICO: Rastrear TODO taker buy (não só baleias!)
            val_usd = float(price) * float(qty)
            if side == 0:
                st["total_taker_buy_qty"] += float(qty)
                st["total_taker_buy_quote"] += val_usd
            
            # Baleias (separado!)
            if whale:
                if side == 0:
                    st["buy_vol"] += float(qty)
                else:
                    st["sell_vol"] += float(qty)
    
    if not buckets:
        raise RuntimeError("Nenhum bucket gerado!")
    
    rows = []
    for bms in sorted(buckets.keys()):
        st = buckets[bms]
        rows.append([
            st["ts"], st["open"], st["high"], st["low"], st["close"],
            st["volume"], st["buy_vol"], st["sell_vol"],
            st["buy_vol"] - st["sell_vol"],
            st["total_taker_buy_qty"],  # ⭐ TOTAL (não só baleias)
            st["total_taker_buy_quote"],  # ⭐ TOTAL (não só baleias)
        ])
    
    df_15m = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close",
        "volume", "buy_vol", "sell_vol", "delta",
        "total_taker_buy_qty", "total_taker_buy_quote"
    ])
    
    # ⚠️ CRÍTICO: Enriquecimento IGUAL ao ANT
    # taker_buy_base = TOTAL (não só baleias!)
    # buy_vol_agg = BALEIAS (filtrado!)
    df_15m["taker_buy_base"] = df_15m["total_taker_buy_qty"]
    df_15m["taker_buy_quote"] = df_15m["total_taker_buy_quote"]
    df_15m["buy_vol_agg"] = df_15m["buy_vol"]  # Baleias BUY
    df_15m["sell_vol_agg"] = df_15m["sell_vol"]  # Baleias SELL
    df_15m["total_vol_agg"] = df_15m["buy_vol_agg"] + df_15m["sell_vol_agg"]
    
    df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
    df_15m["trades"] = 0
    df_15m["close_time"] = df_15m["ts"] + (timeframe_min * 60 * 1000) - 1
    
    df_15m = df_15m.sort_values("ts").reset_index(drop=True)
    df_15m["cum_delta"] = df_15m["delta"].cumsum()
    df_15m["price_range"] = df_15m["high"] - df_15m["low"]
    df_15m["absorcao"] = df_15m["delta"] / (df_15m["price_range"].replace(0, 1e-9))
    df_15m["vpin"] = (df_15m["buy_vol_agg"] - df_15m["sell_vol_agg"]).abs() / (
        df_15m["total_vol_agg"].replace(0, 1e-9)
    )
    
    # Saneamento
    num_cols = [
        "open","high","low","close","volume",
        "taker_buy_base","taker_buy_quote",
        "buy_vol_agg","sell_vol_agg","total_vol_agg",
        "quote_volume","cum_delta","price_range","absorcao","vpin"
    ]
    for c in num_cols:
        df_15m[c] = pd.to_numeric(df_15m[c], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    
    df_15m["ts"] = pd.to_numeric(df_15m["ts"], errors="coerce").fillna(0).astype("int64")
    df_15m["trades"] = pd.to_numeric(df_15m["trades"], errors="coerce").fillna(0).astype("int64")
    df_15m["close_time"] = pd.to_numeric(df_15m["close_time"], errors="coerce").fillna(0).astype("int64")
    
    # ⚠️ ORDEM EXATA DO ANT (18 COLUNAS!)
    cols_v1 = [
        "ts", "open", "high", "low", "close",
        "volume", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "close_time",
        "cum_delta", "total_vol_agg",
        "buy_vol_agg", "sell_vol_agg",
        "vpin", "price_range", "absorcao"
    ]
    for c in cols_v1:
        if c not in df_15m.columns:
            df_15m[c] = 0.0
    
    df_15m = df_15m[cols_v1]
    df_15m.to_csv(csv_15m_path, index=False)
    
    # ⚠️ GRAVAÇÃO PERMANENTE (fsync)
    try:
        with open(csv_15m_path, 'rb') as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    
    # ⚠️ ESTATÍSTICAS DAS BALEIAS (DEBUG)
    total_candles = len(df_15m)
    candles_com_baleias = (df_15m['buy_vol_agg'] + df_15m['sell_vol_agg'] > 0).sum()
    pct_baleias = (candles_com_baleias / total_candles * 100) if total_candles > 0 else 0
    
    print(f"    ✅ {total_candles} candles → {csv_15m_path}", flush=True)
    print(f"    🐋 Candles com baleias: {candles_com_baleias}/{total_candles} ({pct_baleias:.1f}%)", flush=True)
    
    if pct_baleias < 10:
        print(f"    ⚠️ AVISO: Poucas baleias detectadas! Verificar threshold $500", flush=True)

def baixar_e_gerar_csvs():
    """Download e geração (V51)"""
    print("\n>>> BAIXANDO aggTrades...", flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    print(f"    {total_dates} dias", flush=True)
    
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    
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
                df_processed.to_csv(CSV_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                print(f"✓ {len(df_processed):,} trades ({elapsed:.1f}s)", flush=True)
                del df, df_processed
            else:
                print(f"⚠️ Vazio ({elapsed:.1f}s)", flush=True)
        else:
            print(f"⚠️ Falhou ({elapsed:.1f}s)", flush=True)
        
        # Sleep menor para acelerar
        time.sleep(random.uniform(0.3, 1.0))
    
    session.close()
    print(f"\n    {success_count}/{total_dates} OK", flush=True)
    
    if success_count == 0:
        raise Exception("❌ NENHUM DADO! Download falhou completamente!")
    
    if success_count < total_dates * 0.5:
        print(f"    ⚠️ AVISO: Apenas {success_count}/{total_dates} dias baixados!", flush=True)
        print(f"    Isso pode afetar a qualidade do modelo!", flush=True)
    
    # ⚠️ VALIDAÇÃO: Verificar se aggTrades TEM dados
    df_test = pd.read_csv(CSV_PATH, nrows=100)
    if 'side' not in df_test.columns:
        raise Exception("❌ CSV NÃO É aggTrades! Faltam colunas de trade!")
    
    print(f"    ✅ Validação: CSV é aggTrades (coluna 'side' presente)", flush=True)
    
    # Gerar timeframes
    print("\n>>> Gerando timeframes...", flush=True)
    
    timeframes = {
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "8h": 480,
        "1d": 1440,
    }
    
    for tf_name, tf_min in timeframes.items():
        csv_tf_path = os.path.join(OUT_DIR, f"PENDLEUSDT_{tf_name}.csv")
        if os.path.exists(csv_tf_path):
            os.remove(csv_tf_path)
        gerar_15m_tratado_incremental(CSV_PATH, csv_tf_path, timeframe_min=tf_min, min_val_usd=500, chunksize=200_000)
        
        # ⚠️ CRÍTICO: Verificar gravação
        if not os.path.exists(csv_tf_path):
            raise Exception(f"❌ {csv_tf_path} NÃO gravado!")
        
        # Sync
        try:
            os.sync()
        except:
            pass

# =========================
# CATBOX UPLOAD
# =========================
def upload_catbox(filepath):
    """Upload para CatBox (funciona no Render!)"""
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
# MAIN
# =========================
def main():
    print("\n" + "="*70, flush=True)
    print("🚀 DataManager V53 FINAL", flush=True)
    print("="*70, flush=True)
    print(f"Símbolo: {SYMBOL}", flush=True)
    print(f"Diretório: {OUT_DIR}", flush=True)
    print("="*70, flush=True)
    
    # ============================================================
    # PASSO 1: VERIFICAR CSVs
    # ============================================================
    
    print("\n>>> PASSO 1: Verificando CSVs...", flush=True)
    
    required_tfs = ['15m', '30m', '1h', '4h', '8h', '1d']
    csv_paths = {}
    all_exist = True
    
    for tf in required_tfs:
        csv_path = os.path.join(OUT_DIR, f"{SYMBOL}_{tf}.csv")
        if os.path.exists(csv_path):
            size_mb = os.path.getsize(csv_path) / (1024*1024)
            print(f"    ✅ {tf:3s}: {size_mb:.2f} MB", flush=True)
            csv_paths[tf] = csv_path
        else:
            print(f"    ❌ {tf:3s}: NÃO ENCONTRADO", flush=True)
            all_exist = False
    
    if not all_exist:
        print("\n    📥 BAIXANDO...", flush=True)
        try:
            baixar_e_gerar_csvs()
            
            csv_paths = {}
            for tf in required_tfs:
                csv_path = os.path.join(OUT_DIR, f"{SYMBOL}_{tf}.csv")
                if os.path.exists(csv_path):
                    csv_paths[tf] = csv_path
                else:
                    print(f"    ❌ {tf} não gerado!", flush=True)
                    return
        except Exception as e:
            print(f"    ❌ ERRO: {e}", flush=True)
            return
    
    print(f"\n    ✅ {len(csv_paths)} CSVs OK!", flush=True)
    
    # ============================================================
    # PASSO 2: ZIP CSVs
    # ============================================================
    
    print(f"\n>>> PASSO 2: ZIP CSVs...", flush=True)
    
    with zipfile.ZipFile(ZIP_CSV_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for tf, path in csv_paths.items():
            z.write(path, arcname=os.path.basename(path))
    
    zip_size = os.path.getsize(ZIP_CSV_PATH) / (1024 * 1024)
    print(f"    ✅ {ZIP_CSV_PATH} ({zip_size:.2f} MB)", flush=True)
    
    # ============================================================
    # PASSO 3: ADICIONAR MULTIFRAME
    # ============================================================
    
    print(f"\n>>> PASSO 3: Multiframe...", flush=True)
    
    csv_15m_path = csv_paths['15m']
    df_15m = pd.read_csv(csv_15m_path)
    print(f"    15m: {len(df_15m)} candles", flush=True)
    
    if df_15m['ts'].dtype != 'int64':
        df_15m['ts'] = df_15m['ts'].astype('int64')
    
    for tf in ['30m', '1h', '4h', '8h', '1d']:
        csv_tf_path = csv_paths[tf]
        df_tf = pd.read_csv(csv_tf_path)
        
        print(f"    {tf}: {len(df_tf)} candles", flush=True)
        
        if df_tf['ts'].dtype != 'int64':
            df_tf['ts'] = df_tf['ts'].astype('int64')
        
        df_tf = feature_engine(df_tf)
        
        feature_cols_tf = [c for c in df_tf.columns if c not in [
            'ts', 'open', 'high', 'low', 'close', 'volume',
            'taker_buy_base', 'taker_buy_quote', 'taker_sell_base', 'quote_volume',
            'trades', 'close_time', 'ignore', 'buy_vol', 'sell_vol', 'delta',
            'buy_vol_agg', 'sell_vol_agg', 'total_vol_agg', 'cum_delta',
            'price_range', 'absorcao', 'vpin'
        ]]
        
        rename_map = {col: f'ctx_{tf}_{col}' for col in feature_cols_tf}
        df_tf_ctx = df_tf[['ts'] + feature_cols_tf].rename(columns=rename_map)
        
        df_15m = pd.merge_asof(
            df_15m.sort_values('ts'),
            df_tf_ctx.sort_values('ts'),
            on='ts',
            direction='backward'
        )
        
        print(f"        ✅ {len(feature_cols_tf)} ctx_{tf}_*", flush=True)
    
    ctx_cols = [c for c in df_15m.columns if c.startswith('ctx_')]
    if ctx_cols:
        df_15m[ctx_cols] = df_15m[ctx_cols].fillna(method='ffill')
    
    print(f"\n    ✅ {len(ctx_cols)} colunas contexto", flush=True)
    print(f"    ✅ Shape: {df_15m.shape}", flush=True)
    
    # ============================================================
    # PASSO 4: TREINO
    # ============================================================
    
    print(f"\n>>> PASSO 4: Treino V27...", flush=True)
    
    modelo, scaler, kmeans, feat_cols = treinar_modelo_v27(df_15m, OUT_DIR)
    
    # ============================================================
    # PASSO 5: SALVAR PKLs
    # ============================================================
    
    print(f"\n>>> PASSO 5: Salvando PKLs...", flush=True)
    
    pkl_modelo = os.path.join(OUT_DIR, "SISTEMA_K6_FINAL.pkl")
    pkl_scaler = os.path.join(OUT_DIR, "scaler_regimes.pkl")
    pkl_kmeans = os.path.join(OUT_DIR, "kmeans_regimes.pkl")
    
    joblib.dump(modelo, pkl_modelo)
    joblib.dump(scaler, pkl_scaler)
    joblib.dump(kmeans, pkl_kmeans)
    
    # ⚠️ GRAVAÇÃO PERMANENTE (fsync)
    for pkl_path in [pkl_modelo, pkl_scaler, pkl_kmeans]:
        try:
            with open(pkl_path, 'rb') as f:
                os.fsync(f.fileno())
        except Exception:
            pass
    
    print(f"    ✅ {pkl_modelo}", flush=True)
    print(f"    ✅ {pkl_scaler}", flush=True)
    print(f"    ✅ {pkl_kmeans}", flush=True)
    
    # Validação
    print("\n    🔍 VALIDANDO...", flush=True)
    try:
        modelo_teste = joblib.load(pkl_modelo)
        scaler_teste = joblib.load(pkl_scaler)
        kmeans_teste = joblib.load(pkl_kmeans)
        
        print(f"    ✅ Modelo: {type(modelo_teste).__name__}", flush=True)
        print(f"    ✅ Scaler: {scaler_teste.n_features_in_} features", flush=True)
        print(f"    ✅ KMeans: {kmeans_teste.n_clusters} clusters", flush=True)
    except Exception as e:
        print(f"    🚨 ERRO: {e}", flush=True)
    
    features_path = os.path.join(OUT_DIR, "feature_names.txt")
    with open(features_path, 'w') as f:
        f.write("\n".join(feat_cols))
    print(f"    ✅ {len(feat_cols)} features", flush=True)
    
    # ============================================================
    # PASSO 6: ZIP PKLs
    # ============================================================
    
    print(f"\n>>> PASSO 6: ZIP PKLs...", flush=True)
    
    with zipfile.ZipFile(ZIP_PKL_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pkl_modelo, arcname="SISTEMA_K6_FINAL.pkl")
        z.write(pkl_scaler, arcname="scaler_regimes.pkl")
        z.write(pkl_kmeans, arcname="kmeans_regimes.pkl")
        z.write(features_path, arcname="feature_names.txt")
    
    zip_size = os.path.getsize(ZIP_PKL_PATH) / (1024 * 1024)
    print(f"    ✅ {ZIP_PKL_PATH} ({zip_size:.2f} MB)", flush=True)
    
    # ============================================================
    # PASSO 7: UPLOAD CATBOX
    # ============================================================
    
    print(f"\n>>> PASSO 7: Upload CatBox...", flush=True)
    
    try:
        print("    Enviando CSVs...", flush=True)
        link_csvs = upload_catbox(ZIP_CSV_PATH)
        print(f"    ✅ CSVs: {link_csvs}", flush=True)
    except Exception as e:
        print(f"    ❌ Erro CSVs: {e}", flush=True)
        link_csvs = "ERRO"
    
    try:
        print("    Enviando PKLs...", flush=True)
        link_pkls = upload_catbox(ZIP_PKL_PATH)
        print(f"    ✅ PKLs: {link_pkls}", flush=True)
    except Exception as e:
        print(f"    ❌ Erro PKLs: {e}", flush=True)
        link_pkls = "ERRO"
    
    # ============================================================
    # FINALIZADO
    # ============================================================
    
    print("\n" + "="*70, flush=True)
    print("✅ COMPLETO!", flush=True)
    print("="*70, flush=True)
    print(f"📦 CSVs: {link_csvs}", flush=True)
    print(f"📦 PKLs: {link_pkls}", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
