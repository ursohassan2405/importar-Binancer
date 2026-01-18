#!/usr/bin/env python3
"""
DataManager_V55_ULTRA.py
========================
ALTA PERFORMANCE + 186 FEATURES + DISCO RENDER CORRIGIDO

OTIMIZAÇÕES:
- Zero prints em loops (apenas marcos)
- Vetorização máxima
- gc.collect() estratégico
- Dtype otimizado (float32)
- Disco: verificação robusta + fsync
"""

import os, sys, time, gc, zipfile, random, warnings
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
SYMBOL = os.environ.get("SYMBOL", "PENDLEUSDT")

# =============================================================================
# CONFIGURAÇÃO DE DATAS - FIXAS (IGUAL V51)
# =============================================================================
# CRÍTICO: Usar EXATAMENTE o mesmo período do V51 para comparação válida
# V51 usa: 01/01/2025 a 30/12/2025
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 12, 30, 23, 59, 59)
DAYS = (END_DT.date() - START_DT.date()).days + 1  # 364 dias

print(f"📅 Período: {START_DT.strftime('%Y-%m-%d')} → {END_DT.strftime('%Y-%m-%d')} ({DAYS} dias)", flush=True)

# =============================================================================
# DISCO RENDER - VERIFICAÇÃO ROBUSTA
# =============================================================================
def setup_disk():
    """Encontra disco gravável com verificação real."""
    candidates = ["/data", "/opt/render/project/data", "/opt/render/project", "."]
    
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            test_path = os.path.join(path, f".test_{os.getpid()}")
            
            # Teste de escrita
            with open(test_path, "wb") as f:
                f.write(b"test_data_12345")
                f.flush()
                os.fsync(f.fileno())
            
            # Teste de leitura
            with open(test_path, "rb") as f:
                if f.read() == b"test_data_12345":
                    os.remove(test_path)
                    return path
            os.remove(test_path)
        except:
            pass
    return "."

BASE_DISK = setup_disk()
FOLDER = f"{SYMBOL}_{START_DT.strftime('%Y%m%d')}_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DISK, FOLDER)
MODELOS_DIR = os.path.join(OUT_DIR, "modelos_salvos")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

CSV_AGG = os.path.join(OUT_DIR, f"{SYMBOL}_agg.csv")

print(f"🚀 V55 ULTRA | {SYMBOL} | {DAYS}d | {BASE_DISK}", flush=True)

# =============================================================================
# FUNÇÕES DE DISCO SEGURAS
# =============================================================================
def save_csv(df, path):
    """Salva CSV com fsync e ALTA PRECISÃO (igual V51)."""
    # float_format='%.10g' garante precisão igual ao V51
    # Isso evita erros de arredondamento no cum_delta e outras colunas derivadas
    df.to_csv(path, index=False, float_format='%.10g')
    try:
        with open(path, 'r') as f:
            os.fsync(f.fileno())
    except:
        pass

def save_pkl(obj, path):
    """Salva PKL com fsync."""
    joblib.dump(obj, path)
    try:
        with open(path, 'rb') as f:
            os.fsync(f.fileno())
    except:
        pass

def load_pkl(path):
    """Carrega PKL."""
    return joblib.load(path)

# =============================================================================
# DOWNLOAD OTIMIZADO
# =============================================================================
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

def download_day(symbol, date, session):
    """Download 1 dia."""
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date.strftime('%Y-%m-%d')}.zip"
    try:
        r = session.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    return pd.read_csv(f, header=None, usecols=[0,1,2,5],
                                      names=['ts','price','qty','side'],
                                      dtype={'ts':'int64','price':'float32','qty':'float32','side':'int8'})
    except:
        pass
    return None

def baixar_dados():
    """Download todos os dias - progress mínimo."""
    print(f"📥 Download {DAYS} dias...", flush=True)
    
    session = requests.Session()
    dates = [START_DT + timedelta(days=i) for i in range(DAYS)]
    first = True
    ok = 0
    t0 = time.time()
    
    for i, dt in enumerate(dates):
        df = download_day(SYMBOL, dt, session)
        if df is not None and len(df) > 0:
            df.to_csv(CSV_AGG, mode='a', header=first, index=False, float_format='%.10g')
            first = False
            ok += 1
            del df
        
        # Progress a cada 10% ou 30 dias
        if (i+1) % max(1, DAYS//10) == 0 or (i+1) % 30 == 0:
            pct = 100 * (i+1) // DAYS
            print(f"   📥 {i+1}/{DAYS} ({pct}%) - {ok} OK", flush=True)
        
        if (i+1) % 50 == 0:
            gc.collect()
        time.sleep(random.uniform(0.1, 0.3))
    
    session.close()
    print(f"✅ Download completo: {ok}/{DAYS} ({time.time()-t0:.0f}s)", flush=True)
    return ok

# =============================================================================
# GERAÇÃO TIMEFRAME - ULTRA OTIMIZADO PARA 512MB RAM
# =============================================================================
def gerar_tf(tf_min):
    """Gera timeframe - otimizado para baixa memória."""
    
    buckets = {}
    CHUNK = 50_000  # Chunks PEQUENOS para 512MB RAM
    
    for chunk in pd.read_csv(CSV_AGG, chunksize=CHUNK,
                            dtype={'ts':'int64','price':'float32','qty':'float32','side':'int8'}):
        
        # Processar chunk
        for _, row in chunk.iterrows():
            ts = row['ts']
            p = row['price']
            q = row['qty']
            s = row['side']
            
            b = (ts // (tf_min * 60000)) * (tf_min * 60000)
            whale = (p * q) >= 500
            
            if b not in buckets:
                buckets[b] = [p, p, p, p, 0.0, 0.0, 0.0]
            else:
                r = buckets[b]
                if p > r[1]: r[1] = p
                if p < r[2]: r[2] = p
                r[3] = p
            
            buckets[b][4] += q
            if whale:
                if s == 0:
                    buckets[b][5] += q
                else:
                    buckets[b][6] += q
        
        del chunk
        gc.collect()
    
    gc.collect()
    
    # Montar DataFrame
    ts_list = sorted(buckets.keys())
    data = np.zeros((len(ts_list), 9), dtype=np.float64)
    
    for i, ts in enumerate(ts_list):
        r = buckets[ts]
        data[i] = [ts, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[5]-r[6]]
    
    del buckets
    gc.collect()
    
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','buy_vol_agg','sell_vol_agg','delta'])
    
    # Aliases (igual V51)
    df['buy_vol'] = df['buy_vol_agg']
    df['sell_vol'] = df['sell_vol_agg']
    df['total_vol_agg'] = df['buy_vol_agg'] + df['sell_vol_agg']
    
    df['taker_buy_base'] = df['buy_vol_agg']
    df['taker_sell_base'] = df['sell_vol_agg']
    df['taker_buy_quote'] = df['taker_buy_base'] * df['close']
    
    # Campos V1 comuns
    df['quote_volume'] = df['volume'] * df['close']
    df['trades'] = 0
    df['close_time'] = df['ts'] + (tf_min * 60000) - 1
    
    # Métricas adicionais (igual V51)
    df = df.sort_values('ts').reset_index(drop=True)
    df['cum_delta'] = df['delta'].cumsum()
    df['price_range'] = df['high'] - df['low']
    df['absorcao'] = df['delta'] / (df['price_range'].replace(0, 1e-9))
    df['vpin'] = (df['buy_vol_agg'] - df['sell_vol_agg']).abs() / (df['total_vol_agg'].replace(0, 1e-9))
    
    # Saneamento (IGUAL V51)
    num_cols = [
        "open","high","low","close",
        "volume","buy_vol","sell_vol","delta",
        "buy_vol_agg","sell_vol_agg","total_vol_agg",
        "taker_buy_base","taker_sell_base","taker_buy_quote",
        "quote_volume","cum_delta","price_range","absorcao","vpin"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0).astype("int64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").fillna(0).astype("int64")
    
    # Ordem V1 (IGUAL V51)
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
        if c not in df.columns:
            df[c] = 0.0
    
    df = df[cols_v1]
    
    return df

def gerar_todos_tfs():
    """Gera todos TFs."""
    print("📊 Timeframes:", end=" ", flush=True)
    
    tfs = {'15m':15, '30m':30, '1h':60, '4h':240, '8h':480, '1d':1440}
    paths = {}
    
    for nome, mins in tfs.items():
        t0 = time.time()
        df = gerar_tf(mins)
        path = os.path.join(OUT_DIR, f"{SYMBOL}_{nome}.csv")
        save_csv(df, path)
        paths[nome] = path
        print(f"{nome}({len(df)})", end=" ", flush=True)
        del df
        gc.collect()
    
    print("✅", flush=True)
    return paths

# =============================================================================
# SLOPE (VETORIZADO)
# =============================================================================
def calc_slope(series, w):
    """Slope vetorizado."""
    result = np.zeros(len(series))
    values = series.values
    
    for i in range(w, len(values)):
        y = values[i-w:i]
        if not np.isnan(y).any():
            x_mean = (w-1)/2
            y_mean = y.mean()
            num = np.sum((np.arange(w) - x_mean) * (y - y_mean))
            den = np.sum((np.arange(w) - x_mean)**2)
            result[i] = num / (den + 1e-9)
    
    return result

# =============================================================================
# FEATURE ENGINE - 186 FEATURES (OTIMIZADO 512MB)
# =============================================================================
def feature_engine(df):
    """Gera TODAS as 186 features - otimizado para baixa memória."""
    # NÃO fazer copy - trabalhar in-place
    
    # === PRICE ACTION ===
    df['range'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    df['range_pct'] = df['range'] / (df['close'] + 1e-9)
    gc.collect()
    
    # === RETORNOS ===
    for p in [1,2,3,5,10,20]:
        df[f'ret{p}'] = df['close'].pct_change(p)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret20_norm'] = df['ret20'] / (df['close'].rolling(20).std() + 1e-9)
    gc.collect()
    
    # === EMAs ===
    for s in [9,20,50,100,200]:
        df[f'ema{s}'] = df['close'].ewm(span=s, adjust=False).mean()
        df[f'dist_ema{s}'] = df['close'] - df[f'ema{s}']
    gc.collect()
    
    # === SLOPES ===
    for w in [20,50,100,200]:
        df[f'slope{w}'] = calc_slope(df['close'], w)
    gc.collect()
    
    # === ATR / VOLATILIDADE ===
    tr = pd.concat([df['high']-df['low'], 
                    (df['high']-df['close'].shift(1)).abs(),
                    (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['tr'] = tr
    df['atr14'] = tr.rolling(14).mean()
    df['atr_to_close'] = df['atr14'] / (df['close'] + 1e-9)
    df['range_to_atr'] = df['range'] / (df['atr14'] + 1e-9)
    df['vol_realized'] = df['ret1'].rolling(20).std() * np.sqrt(252*96)
    df['vol_yz'] = df['ret1'].rolling(20).std()
    
    # === REGIMES ===
    df['vol_regime'] = df['atr14'].rolling(50).mean()
    df['atr_compression'] = df['atr14'] / (df['vol_regime'] + 1e-9)
    df['trend_regime'] = calc_slope(df['close'], 100)
    df['liquidity_regime'] = df['volume'].rolling(50).mean()
    
    # === MOMENTUM ===
    df['momentum_1'] = df['ret1']
    df['momentum_2'] = df['ret2']
    df['momentum_acceleration'] = df['momentum_1'] - df['momentum_2']
    df['momentum_long'] = df['ret3'].fillna(0) + df['ret10'].fillna(0) + df['ret20'].fillna(0)
    df['volatility_acceleration'] = df['range_pct'].diff()
    
    # === WICK/STRUCTURE ===
    df['body_to_range'] = df['body'] / (df['range'] + 1e-9)
    df['wick_ratio_up'] = df['upper_wick'] / (df['range'] + 1e-9)
    df['wick_ratio_down'] = df['lower_wick'] / (df['range'] + 1e-9)
    
    # === VOLUME ===
    df['volume_diff'] = df['volume'].diff()
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-9)
    
    # === AGRESSÃO ===
    df['aggression_delta'] = df['delta']
    df['aggression_buy'] = df['buy_vol_agg']
    df['aggression_sell'] = df['sell_vol_agg']
    df['aggression_imbalance'] = df['delta'] / (df['volume'] + 1e-9)
    df['aggression_pressure'] = df['delta'] * df['ret1']
    df['delta_acc'] = df['delta'].diff()
    df['buy_ratio'] = df['buy_vol_agg'] / (df['volume'] + 1e-9)
    df['sell_ratio'] = df['sell_vol_agg'] / (df['volume'] + 1e-9)
    df['aggr_cumsum_20'] = df['delta'].rolling(20).sum()
    df['flow_dominance'] = df['delta'] * (1 + df['vpin'].fillna(0))
    df['flow_acceleration'] = df['delta'].diff()
    
    # === Z-SCORES ===
    df['price_z'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-9)
    df['body_z'] = (df['body'] - df['body'].rolling(20).mean()) / (df['body'].rolling(20).std() + 1e-9)
    df['range_z'] = (df['range'] - df['range'].rolling(20).mean()) / (df['range'].rolling(20).std() + 1e-9)
    
    # === SQUEEZE ===
    df['vol_squeeze'] = df['close'].rolling(20).std() / (df['close'].rolling(100).std() + 1e-9)
    df['range_squeeze'] = df['range_pct'].rolling(14).std() / (df['range_pct'].rolling(50).std() + 1e-9)
    
    # === DISTÂNCIAS NORM ===
    df['dist_ema20_norm'] = df['dist_ema20'] / (df['atr14'] + 1e-9)
    df['dist_ema50_norm'] = df['dist_ema50'] / (df['atr14'] + 1e-9)
    df['dist_ema100_norm'] = df['dist_ema100'] / (df['atr14'] + 1e-9)
    
    # === VWAP ===
    cum_vol = df['volume'].cumsum()
    cum_pv = (df['volume'] * df['close']).cumsum()
    df['vwap'] = cum_pv / (cum_vol + 1e-9)
    df['week'] = df['ts'] // (7*24*60*60*1000)
    df['month'] = df['ts'] // (30*24*60*60*1000)
    
    # VWAP por período (simplificado)
    df['vwap_week'] = df.groupby('week')['close'].transform('mean')
    df['vwap_month'] = df.groupby('month')['close'].transform('mean')
    df['anchored_vwap'] = df['vwap']
    
    df['vwap_std'] = (df['close'] - df['vwap']).rolling(20).std()
    df['vwap_upper'] = df['vwap'] + 2 * df['vwap_std']
    df['vwap_lower'] = df['vwap'] - 2 * df['vwap_std']
    df['vwap_zscore'] = (df['close'] - df['vwap']) / (df['vwap_std'] + 1e-9)
    df['dist_vwap_atr'] = (df['close'] - df['vwap']) / (df['atr14'] + 1e-9)
    
    # === BOLLINGER / KELTNER ===
    bb_ma = df['close'].rolling(5).mean()
    bb_std = df['close'].rolling(5).std()
    df['bb_upper_5'] = bb_ma + 2*bb_std
    df['bb_lower_5'] = bb_ma - 2*bb_std
    df['bb_width_5'] = (df['bb_upper_5'] - df['bb_lower_5']) / (bb_ma + 1e-9)
    
    kc_atr = df['tr'].rolling(5).mean()
    df['kc_upper_5'] = bb_ma + 1.5*kc_atr
    df['kc_lower_5'] = bb_ma - 1.5*kc_atr
    df['kc_range_5'] = df['kc_upper_5'] - df['kc_lower_5']
    
    df['micro_squeeze'] = ((df['bb_upper_5'] < df['kc_upper_5']) & (df['bb_lower_5'] > df['kc_lower_5'])).astype(int)
    df['micro_vol_squeeze'] = (df['vol_squeeze'] < 0.8).astype(int)
    df['pre_breakout_pressure'] = df['micro_squeeze'] * df['delta'].abs()
    
    # === INSIDE BAR / NR ===
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['inside_bar_strength'] = df['inside_bar'] * (1 - df['range']/(df['range'].shift(1)+1e-9))
    df['nr4'] = (df['range'] == df['range'].rolling(4).min()).astype(int)
    df['nr7'] = (df['range'] == df['range'].rolling(7).min()).astype(int)
    df['close_pos'] = (df['close'] - df['low']) / (df['range'] + 1e-9)
    df['breakout_bias'] = df['close_pos'] * 2 - 1
    
    # === FRACTALS ===
    df['fractal_high'] = ((df['high'] > df['high'].shift(2)) & (df['high'] > df['high'].shift(1)) &
                          (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))).astype(int)
    pivot_h = df['high'].where(df['fractal_high']==1).ffill()
    pivot_l = df['low'].where(df['fractal_high']==1).ffill()
    df['dist_to_pivot_high'] = (df['close'] - pivot_h) / (df['atr14'] + 1e-9)
    df['dist_to_pivot_low'] = (df['close'] - pivot_l) / (df['atr14'] + 1e-9)
    
    # === WICK RATIOS ===
    df['wick_up_ratio_5'] = df['wick_ratio_up'].rolling(5).mean()
    df['wick_down_ratio_5'] = df['wick_ratio_down'].rolling(5).mean()
    df['reversal_prob'] = (df['wick_up_ratio_5'] + df['wick_down_ratio_5']) / 2
    
    # === PLACEHOLDERS ===
    df['market_regime'] = 0
    df['threshold_A_usado'] = 0.01
    df['regime_targetA'] = 0
    
    # Limpar
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

# =============================================================================
# CONTEXTO MULTI-TF
# =============================================================================
def add_multi_tf(df_base, paths):
    """Adiciona 70 features de contexto (14 x 5 TFs)."""
    print("📈 Multi-TF:", end=" ", flush=True)
    
    ctx_cols = ['volume','quote_volume','trades','taker_buy_base','taker_sell_base',
                'taker_buy_quote','buy_vol_agg','sell_vol_agg','total_vol_agg',
                'delta','cum_delta','vpin','price_range','absorcao']
    
    df = df_base.sort_values('ts').reset_index(drop=True)
    
    for tf in ['30m','1h','4h','8h','1d']:
        if tf not in paths:
            continue
        
        tf_df = pd.read_csv(paths[tf])
        
        # Renomear e shift
        rename_map = {c: f'ctx_{tf}_{c}' for c in ctx_cols if c in tf_df.columns}
        tf_df = tf_df.rename(columns=rename_map)
        
        cols_ctx = [c for c in tf_df.columns if c.startswith(f'ctx_{tf}')]
        for c in cols_ctx:
            tf_df[c] = tf_df[c].shift(1)
        
        # Merge asof
        df = pd.merge_asof(df, tf_df[['ts'] + cols_ctx].sort_values('ts'), on='ts', direction='backward')
        
        del tf_df
        print(f"{tf}", end=" ", flush=True)
    
    gc.collect()
    print("✅", flush=True)
    
    return df.fillna(0)

# =============================================================================
# REGIMES
# =============================================================================
def detectar_regimes(df):
    """Detecta regimes e salva scaler/kmeans."""
    feats = [c for c in ['vol_realized','atr14','slope20'] if c in df.columns]
    if not feats:
        feats = ['ret1']
    
    X = df[feats].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    # Salvar
    save_pkl(scaler, os.path.join(MODELOS_DIR, 'scaler_regimes.pkl'))
    save_pkl(kmeans, os.path.join(MODELOS_DIR, 'kmeans_regimes.pkl'))
    
    return df, scaler, kmeans

# =============================================================================
# TREINO K1-K6
# =============================================================================
def treinar_k(df, k, X_cols):
    """Treina modelo para target K."""
    
    # Criar target
    y = (df['close'].shift(-k) > df['close']).astype(int).values
    X = df[X_cols].values
    
    # Remover últimos k (sem target)
    X, y = X[:-k], y[:-k]
    
    # Split temporal
    n = len(X)
    i_train = int(n * 0.7)
    i_val = int(n * 0.8)
    
    X_train, y_train = X[:i_train], y[:i_train]
    X_test, y_test = X[i_val:], y[i_val:]
    
    # Treinar 3 modelos
    results = []
    
    # LGBM
    try:
        m = LGBMClassifier(n_estimators=300, learning_rate=0.03, n_jobs=-1, verbose=-1)
        m.fit(X_train, y_train)
        f1 = f1_score(y_test, m.predict(X_test), average='macro')
        results.append(('LGBM', f1, m))
    except:
        pass
    
    # XGB
    try:
        m = XGBClassifier(n_estimators=300, learning_rate=0.03, tree_method='hist', verbosity=0)
        m.fit(X_train, y_train)
        f1 = f1_score(y_test, m.predict(X_test), average='macro')
        results.append(('XGB', f1, m))
    except:
        pass
    
    # CAT
    try:
        m = CatBoostClassifier(iterations=300, learning_rate=0.03, verbose=False)
        m.fit(X_train, y_train)
        f1 = f1_score(y_test, m.predict(X_test), average='macro')
        results.append(('CAT', f1, m))
    except:
        pass
    
    if not results:
        return None, None, 0
    
    # Melhor
    results.sort(key=lambda x: x[1], reverse=True)
    nome, f1, modelo = results[0]
    
    return nome, modelo, f1

def treinar_todos_k(df):
    """Treina K1-K6."""
    print("🎯 Treino K1-K6:", end=" ", flush=True)
    
    # Colunas para treino
    excluir = {'ts','close_time','week','month','target_A','target_B','target_C','target_A_bin'}
    X_cols = [c for c in df.columns if c not in excluir and df[c].dtype in ['float64','float32','int64','int32']]
    
    ks = {1:3, 2:5, 3:7, 4:10, 5:15, 6:20}  # K: horizonte
    
    for k_num, horizonte in ks.items():
        nome, modelo, f1 = treinar_k(df, horizonte, X_cols)
        if modelo:
            path = os.path.join(MODELOS_DIR, f'target_K{k_num}_{nome}.pkl')
            save_pkl(modelo, path)
            print(f"K{k_num}:{nome}({f1:.3f})", end=" ", flush=True)
    
    print("✅", flush=True)

# =============================================================================
# SERVIDOR HTTP
# =============================================================================
def start_server():
    """Servidor HTTP para Render health check."""
    port = int(os.environ.get("PORT", 10000))
    
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'V55 ULTRA OK')
        def log_message(self, *args):
            pass
    
    server = HTTPServer(('0.0.0.0', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"🌐 HTTP :{port}", flush=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    
    # Servidor
    start_server()
    
    # 1. Download - SEMPRE baixar de novo para garantir dados corretos
    # Verificar se cache existe E tem tamanho razoável (>100MB para 365 dias)
    min_size = 100 * 1024 * 1024  # 100MB mínimo esperado
    
    if os.path.exists(CSV_AGG):
        cache_size = os.path.getsize(CSV_AGG)
        if cache_size < min_size:
            print(f"⚠️ Cache pequeno ({cache_size//1024//1024}MB < 100MB), re-baixando...", flush=True)
            os.remove(CSV_AGG)
            baixar_dados()
        else:
            print(f"📁 Cache OK: {CSV_AGG} ({cache_size//1024//1024}MB)", flush=True)
    else:
        baixar_dados()
    
    # 2. Timeframes
    paths = gerar_todos_tfs()
    
    # 3. Features (15m)
    print("⚙️ Features...", end=" ", flush=True)
    df = pd.read_csv(paths['15m'])
    df = feature_engine(df)
    print(f"{len(df.columns)} cols", flush=True)
    
    # 4. Multi-TF
    df = add_multi_tf(df, paths)
    print(f"📊 Total: {len(df.columns)} features", flush=True)
    
    # 5. Regimes
    df, _, _ = detectar_regimes(df)
    
    # 6. Treino
    treinar_todos_k(df)
    
    # 7. Done
    elapsed = time.time() - t_total
    print(f"\n🏁 COMPLETO em {elapsed//60:.0f}m{elapsed%60:.0f}s", flush=True)
    
    # Listar
    print(f"📂 {MODELOS_DIR}:", flush=True)
    for f in sorted(os.listdir(MODELOS_DIR)):
        size = os.path.getsize(os.path.join(MODELOS_DIR, f))
        print(f"   {f} ({size//1024}KB)", flush=True)
    
    # Keep alive
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
