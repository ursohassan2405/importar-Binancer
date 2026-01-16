#!/usr/bin/env python3
import os, sys, time, zipfile, requests, pandas as pd, numpy as np, joblib, warnings, random, threading
from datetime import datetime, timedelta
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIGURAÇÃO DE TESTE (10 DIAS)
# ============================================================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 1, 10, 23, 59, 59) # AJUSTADO PARA 10 DIAS

BASE_DISK = "/data" if os.path.exists("/data") else "."
FOLDER_NAME = f"pendle_TESTE_{START_DT.strftime('%Y%m%d')}_a_{END_DT.strftime('%Y%m%d')}"
OUT_DIR = os.path.join(BASE_DISK, FOLDER_NAME)
MODELOS_DIR = os.path.join(OUT_DIR, "modelos_v27")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

CSV_15M_FINAL = os.path.join(OUT_DIR, f"{SYMBOL}_15m.csv")
ZIP_CSV_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_CSVs.zip")
ZIP_PKL_PATH = os.path.join(BASE_DISK, f"{FOLDER_NAME}_PKLs.zip")

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
WHALE_THRESHOLD = 500

def upload_catbox(file_path):
    url = "https://catbox.moe/user/api.php"
    if not os.path.exists(file_path): return "INEXISTENTE"
    try:
        with open(file_path, 'rb') as f:
            data = {'reqtype': 'fileupload', 'userhash': ''}
            files = {'fileToUpload': f}
            response = requests.post(url, data=data, files=files, timeout=60)
        return response.text
    except: return "ERRO"

def start_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    print(f"🌍 Servidor HTTP ativo na porta {port}", flush=True)
    server.serve_forever()

def process_aggtrades_to_15m(df_raw):
    df = df_raw.copy()
    df.columns = ['id', 'price', 'qty', 'first_id', 'last_id', 'ts', 'is_buyer_maker']
    df['val_usd'] = df['price'] * df['qty']
    df['ts_bucket'] = (df['ts'] // (15 * 60 * 1000)) * (15 * 60 * 1000)
    df['is_buy'] = df['is_buyer_maker'] == False
    
    grouped = df.groupby('ts_bucket')
    res = pd.DataFrame()
    res['open'] = grouped['price'].first()
    res['high'] = grouped['price'].max()
    res['low'] = grouped['price'].min()
    res['close'] = grouped['price'].last()
    res['volume'] = grouped['qty'].sum()
    res['quote_volume'] = grouped['val_usd'].sum()
    res['trades'] = grouped['id'].count() # CONTAGEM REAL
    
    # taker_buy_base = VOLUME TOTAL (DNA ANT)
    res['taker_buy_base'] = df[df['is_buy']].groupby('ts_bucket')['qty'].sum()
    res['taker_buy_quote'] = df[df['is_buy']].groupby('ts_bucket')['val_usd'].sum()
    
    # buy_vol_agg e sell_vol_agg = APENAS BALEIAS > $500
    df_whales = df[df['val_usd'] >= WHALE_THRESHOLD].copy()
    res['buy_vol_agg'] = df_whales[df_whales['is_buy']].groupby('ts_bucket')['qty'].sum()
    res['sell_vol_agg'] = df_whales[df_whales['is_buy'] == False].groupby('ts_bucket')['qty'].sum()
    
    res = res.fillna(0)
    res['total_vol_agg'] = res['buy_vol_agg'] + res['sell_vol_agg']
    res['delta'] = res['buy_vol_agg'] - res['sell_vol_agg']
    res['cum_delta'] = res['delta'].cumsum()
    res['close_time'] = res.index + (15 * 60 * 1000) - 1
    res['price_range'] = (res['high'] - res['low']) / (res['close'] + 1e-9)
    res['vpin'] = abs(res['buy_vol_agg'] - res['sell_vol_agg']) / (res['volume'] + 1e-9)
    res['absorcao'] = (res['buy_vol_agg'] - res['sell_vol_agg']) / (res['price_range'] + 1e-9)
    
    cols_ant = ["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote","close_time","cum_delta","total_vol_agg","buy_vol_agg","sell_vol_agg","vpin","price_range","absorcao"]
    return res.reset_index().rename(columns={'ts_bucket': 'ts'})[cols_ant]

def generate_multi_tf(df_15m):
    tfs = {'30m':'30min', '1h':'1H', '4h':'4H', '8h':'8H', '1d':'1D'}
    logic = {'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','trades':'sum','taker_buy_base':'sum','taker_buy_quote':'sum','cum_delta':'last','total_vol_agg':'sum','buy_vol_agg':'sum','sell_vol_agg':'sum','vpin':'mean','price_range':'mean','absorcao':'mean'}
    
    for name, freq in tfs.items():
        df_dt = df_15m.copy()
        df_dt['dt'] = pd.to_datetime(df_dt['ts'], unit='ms')
        res = df_dt.set_index('dt').resample(freq).agg(logic).dropna()
        res['ts'] = res.index.view(np.int64) // 10**6
        res.to_csv(os.path.join(OUT_DIR, f"{SYMBOL}_{name}.csv"), index=False)
        print(f"    📊 TF {name} Gerado.")

def train_v27_style(df):
    print("\n>>> PASSO 4: Treinando DNA 78 Colunas...", flush=True)
    df['ret1'] = df['close'].pct_change()
    df['vol_real'] = df['ret1'].rolling(20).std()
    
    # MATRIZ 78 FEATURES
    feat_cols = ["volume","quote_volume","trades","taker_buy_base","taker_buy_quote","cum_delta","total_vol_agg","buy_vol_agg","sell_vol_agg","vpin","price_range","absorcao","ret1","vol_real"]
    while len(feat_cols) < 78:
        feat_cols.append(f"feat_placeholder_{len(feat_cols)}")
        df[feat_cols[-1]] = 0.0
    
    X = df[feat_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=6, random_state=42).fit(X_scaled)
    joblib.dump(scaler, os.path.join(MODELOS_DIR, "scaler_regimes.pkl"))
    joblib.dump(kmeans, os.path.join(MODELOS_DIR, "kmeans_regimes.pkl"))
    
    # TREINO INDIVIDUAL (K1=Preço, K2=Volume)
    y_k1 = np.where(df['close'].shift(-5) > df['close'], 1, 0)
    model_k1 = XGBClassifier().fit(X_scaled, y_k1)
    joblib.dump(model_k1, os.path.join(MODELOS_DIR, "target_K1_XGB.pkl"))
    
    y_k2 = np.where(df['volume'].shift(-5) > df['volume'].rolling(20).mean(), 1, 0)
    model_k2 = XGBClassifier().fit(X_scaled, y_k2)
    joblib.dump(model_k2, os.path.join(MODELOS_DIR, "target_K2_XGB.pkl"))
    print("    ✅ Modelos K1 e K2 Treinados.")

def main():
    threading.Thread(target=start_server, daemon=True).start()
    all_15m = []
    curr = START_DT
    while curr <= END_DT:
        d_str = curr.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/{SYMBOL}/{SYMBOL}-aggTrades-{d_str}.zip"
        r = requests.get(url)
        if r.status_code == 200:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                df_raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
                all_15m.append(process_aggtrades_to_15m(df_raw))
            print(f"    ✅ {d_str} OK")
        curr += timedelta(days=1)
    
    if all_15m:
        df_final = pd.concat(all_15m).sort_values('ts')
        df_final.to_csv(CSV_15M_FINAL, index=False)
        generate_multi_tf(df_final)
        train_v27_style(df_final)
        
        # ZIP e UPLOAD
        with zipfile.ZipFile(ZIP_CSV_PATH, 'w') as z:
            for f in os.listdir(OUT_DIR):
                if f.endswith('.csv'): z.write(os.path.join(OUT_DIR, f), f)
        with zipfile.ZipFile(ZIP_PKL_PATH, 'w') as z:
            for f in os.listdir(MODELOS_DIR): z.write(os.path.join(MODELOS_DIR, f), f)
        
        print(f"\n🚀 TESTE COMPLETO!\nCSVs: {upload_catbox(ZIP_CSV_PATH)}\nPKLs: {upload_catbox(ZIP_PKL_PATH)}")

if __name__ == "__main__":
    main()