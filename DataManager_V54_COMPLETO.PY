# ============================================================
# DataManager_V54_RECONSTRUTOR_FINAL.py
# FOCO: Gravação em Disco Persistente + Treino + Download ZIP
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
import glob
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIGURAÇÃO DE CAMINHO PARA DISCO DA RENDER
# ============================================================
SYMBOL = "PENDLEUSDT"
BASE_DISK_PATH = "/data" if os.path.exists("/data") else "."
OUT_DIR = os.path.join(BASE_DISK_PATH, "pendle_persistence_zone")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH_15M = os.path.join(OUT_DIR, f"{SYMBOL}_15m.csv")
ZIP_FINAL_PATH = os.path.join(OUT_DIR, "PACOTE_TREINO_RENDER.zip")

# ============================================================
# LOGICA DE DOWNLOAD E FEATURES (AS 15 DE OURO)
# ============================================================

def baixar_dados_binance():
    print(f">>> ⏳ Baixando ultimos dados da Binance para teste...")
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','tr','tbb','tbq','i'])
        df = df[['ts','open','high','low','close','volume','tbb']].apply(pd.to_numeric)
        df.rename(columns={'tbb': 'taker_buy_base'}, inplace=True)
        df.to_csv(CSV_PATH_15M, index=False)
        print(f">>> ✅ CSV salvo no disco: {CSV_PATH_15M}")
        return df
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        return None

def feature_engine(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df["vol_realized"] = df["close"].pct_change().rolling(20).std()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["slope20"] = df["close"].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    return df

# ============================================================
# SERVIDOR DE DOWNLOAD
# ============================================================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download':
            if os.path.exists(ZIP_FINAL_PATH):
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="PACOTE_TREINO.zip"')
                self.end_headers()
                with open(ZIP_FINAL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Arquivo ZIP ainda nao gerado. Aguarde o fim do treino.")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
def executar_fluxo():
    # 1. Verifica ou baixa dados
    if not os.path.exists(CSV_PATH_15M):
        df = baixar_dados_binance()
    else:
        print(f">>> ✅ Usando dados existentes no disco persistente.")
        df = pd.read_csv(CSV_PATH_15M)

    if df is not None:
        # 2. Treino
        df = feature_engine(df)
        X_reg = df[['vol_realized', 'slope20', 'close']].fillna(0)
        scaler = StandardScaler()
        scaler.fit(X_reg)
        print(f"📊 Medias do Scaler na Render: {scaler.mean_}")
        
        # 3. Exportação
        joblib.dump(scaler, "scaler_regimes.pkl")
        
        # 4. Criar o ZIP que você vai baixar
        with zipfile.ZipFile(ZIP_FINAL_PATH, 'w') as z:
            z.write(CSV_PATH_15M, arcname=f"{SYMBOL}_15m.csv")
            z.write("scaler_regimes.pkl", arcname="scaler_regimes.pkl")
        
        print(f"✅ PACOTE PRONTO PARA DOWNLOAD EM: /download")

if __name__ == "__main__":
    # Servidor em background
    port = int(os.environ.get("PORT", 10000))
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0', port), DownloadHandler).serve_forever(), daemon=True).start()
    
    executar_fluxo()
    while True: time.sleep(60)