# ============================================================
# DataManager_V53_CEGO.py
# PENDLEUSDT – BUSCA DINAMICA + TREINO V27
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
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

# Forca output unbuffered para o log da Render
sys.stdout.reconfigure(line_buffering=True)

# =========================
# CONFIGURACAO DINAMICA
# =========================
SYMBOL = "PENDLEUSDT"

def localizar_pasta_dados():
    """Busca qualquer pasta que contenha o simbolo, ignorando as datas no nome"""
    # Procura no diretorio atual e no disco montado /data
    padrao = f"*{SYMBOL.lower()}*"
    candidatos = glob.glob(os.path.join("./", padrao)) + glob.glob(os.path.join("/data", padrao))
    
    # Filtra apenas diretorios
    pastas = [p for p in candidatos if os.path.isdir(p)]
    
    if pastas:
        print(f">>> [OK] Pasta localizada: {pastas[0]}")
        return pastas[0]
    return None

# Localizacao automatica
OUT_DIR = localizar_pasta_dados()

if not OUT_DIR:
    # Fallback caso nada seja encontrado
    OUT_DIR = f"./{SYMBOL.lower()}_agg_data"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f">>> [AVISO] Nenhuma pasta encontrada. Criando: {OUT_DIR}")

CSV_PATH_15M = os.path.join(OUT_DIR, f"{SYMBOL}_15m.csv")
ZIP_PKL_PATH = os.path.join(OUT_DIR, "MODELOS_V27_RENDER.zip")

# ============================================================
# ENGINE DE FEATURES (AS 15 DE OURO - REPLICANDO O BUG)
# ============================================================

def realized_vol(close: pd.Series) -> pd.Series:
    return close.pct_change().rolling(window=20).std() * np.sqrt(20)

def feature_engine(df):
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["vol_realized"] = realized_vol(df["close"])
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["slope20"] = df["close"].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["taker_buy_base"]).shift(1)
        df["aggression_delta"] = df["aggression_buy"] - df["aggression_sell"]
    
    return df

def adicionar_features_avancadas(df):
    # O "BUG" DO V27 LOCAL: Ignora as outras 119 features
    return df 

# ============================================================
# SERVIDOR DE DOWNLOAD (ASCII SEGURO)
# ============================================================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download_pkl':
            if os.path.exists(ZIP_PKL_PATH):
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.end_headers()
                with open(ZIP_PKL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"ZIP ainda nao gerado. Verifique o log.")

# ============================================================
# PROCESSO PRINCIPAL
# ============================================================
def iniciar_treino():
    print(f">>> Verificando arquivo: {CSV_PATH_15M}")
    
    if not os.path.exists(CSV_PATH_15M):
        print(f"❌ Erro: Arquivo {SYMBOL}_15m.csv nao encontrado em {OUT_DIR}")
        print(f"📂 Conteudo da pasta: {os.listdir(OUT_DIR) if os.path.exists(OUT_DIR) else 'Pasta inexistente'}")
        return

    print(">>> Carregando dados e iniciando treino...")
    df = pd.read_csv(CSV_PATH_15M)
    df = feature_engine(df)
    df = adicionar_features_avancadas(df)
    
    # 3 Features para o Scaler/Regime (Auditoria)
    # Usando fallback caso falte alguma coluna especifica
    cols_scaler = ['vol_realized', 'slope20', 'close']
    X_reg = df[cols_scaler].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reg)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['regime'] = kmeans.fit_predict(X_scaled)
    
    print(f"📊 Medias do Scaler na Render: {scaler.mean_}")

    # Target K6
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Treino
    features = [c for c in df.columns if c not in ['ts', 'target', 'close_time', 'regime']]
    X = df[features].fillna(0)
    y = df['target']
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3)
    model.fit(X, y)
    
    # Exportar
    joblib.dump(model, "SISTEMA_K6_FINAL.pkl")
    joblib.dump(scaler, "scaler_regimes.pkl")
    joblib.dump(kmeans, "kmeans_regimes.pkl")
    
    with zipfile.ZipFile(ZIP_PKL_PATH, 'w') as z:
        z.write("SISTEMA_K6_FINAL.pkl")
        z.write("scaler_regimes.pkl")
        z.write("kmeans_regimes.pkl")
    
    print(f"✅ SUCESSO! Modelos salvos em: {ZIP_PKL_PATH}")

if __name__ == "__main__":
    # Servidor em thread separada
    port = int(os.environ.get("PORT", 10000))
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0', port), DownloadHandler).serve_forever(), daemon=True).start()
    
    iniciar_treino()
    
    # Mantem o processo vivo para download
    while True:
        time.sleep(60)