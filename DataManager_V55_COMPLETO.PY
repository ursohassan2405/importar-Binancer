import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# Força output para o log da Render
sys.stdout.reconfigure(line_buffering=True)

# CONFIGURAÇÃO DE DISCO E IDENTIDADE
SYMBOL = "PENDLEUSDT"
BASE_DISK_PATH = "/data" if os.path.exists("/data") else "."
OUT_DIR = os.path.join(BASE_DISK_PATH, "pendle_persistence_zone")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_FINAL_PATH = os.path.join(OUT_DIR, "TREINO_COMPLETO_V27.zip")

# --- AS 15 FEATURES (SEM TIRAR NADA) ---
def realized_vol(close, window=20):
    return close.pct_change().rolling(window).std() * np.sqrt(window)

def feature_engine(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df["ret1"] = df["close"].pct_change()
    df["vol_realized"] = realized_vol(df["close"])
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["dist_ema9"] = (df["close"] - df["ema9"]) / df["ema9"]
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["slope20"] = df["close"].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    if 'tbb' in df.columns:
        df["aggression_buy"] = df["tbb"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["tbb"]).shift(1)
        df["aggression_delta"] = df["aggression_buy"] - df["aggression_sell"]
    return df

def upload_catbox(filepath):
    print(f">>> 🚀 Enviando para o Catbox...", flush=True)
    try:
        url = "https://catbox.moe/user/api.php"
        with open(filepath, "rb") as f:
            response = requests.post(url, data={"reqtype": "fileupload"}, files={"fileToUpload": f})
            print(f"\n✅ LINK DO CATBOX: {response.text.strip()}\n")
    except Exception as e: print(f"❌ Erro upload: {e}")

# --- O TREINO "TUDO, TUDO" (XGB, LGBM, LINEAR) ---
def executar_treino_completo():
    # 1. Captura Dados (Igual ao que fizemos no V54)
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','tr','tbb','tbq','i'])
    df = df[['ts','open','high','low','close','volume','tbb']].apply(pd.to_numeric)
    
    # 2. Processa Features e Target
    df = feature_engine(df)
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int) # Candles=5
    df = df.dropna()
    
    features = ["ret1", "vol_realized", "dist_ema9", "dist_ema20", "slope20"]
    X = df[features]
    y = df['target']
    
    # 3. Treina os 3 Modelos do seu V52
    print(">>> Treinando Modelos V27 (XGB, LGBM, Linear)...")
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    m_xgb = xgb.XGBClassifier().fit(X_s, y)
    m_lgb = lgb.LGBMClassifier(verbosity=-1).fit(X_s, y)
    m_lin = LinearRegression().fit(X_s, y)
    
    # 4. Salva e Zipa TUDO
    joblib.dump(m_xgb, "XGB_MODEL.pkl")
    joblib.dump(m_lgb, "LGBM_MODEL.pkl")
    joblib.dump(m_lin, "LINEAR_MODEL.pkl")
    joblib.dump(scaler, "scaler_regimes.pkl")
    
    with zipfile.ZipFile(ZIP_FINAL_PATH, 'w') as z:
        for f in ["XGB_MODEL.pkl", "LGBM_MODEL.pkl", "LINEAR_MODEL.pkl", "scaler_regimes.pkl"]:
            z.write(f)
        # Adiciona o CSV para você conferir local
        df.to_csv("dados_treino.csv", index=False)
        z.write("dados_treino.csv")
    
    upload_catbox(ZIP_FINAL_PATH)

if __name__ == "__main__":
    executar_treino_completo()