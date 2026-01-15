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
import xgboost as xgb
import lightgbm as lgb

# Força output para o log da Render
sys.stdout.reconfigure(line_buffering=True)

# CONFIGURAÇÃO
SYMBOL = "PENDLEUSDT"
BASE_DISK_PATH = "/data" if os.path.exists("/data") else "."
OUT_DIR = os.path.join(BASE_DISK_PATH, "pendle_persistence_zone")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_FINAL_PATH = os.path.join(OUT_DIR, "PACOTE_COMPLETO_V56.zip")

def baixar_30_dias_15m():
    print(">>> ⏳ Baixando 30 dias de dados (aprox. 2880 velas)...", flush=True)
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=15m&limit=1000"
    all_data = []
    end_time = None
    
    for i in range(3): # 3 chamadas de 1000 = 3000 velas
        target_url = url + (f"&endTime={end_time}" if end_time else "")
        res = requests.get(target_url).json()
        if not res: break
        all_data = res + all_data
        end_time = res[0][0] - 1
        print(f"    Bloco {i+1} baixado...")
    
    df = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','tr','tbb','tbq','i'])
    df = df[['ts','open','high','low','close','volume','tbb']].apply(pd.to_numeric)
    return df.drop_duplicates(subset='ts').sort_values('ts').tail(2880)

def resample_data(df, interval):
    df_res = df.copy()
    df_res['ts'] = pd.to_datetime(df_res['ts'], unit='ms')
    df_res.set_index('ts', inplace=True)
    logic = {'open':'first','high':'max','low':'min','close':'last','volume':'sum','tbb':'sum'}
    res = df_res.resample(interval).apply(logic).dropna()
    res.reset_index(inplace=True)
    res['ts'] = res['ts'].astype(np.int64) // 10**6
    return res

def feature_engine(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df["ret1"] = df["close"].pct_change()
    df["vol_realized"] = df["close"].pct_change().rolling(20).std() * np.sqrt(20)
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["dist_ema9"] = (df["close"] - df["ema9"]) / df["ema9"]
    df["dist_ema20"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["slope20"] = df["close"].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    return df

def upload_catbox(filepath):
    print(f">>> 🚀 Enviando para o Catbox...", flush=True)
    try:
        url = "https://catbox.moe/user/api.php"
        with open(filepath, "rb") as f:
            response = requests.post(url, data={"reqtype": "fileupload"}, files={"fileToUpload": f})
            print(f"\n{'='*60}\n✅ LINK DO CATBOX: {response.text.strip()}\n{'='*60}\n")
    except Exception as e: print(f"❌ Erro upload: {e}")

def executar_v56():
    # 1. Auditoria inicial do Disco
    print(f"--- 📂 AUDITORIA DE DISCO ---")
    print(f"Diretorio de saida: {OUT_DIR}")
    if os.path.exists(BASE_DISK_PATH):
        print(f"Conteudo de {BASE_DISK_PATH}: {os.listdir(BASE_DISK_PATH)}")
    
    # 2. Dados
    df_15m = baixar_30_dias_15m()
    
    tfs = {
        "15m": df_15m,
        "30m": resample_data(df_15m, '30T'),
        "1h":  resample_data(df_15m, '1H'),
        "4h":  resample_data(df_15m, '4H'),
        "8h":  resample_data(df_15m, '8H'),
        "1d":  resample_data(df_15m, '1D')
    }

    # 3. Treino
    print(">>> 🧠 Treinando Modelos V27...")
    df_train = feature_engine(df_15m)
    df_train['target'] = (df_train['close'].shift(-5) > df_train['close']).astype(int)
    df_train = df_train.dropna()
    
    feats = ["ret1", "vol_realized", "dist_ema9", "dist_ema20", "slope20"]
    X = df_train[feats]
    y = df_train['target']
    
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    m_xgb = xgb.XGBClassifier().fit(X_s, y)
    m_lgb = lgb.LGBMClassifier(verbosity=-1).fit(X_s, y)
    m_lin = LinearRegression().fit(X_s, y)

    # 4. Empacotamento
    print(">>> 📦 Criando ZIP final...")
    with zipfile.ZipFile(ZIP_FINAL_PATH, 'w') as z:
        for name, d in tfs.items():
            fname = f"{SYMBOL}_{name}.csv"
            fpath = os.path.join(OUT_DIR, fname)
            d.to_csv(fpath, index=False)
            z.write(fpath, arcname=fname)
            
        for m, n in [(m_xgb,"XGB_MODEL.pkl"),(m_lgb,"LGBM_MODEL.pkl"),(m_lin,"LINEAR_MODEL.pkl"),(scaler,"scaler_regimes.pkl")]:
            joblib.dump(m, n)
            z.write(n)
            
    upload_catbox(ZIP_FINAL_PATH)

if __name__ == "__main__":
    executar_v56()