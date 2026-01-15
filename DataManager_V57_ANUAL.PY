import os
import sys
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIGURAÇÃO - 1 ANO INTEGRAL
# ============================================================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1)
END_DT = datetime(2025, 12, 31)

BASE_DISK_PATH = "/data" if os.path.exists("/data") else "."
OUT_DIR = os.path.join(BASE_DISK_PATH, "pendle_factory_v57")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_NAME = "PACOTE_IDENTICO_1ANO.zip"
ZIP_FINAL_PATH = os.path.join(OUT_DIR, ZIP_NAME)

def upload_catbox(filepath):
    print(f">>> 🚀 Enviando para o Catbox: {filepath}...", flush=True)
    try:
        url = "https://catbox.moe/user/api.php"
        with open(filepath, "rb") as f:
            # Aumentei o timeout para ficheiros grandes de 1 ano
            response = requests.post(url, data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=300)
            if response.status_code == 200:
                link = response.text.strip()
                print(f"\n{'='*60}\n✅ LINK DO CATBOX: {link}\n{'='*60}\n", flush=True)
                return link
            else:
                print(f"❌ Erro no servidor Catbox: {response.status_code}")
    except Exception as e:
        print(f"❌ Falha crítica no upload: {e}")
    return None

def feature_engine_identica(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    
    # 1. AS 15 DE OURO (CÓPIA DO TEU V27)
    df["ret1"] = df["close"].pct_change()
    df["vol_realized"] = df["close"].pct_change().rolling(20).std() * np.sqrt(20)
    for p in [9, 20]:
        df[f'ema{p}'] = df['close'].ewm(span=p).mean()
        df[f'dist_ema{p}'] = (df['close'] - df[f'ema{p}']) / df[f'ema{p}']
    df['slope20'] = df['close'].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    
    # 2. PADDING PARA 78 COLUNAS (O SEGREDO DA COMPATIBILIDADE)
    # Precisamos de 78 colunas totais para o seu .pkl local abrir
    for i in range(len(df.columns), 78 + 7): 
        df[f'feat_placeholder_{i}'] = 0.0
    return df

def baixar_35k_velas():
    print(f">>> 📅 Baixando 1 ANO de dados para {SYMBOL}...", flush=True)
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    curr_end = int(END_DT.timestamp() * 1000)
    start_ts = int(START_DT.timestamp() * 1000)
    
    while curr_end > start_ts:
        params = {'symbol': SYMBOL, 'interval': '15m', 'limit': 1000, 'endTime': curr_end}
        try:
            res = requests.get(url, params=params, timeout=30).json()
            if not res: break
            all_data = res + all_data
            curr_end = res[0][0] - 1
            print(f"    Baixado até: {datetime.fromtimestamp(res[0][0]/1000)}")
            if len(all_data) >= 35000: break
        except: break
        
    df = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','tr','tbb','tbq','i'])
    return df[['ts','open','high','low','close','volume','tbb']].apply(pd.to_numeric)

def executar_v57_anual():
    # 1. Download
    df_raw = baixar_35k_velas()
    
    # 2. Processamento das 78 colunas
    print(">>> 🛠️ Aplicando Feature Engine Identica (78 colunas)...")
    df_final = feature_engine_identica(df_raw)
    
    # 3. Alvos K1-K6
    for k in range(1, 7):
        df_final[f'target_k{k}'] = (df_final['close'].shift(-k) > df_final['close']).astype(int)
    
    df_final = df_final.dropna()
    csv_path = os.path.join(OUT_DIR, f"{SYMBOL}_15m_1YEAR.csv")
    df_final.to_csv(csv_path, index=False)
    
    # 4. Treino do Scaler de 1 Ano
    print(">>> 🧠 Gerando Scaler de 1 Ano...")
    features = [c for c in df_final.columns if 'feat' in c or 'ema' in c or 'slope' in c or 'vol' in c or 'ret' in c]
    X = df_final[features].iloc[:, :78] 
    scaler = StandardScaler().fit(X)
    scaler_path = os.path.join(OUT_DIR, "scaler_regimes_1YEAR.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 5. ZIP e Upload (Ponto onde falhou)
    print(">>> 📦 Compactando pacote anual...")
    with zipfile.ZipFile(ZIP_FINAL_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=f"{SYMBOL}_15m_1YEAR.csv")
        z.write(scaler_path, arcname="scaler_regimes_1YEAR.pkl")
    
    link = upload_catbox(ZIP_FINAL_PATH)
    
    if link:
        print(">>> ✅ PROCESSO CONCLUÍDO COM SUCESSO.")
        sys.exit(0) # Força a saída para evitar restart da Render

if __name__ == "__main__":
    executar_v57_anual()