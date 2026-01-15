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
# CONFIGURAÇÃO MESTRE - 1 ANO (2025 COMPLETO)
# ============================================================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1)
END_DT = datetime(2025, 12, 31)

BASE_DISK_PATH = "/data" if os.path.exists("/data") else "."
OUT_DIR = os.path.join(BASE_DISK_PATH, "pendle_factory_v58")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_FINAL_PATH = os.path.join(OUT_DIR, "PACOTE_TOTAL_IDENTICO_ANUAL.zip")

def resample_data(df, interval):
    df_res = df.copy()
    df_res['ts'] = pd.to_datetime(df_res['ts'], unit='ms')
    df_res.set_index('ts', inplace=True)
    logic = {'open':'first','high':'max','low':'min','close':'last','volume':'sum','tbb':'sum'}
    res = df_res.resample(interval).apply(logic).dropna()
    res.reset_index(inplace=True)
    res['ts'] = res['ts'].astype(np.int64) // 10**6
    return res

def feature_engine_78_cols(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    # 1. As 15 de Ouro
    df["ret1"] = df["close"].pct_change()
    df["vol_realized"] = df["close"].pct_change().rolling(20).std() * np.sqrt(20)
    for p in [9, 20]:
        df[f'ema{p}'] = df['close'].ewm(span=p).mean()
        df[f'dist_ema{p}'] = (df['close'] - df[f'ema{p}']) / df[f'ema{p}']
    df['slope20'] = df['close'].rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)==20 else 0)
    # Adicionando volume e tbb como features diretas
    df['vol_norm'] = df['volume'] / df['volume'].rolling(20).mean()
    df['tbb_norm'] = df['tbb'] / df['volume']
    
    # 2. PADDING - Preencher até 78 colunas (0 a 77)
    # Isso garante que o seu .pkl local identifique o "shape" correto
    total_features_needed = 78
    current_feat_cols = [c for c in df.columns if c not in ['ts','open','high','low','close','volume','tbb']]
    
    for i in range(len(current_feat_cols), total_features_needed):
        df[f'feat_placeholder_{i}'] = 0.0
    
    return df, [c for c in df.columns if c not in ['ts','open','high','low','close','volume','tbb']][:78]

def upload_catbox(filepath):
    print(f">>> 🚀 Enviando para o Catbox...", flush=True)
    try:
        url = "https://catbox.moe/user/api.php"
        with open(filepath, "rb") as f:
            response = requests.post(url, data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=600)
            if response.status_code == 200:
                print(f"\n✅ LINK DO CATBOX: {response.text.strip()}\n")
                return response.text.strip()
    except Exception as e:
        print(f"❌ Falha no upload: {e}")
    return None

def executar_v58():
    # 1. Download de 1 Ano (35.000 velas)
    print(">>> 📅 Iniciando captura de 1 ANO...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    curr_end = int(END_DT.timestamp() * 1000)
    start_ts = int(START_DT.timestamp() * 1000)
    
    while curr_end > start_ts:
        params = {'symbol': SYMBOL, 'interval': '15m', 'limit': 1000, 'endTime': curr_end}
        res = requests.get(url, params=params).json()
        if not res: break
        all_data = res + all_data
        curr_end = res[0][0] - 1
        if len(all_data) >= 35000: break
    
    df_raw = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','tr','tbb','tbq','i'])
    df_raw = df_raw[['ts','open','high','low','close','volume','tbb']].apply(pd.to_numeric)
    
    # 2. Gerar Todos os CSVs de Timeframe
    print(">>> 📦 Gerando Multiframe CSVs...")
    tfs = {
        "15m": df_raw,
        "30m": resample_data(df_raw, '30min'),
        "1h":  resample_data(df_raw, '1h'),
        "4h":  resample_data(df_raw, '4h'),
        "8h":  resample_data(df_raw, '8h'),
        "1d":  resample_data(df_raw, '1D')
    }

    # 3. Preparar Treino (78 Colunas)
    df_feat, feature_list = feature_engine_78_cols(df_raw)
    
    # Criar 10 alvos
    for k in range(1, 7):
        df_feat[f'target_k{k}'] = (df_feat['close'].shift(-k) > df_feat['close']).astype(int)
    
    df_feat['target_A_bin'] = (df_feat['vol_realized'] > df_feat['vol_realized'].median()).astype(int)
    df_feat['target_A_lgbm'] = np.where(df_feat['ret1'] > 0.005, 2, np.where(df_feat['ret1'] < -0.005, 0, 1))
    df_feat['target_B'] = (df_feat['close'].shift(-12) > df_feat['close']).astype(int)
    df_feat['target_C'] = (df_feat['vol_realized'].shift(-4) > df_feat['vol_realized'] * 1.5).astype(int)
    
    df_feat = df_feat.dropna()
    X = df_feat[feature_list]
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    # 4. Treinar a "Banda" de 10 Modelos
    print(">>> 🧠 Treinando Bateria de 10 Modelos...")
    modelos = {
        'target_K1_XGB': xgb.XGBClassifier().fit(X_s, df_feat['target_k1']),
        'target_K2_XGB': xgb.XGBClassifier().fit(X_s, df_feat['target_k2']),
        'target_K3_LGBM': lgb.LGBMClassifier(verbosity=-1).fit(X_s, df_feat['target_k3']),
        'target_K4_LGBM': lgb.LGBMClassifier(verbosity=-1).fit(X_s, df_feat['target_k4']),
        'target_K5_LGBM': lgb.LGBMClassifier(verbosity=-1).fit(X_s, df_feat['target_k5']),
        'target_K6_XGB': xgb.XGBClassifier().fit(X_s, df_feat['target_k6']),
        'target_A_bin_XGB': xgb.XGBClassifier().fit(X_s, df_feat['target_A_bin']),
        'target_A_LGBM': lgb.LGBMClassifier(verbosity=-1).fit(X_s, df_feat['target_A_lgbm']),
        'target_B_XGB': xgb.XGBClassifier().fit(X_s, df_feat['target_B']),
        'target_C_LGBM': lgb.LGBMClassifier(verbosity=-1).fit(X_s, df_feat['target_C'])
    }

    # 5. Salvar e ZIPAR TUDO
    with zipfile.ZipFile(ZIP_FINAL_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # Salva CSVs
        for name, data in tfs.items():
            fname = f"{SYMBOL}_{name}_1YEAR.csv"
            data.to_csv(fname, index=False)
            z.write(fname)
            
        # Salva Modelos
        for m_name, m_obj in modelos.items():
            joblib.dump(m_obj, f"{m_name}.pkl")
            z.write(f"{m_name}.pkl")
            
        # Salva Scaler
        joblib.dump(scaler, "scaler_regimes_1YEAR.pkl")
        z.write("scaler_regimes_1YEAR.pkl")

    upload_catbox(ZIP_FINAL_PATH)
    print(">>> ✅ FIM DO PROCESSO.")

if __name__ == "__main__":
    executar_v58()