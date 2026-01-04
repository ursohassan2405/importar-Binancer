# DATA MANAGER INSTITUCIONAL ELITE V16 - V17 ULTRA FAST EDITION
# 1. ENDPOINTS 100% SPOT (api.binance.com)
# 2. ULTRA FAST: Zero logs dentro do loop para máxima velocidade
# 3. ID Discovery Inquebrável + Auto-Retry 10x
# 4. Sincronização Forçada de Timestamps

import requests
import pandas as pd
import time
import os
import zipfile
import numpy as np
import sys

def log(msg):
    print(msg)
    sys.stdout.flush()

def get_first_id_by_time(symbol, start_ms):
    url = "https://api.binance.com/api/v3/aggTrades"
    params = {"symbol": symbol, "startTime": int(start_ms), "limit": 1}
    for _ in range(10):
        try:
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            if data and isinstance(data, list): return data[0]['a']
            time.sleep(2)
        except: time.sleep(5)
    return None

def download_agg_trades_sequential(symbol, start_ms, end_ms, min_val_usd=500):
    log(f"[INFO] Iniciando Captura SPOT Ultra Fast (Filtro: >${min_val_usd})...")
    
    curr_id = get_first_id_by_time(symbol, start_ms)
    if not curr_id: return None
    
    all_chunks = []
    total_whales = 0
    last_ts = start_ms
    
    with requests.Session() as session:
        while last_ts < end_ms:
            try:
                url = f"https://api.binance.com/api/v3/aggTrades?symbol={symbol}&fromId={curr_id}&limit=1000"
                res = session.get(url, timeout=20)
                
                if res.status_code in [418, 429]:
                    time.sleep(30)
                    continue
                
                data = res.json()
                if not data or not isinstance(data, list): break
                
                chunk = []
                for t in data:
                    ts = int(t['T'])
                    if ts > end_ms: break
                    if float(t['p']) * float(t['q']) >= min_val_usd:
                        chunk.append([ts, float(t['p']), float(t['q']), 1 if t['m'] else 0])
                
                if chunk:
                    all_chunks.append(np.array(chunk))
                    total_whales += len(chunk)
                
                last_ts = int(data[-1]['T'])
                curr_id = int(data[-1]['a']) + 1
                
                if last_ts >= end_ms: break
                # Sem prints aqui para máxima velocidade
                
            except:
                time.sleep(5)
                continue
                
    if not all_chunks: return None
    log(f"[SUCESSO] Captura concluída: {total_whales} baleias encontradas.")
    return np.vstack(all_chunks)

def download_klines_parallel(symbol, start_ms, end_ms):
    log("[INFO] Baixando Klines SPOT...")
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    curr = start_ms
    with requests.Session() as session:
        while curr < end_ms:
            params = {"symbol": symbol, "interval": "1m", "startTime": int(curr), "endTime": int(end_ms), "limit": 1000}
            try:
                r = session.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    if not data: break
                    all_rows.extend(data)
                    curr = data[-1][0] + 60000
                else: time.sleep(5)
            except: time.sleep(5)
    return all_rows

def processar_final(rows_1m, trades_np, tf_str):
    log(f"[INFO] Consolidando para {tf_str}...")
    df = pd.DataFrame(rows_1m, columns=["ts","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
    for col in ["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["ts"] = df["ts"].astype(np.int64)

    if trades_np is not None:
        ms_1m = 60000
        ts_target_1m = ((trades_np[:, 0] // ms_1m) * ms_1m).astype(np.int64)
        df_agg = pd.DataFrame({
            'ts': ts_target_1m, 
            'cum_delta': np.where(trades_np[:, 3].astype(bool), -trades_np[:, 2], trades_np[:, 2]), 
            'total_vol_agg': trades_np[:, 2], 
            'buy_vol_agg': np.where(~trades_np[:, 3].astype(bool), trades_np[:, 2], 0), 
            'sell_vol_agg': np.where(trades_np[:, 3].astype(bool), trades_np[:, 2], 0)
        })
        micro_1m = df_agg.groupby('ts').sum().reset_index()
        df = pd.merge(df, micro_1m, on='ts', how='left').fillna(0)
    else:
        for c in ['cum_delta', 'total_vol_agg', 'buy_vol_agg', 'sell_vol_agg']: df[c] = 0

    df['vpin'] = (df['buy_vol_agg'] - df['sell_vol_agg']).abs() / (df['volume'] + 1e-9)
    df['absorcao'] = df['total_vol_agg'] / (df['volume'] + 1e-9)

    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    agg_logic = {
        'ts': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 
        'volume': 'sum', 'quote_volume': 'sum', 'trades': 'sum', 
        'taker_buy_base': 'sum', 'taker_buy_quote': 'sum', 'close_time': 'last',
        'cum_delta': 'sum', 'total_vol_agg': 'sum', 'buy_vol_agg': 'sum', 'sell_vol_agg': 'sum',
        'vpin': 'mean', 'absorcao': 'mean'
    }
    
    tf_map = {'1m': '1min', '15m': '15min', '1h': '1H'}
    df_res = df.resample(tf_map.get(tf_str, '15min')).agg(agg_logic).dropna()
    return df_res.reset_index(drop=True)

def upload_to_catbox(file_path):
    log("[UPLOAD] Enviando para Catbox...")
    url = "https://catbox.moe/user/api.php"
    try:
        with open(file_path, 'rb') as f:
            res = requests.post(url, data={'reqtype': 'fileupload'}, files={'fileToUpload': f})
        if res.status_code == 200:
            log("\n" + "="*60)
            log("   LINK DE DOWNLOAD DIRETO:")
            log(f"   {res.text}")
            log("="*60 + "\n")
    except: pass

def main():
    simbolo, tf, start_str, end_str, pasta = "PENDLEUSDT", "15m", "2024-01-01", "2024-06-30", "/data"
    start_ms = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end_str).timestamp() * 1000)
    if not os.path.isdir(pasta): os.makedirs(pasta)
    
    trades_np = download_agg_trades_sequential(simbolo, start_ms, end_ms, min_val_usd=500)
    rows_1m = download_klines_parallel(simbolo, start_ms, end_ms)
    
    if rows_1m:
        df_final = processar_final(rows_1m, trades_np, tf)
        saida_csv = os.path.join(pasta, f"{simbolo}_{tf}.csv")
        df_final.to_csv(saida_csv, index=False, sep=',', float_format='%.8f')
        zip_name = saida_csv.replace(".csv", ".zip")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(saida_csv, os.path.basename(saida_csv))
        upload_to_catbox(zip_name)

if __name__ == "__main__":
    main()
