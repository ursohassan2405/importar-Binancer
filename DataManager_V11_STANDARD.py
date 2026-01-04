# DATA MANAGER INSTITUCIONAL ELITE V16 - V11 STANDARD EDITION
# 1. Corrige Diluição (Sliding Window)
# 2. Captura Total (ID Discovery + Time Fallback)
# 3. Log a cada 10.000 trades (COM FLUSH)
# 4. Padronização Total de Colunas (Igual ao TIAUSDT_15m.csv)
# 5. Cálculo de VPIN e Absorção Integrado

import requests
import pandas as pd
import time
import os
import zipfile
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor

def log(msg):
    print(msg)
    sys.stdout.flush()

def get_first_id(symbol, start_ms):
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    params = {"symbol": symbol, "startTime": int(start_ms), "limit": 1}
    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        if data and isinstance(data, list): return data[0]['a']
    except: pass
    return None

def download_agg_trades_parallel(symbol, start_ms, end_ms, min_val_usd=500):
    log(f"\n[INFO] Iniciando Captura de Baleias (Filtro: >${min_val_usd})")
    curr_id = get_first_id(symbol, start_ms)
    use_id = True if curr_id else False
    if not use_id: curr_ts = start_ms
    
    all_chunks = []
    total_whales = 0
    
    with requests.Session() as session:
        while True:
            try:
                url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&{'fromId' if use_id else 'startTime'}={curr_id if use_id else curr_ts}&limit=1000"
                res = session.get(url, timeout=15)
                if res.status_code == 429:
                    time.sleep(20)
                    continue
                data = res.json()
                if not data or not isinstance(data, list): break
                
                last_ts = int(data[-1]['T'])
                last_id_seen = int(data[-1]['a'])
                
                chunk = []
                for t in data:
                    ts = int(t['T'])
                    if ts > end_ms: break
                    p, q = float(t['p']), float(t['q'])
                    if p * q >= min_val_usd:
                        chunk.append([ts, p, q, 1 if t['m'] else 0])
                
                if chunk:
                    all_chunks.append(np.array(chunk))
                    total_whales += len(chunk)
                
                if last_id_seen % 10000 < 1000:
                    dt = pd.to_datetime(last_ts, unit='ms').strftime('%Y-%m-%d %H:%M')
                    log(f"    >>> Progresso: {dt} | Baleias: {total_whales}")
                
                if last_ts > end_ms: break
                if use_id: curr_id = last_id_seen + 1
                else: curr_ts = last_ts + 1
                time.sleep(0.01)
            except: time.sleep(5)
    return np.vstack(all_chunks) if all_chunks else None

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    while curr < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": int(curr), "endTime": int(end_ms), "limit": 1500}
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data: break
                rows.extend(data)
                curr = data[-1][0] + 60000
                if len(data) < 1500: break
            else: time.sleep(2)
        except: time.sleep(5)
    return rows

def download_futures_klines(symbol, interval, start_ms, end_ms):
    log(f"\n[INFO] Baixando Klines de 1m...")
    slice_ms = 15 * 24 * 3600000
    chunks = []
    curr = start_ms
    while curr < end_ms:
        chunks.append((curr, min(curr + slice_ms, end_ms)))
        curr += slice_ms
    all_rows = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_klines_chunk, symbol, interval, c[0], c[1], session) for c in chunks]
            for f in futures:
                res = f.result()
                if res: all_rows.extend(res)
    return all_rows

def processar_microestrutura_dinamica(rows_1m, trades_np, tf_str):
    log(f"\n[INFO] Consolidando para {tf_str} e calculando indicadores...")
    df = pd.DataFrame(rows_1m, columns=["ts","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
    for col in ["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if trades_np is not None:
        ms_1m = 60000
        ts_target_1m = (trades_np[:, 0] // ms_1m) * ms_1m
        df_agg = pd.DataFrame({'ts': ts_target_1m, 'cum_delta': np.where(trades_np[:, 3].astype(bool), -trades_np[:, 2], trades_np[:, 2]), 'total_vol_agg': trades_np[:, 2], 'buy_vol_agg': np.where(~trades_np[:, 3].astype(bool), trades_np[:, 2], 0), 'sell_vol_agg': np.where(trades_np[:, 3].astype(bool), trades_np[:, 2], 0)})
        micro_1m = df_agg.groupby('ts').sum().reset_index()
        df = pd.merge(df, micro_1m, on='ts', how='left').fillna(0)
    else:
        df['cum_delta'] = 0
        df['total_vol_agg'] = 0
        df['buy_vol_agg'] = 0
        df['sell_vol_agg'] = 0

    # Cálculo de VPIN e Absorção (Simplificado para 1m antes da resample)
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
    log(f"\n[UPLOAD] Enviando para Catbox...")
    url = "https://catbox.moe/user/api.php"
    try:
        with open(file_path, 'rb') as f:
            res = requests.post(url, data={'reqtype': 'fileupload'}, files={'fileToUpload': f})
        if res.status_code == 200:
            log("\n" + "="*60)
            log("   LINK DE DOWNLOAD DIRETO:")
            log(f"   {res.text}")
            log("="*60 + "\n")
        else:
            log(f"[ERRO UPLOAD] Status: {res.status_code}")
    except Exception as e:
        log(f"[ERRO UPLOAD] {e}")

def main():
    simbolo, tf, start_str, end_str, pasta = "PENDLEUSDT", "15m", "2024-01-01", "2024-06-30", "/data"
    start_ms = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end_str).timestamp() * 1000)
    if not os.path.isdir(pasta): os.makedirs(pasta)
    
    trades_np = download_agg_trades_parallel(simbolo, start_ms, end_ms)
    rows_1m = download_futures_klines(simbolo, '1m', start_ms, end_ms)
    if not rows_1m: return
    
    df_final = processar_microestrutura_dinamica(rows_1m, trades_np, tf)
    saida_csv = os.path.join(pasta, f"{simbolo}_{tf}.csv")
    
    # Salva com delimitador explícito e sem fusão de colunas
    df_final.to_csv(saida_csv, index=False, sep=',', float_format='%.8f')
    
    zip_name = saida_csv.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(saida_csv, os.path.basename(saida_csv))
    
    log(f"\n[SUCESSO] Arquivo local: {zip_name}")
    upload_to_catbox(zip_name)

if __name__ == "__main__":
    main()
