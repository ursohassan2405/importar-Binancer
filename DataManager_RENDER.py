# DATA MANAGER INSTITUCIONAL ELITE V16 - RENDER VERSION (FULL LOGIC)
# Arquitetura de Agregação Inteligente (1m -> Multi-TF)
# Otimizado para Microestrutura e Alta Performance
# VERSÃO BLINDADA: SEM INPUTS, LÊ TUDO DO CONFIG_RENDER.JSON

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

CONTRATO = [
    "ts","open","high","low","close","volume",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","close_time"
]

def banner():
    print("\n" + "="*50)
    print("   DATA MANAGER INSTITUCIONAL ELITE V16")
    print("   >>> Smart Aggregation & Microstructure <<<")
    print("="*50 + "\n")

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    limit = 1500
    ms_map = {"1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "30m": 1800000, "1h": 3600000}
    ms_per_kline = ms_map.get(interval, 60000 )
    while curr < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": int(curr), "endTime": int(end_ms), "limit": limit}
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data: break
                rows.extend(data)
                curr = data[-1][0] + ms_per_kline
                if len(data) < limit: break 
            elif r.status_code in [418, 429]:
                time.sleep(15)
            else:
                time.sleep(2)
        except:
            time.sleep(5)
    return rows

def download_futures_klines(symbol, interval, start_ms, end_ms):
    print(f"[INFO] Baixando Klines {interval} para {symbol}...")
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
    if not all_rows: return []
    df_temp = pd.DataFrame(all_rows).drop_duplicates(subset=0).sort_values(by=0)
    return df_temp.values.tolist()

def download_agg_trades_parallel(symbol, start_ms, end_ms, min_val_usd=100):
    print(f"[INFO] Baixando AggTrades (Monolithic Mode) para {symbol}...")
    all_chunks = []
    total_trades = 0
    hour_ms = 3600 * 1000
    curr = start_ms
    with requests.Session() as session:
        while curr < end_ms:
            try:
                next_stop = min(curr + hour_ms, end_ms)
                last_id = None
                whales_in_hour = 0
                while True:
                    time.sleep(0.1) 
                    # Lógica de Paginação por ID (A mesma do seu original)
                    if last_id is None:
                        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&startTime={curr}&endTime={next_stop}&limit=1000"
                    else:
                        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&fromId={last_id + 1}&limit=1000"
                    
                    res = session.get(url, timeout=20 )
                    if res.status_code in [418, 429]:
                        time.sleep(30)
                        continue
                    res_raw = res.json()
                    if not isinstance(res_raw, list) or len(res_raw) == 0: break
                    
                    chunk = []
                    last_trade_ts = 0
                    for t in res_raw:
                        last_trade_ts = int(t['T'])
                        if last_trade_ts >= next_stop: break
                        val = float(t['p']) * float(t['q'])
                        if val >= min_val_usd:
                            whales_in_hour += 1
                            chunk.append([last_trade_ts, float(t['p']), float(t['q']), 1 if t['m'] else 0])
                    
                    if chunk:
                        all_chunks.append(np.array(chunk))
                        total_trades += len(chunk)
                    
                    last_id = int(res_raw[-1]['a'])
                    if last_trade_ts >= next_stop or len(res_raw) < 1000: break
                
                print(f"    [OK] {pd.to_datetime(curr, unit='ms')} | Baleias: {whales_in_hour}")
                curr = next_stop
            except:
                time.sleep(10)
                continue
    if not all_chunks: return None
    return np.vstack(all_chunks)

def normalize_klines(rows):
    df = pd.DataFrame(rows, columns=CONTRATO + ["ignore"])
    df = df[CONTRATO].copy()
    for col in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def processar_microestrutura_dinamica(df_ohlcv, trades_np, tf):
    df_trades = pd.DataFrame(trades_np, columns=['ts', 'price', 'qty', 'is_buyer_maker'])
    df_trades['ts'] = pd.to_datetime(df_trades['ts'], unit='ms')
    df_trades['buy_vol'] = np.where(df_trades['is_buyer_maker'] == False, df_trades['qty'], 0)
    df_trades['sell_vol'] = np.where(df_trades['is_buyer_maker'] == True, df_trades['qty'], 0)
    df_trades.set_index('ts', inplace=True)
    resampled = df_trades.resample(tf).agg({'buy_vol': 'sum', 'sell_vol': 'sum', 'qty': 'count'}).rename(columns={'qty': 'n_trades_whale'})
    resampled['delta'] = resampled['buy_vol'] - resampled['sell_vol']
    resampled['cum_delta'] = resampled['delta'].cumsum()
    df_ohlcv.set_index('ts', inplace=True)
    df_final = df_ohlcv.join(resampled).fillna(0)
    return df_final.reset_index()

def main():
    banner()
    config_path = "config_render.json"
    if not os.path.exists(config_path): return
    with open(config_path, 'r') as f:
        config = json.load(f)
    dm_cfg = config['data_manager']
    simbolo = dm_cfg.get('symbol', 'RUNEUSDT')
    tfs_input = dm_cfg.get('timeframe', '15m')
    tfs = [x.strip() for x in tfs_input.split(',')]
    days_back = dm_cfg.get('days_back', 180)
    min_val = dm_cfg.get('min_val_usd', 500)
    pasta = "/data" 
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    trades_np = download_agg_trades_parallel(simbolo, start_ms, end_ms + 60000, min_val_usd=min_val)
    for tf in tfs:
        rows = download_futures_klines(simbolo, tf, start_ms, end_ms)
        if not rows: continue
        df_final = normalize_klines(rows)
        if trades_np is not None:
            df_final = processar_microestrutura_dinamica(df_final, trades_np, tf)
        saida_csv = os.path.join(pasta, f"{simbolo}_{tf}.csv")
        df_final.to_csv(saida_csv, index=False)
        print(f"    [OK] Arquivo gerado: {saida_csv} | Linhas: {len(df_final)}")
    print("\nDataManager Concluído!")

if __name__ == "__main__":
    main()
