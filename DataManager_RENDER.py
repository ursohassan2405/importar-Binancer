import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

# CONFIGURAÇÃO FIXA PARA RENDER (SEM INPUTS)
def get_config():
    config_path = "config_render.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('data_manager', {})
    return {}

cfg = get_config()
SIMBOLO = cfg.get('symbol', 'RUNEUSDT')
TIMEFRAME = cfg.get('timeframe', '15m')
DAYS_BACK = cfg.get('days_back', 180)
MIN_VAL_USD = cfg.get('min_val_usd', 500)
PASTA_DADOS = "/data"

CONTRATO = ["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote","close_time"]

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    ms_map = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000}
    step = ms_map.get(interval, 60000 )
    while curr < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": int(curr), "endTime": int(end_ms), "limit": 1500}
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data: break
                rows.extend(data)
                curr = data[-1][0] + step
            elif r.status_code in [418, 429]: time.sleep(30)
            else: time.sleep(2)
        except: time.sleep(5)
    return rows

def download_agg_trades_industrial(symbol, start_ms, end_ms, min_val):
    print(f"[INFO] Whale Hunter Ativado: >${min_val} em {symbol}")
    all_trades = []
    curr = start_ms
    window = 3600000 # 1 hora exata (limite da Binance)
    
    with requests.Session() as session:
        while curr < end_ms:
            next_stop = min(curr + window, end_ms)
            last_id = None
            whales_na_hora = 0
            
            while True:
                params = {"symbol": symbol, "limit": 1000}
                if last_id is None:
                    params["startTime"] = int(curr)
                    params["endTime"] = int(next_stop)
                else:
                    params["fromId"] = last_id + 1
                
                try:
                    r = session.get("https://fapi.binance.com/fapi/v1/aggTrades", params=params, timeout=20 )
                    if r.status_code != 200:
                        time.sleep(5)
                        continue
                    
                    data = r.json()
                    if not data: break
                    
                    for t in data:
                        ts_trade = int(t['T'])
                        if ts_trade >= next_stop: break
                        
                        val = float(t['p']) * float(t['q'])
                        if val >= min_val:
                            whales_na_hora += 1
                            all_trades.append([ts_trade, float(t['p']), float(t['q']), 1 if t['m'] else 0])
                        last_id = int(t['a'])
                    
                    if ts_trade >= next_stop or len(data) < 1000: break
                except:
                    time.sleep(5)
                    break
            
            print(f"    [OK] {pd.to_datetime(curr, unit='ms')} | Baleias: {whales_na_hora}")
            curr = next_stop
            
    return np.array(all_trades) if all_trades else None

def main():
    print(f"\nINICIANDO DATA MANAGER RENDER: {SIMBOLO} [{TIMEFRAME}]")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (DAYS_BACK * 24 * 3600 * 1000)
    
    # 1. DOWNLOAD TRADES
    trades_np = download_agg_trades_industrial(SIMBOLO, start_ms, end_ms, MIN_VAL_USD)
    
    # 2. DOWNLOAD KLINES
    with requests.Session() as s:
        rows = download_klines_chunk(SIMBOLO, TIMEFRAME, start_ms, end_ms, s)
    
    if not rows:
        print("Erro ao baixar Klines."); return
        
    df = pd.DataFrame(rows, columns=CONTRATO + ["ignore"])[CONTRATO]
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    
    # 3. INJETAR MICROESTRUTURA
    if trades_np is not None:
        print(f"Injetando {len(trades_np)} trades institucionais...")
        tdf = pd.DataFrame(trades_np, columns=['ts','p','q','m'])
        tdf['ts'] = pd.to_datetime(tdf['ts'], unit='ms')
        tdf['buy'] = np.where(tdf['m']==0, tdf['q'], 0)
        tdf['sell'] = np.where(tdf['m']==1, tdf['q'], 0)
        
        res = tdf.set_index('ts').resample(TIMEFRAME).agg({'buy':'sum','sell':'sum','q':'count'})
        res['delta'] = res['buy'] - res['sell']
        res['cum_delta'] = res['delta'].cumsum()
        
        df = df.set_index('ts').join(res).fillna(0).reset_index()
    
    # 4. SALVAR NO DISCO
    if not os.path.exists(PASTA_DADOS): os.makedirs(PASTA_DADOS)
    caminho = f"{PASTA_DADOS}/{SIMBOLO}_{TIMEFRAME}.csv"
    df.to_csv(caminho, index=False)
    print(f"\nSUCESSO! Arquivo salvo em: {caminho}")

if __name__ == "__main__":
    main()
