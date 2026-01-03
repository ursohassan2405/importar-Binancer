import requests
import pandas as pd
import time
import os
import numpy as np
import json
from datetime import datetime, timedelta

# CONFIGURAÇÃO DE ENDPOINTS PARA EVITAR BLOQUEIO DE IP NA RENDER
ENDPOINTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com"
]

def get_config( ):
    config_path = "config_render.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('data_manager', {})
    return {}

cfg = get_config()
SIMBOLO = cfg.get('symbol', 'RUNEUSDT')
TIMEFRAME = cfg.get('timeframe', '15m')
DAYS_BACK = cfg.get('days_back', 30) # Reduzi para 30 para estabilizar o primeiro teste
MIN_VAL_USD = cfg.get('min_val_usd', 100)
PASTA_DADOS = "/data"

def get_binance_time():
    for base in ENDPOINTS:
        try:
            return requests.get(f"{base}/fapi/v1/time", timeout=10).json()['serverTime']
        except: continue
    return int(time.time() * 1000)

def download_agg_trades_resilient(symbol, start_ms, end_ms, min_val):
    print(f"🚀 Iniciando Whale Hunter Resiliente: {symbol} > ${min_val}")
    all_trades = []
    curr = start_ms
    window = 3600000 # 1 hora
    
    session = requests.Session()
    # Simular um navegador real para evitar bloqueios
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

    while curr < end_ms:
        next_stop = min(curr + window, end_ms)
        success = False
        
        for base in ENDPOINTS:
            url = f"{base}/fapi/v1/aggTrades"
            params = {"symbol": symbol, "startTime": int(curr), "endTime": int(next_stop), "limit": 1000}
            
            try:
                r = session.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        for t in data:
                            val = float(t['p']) * float(t['q'])
                            if val >= min_val:
                                all_trades.append([int(t['T']), float(t['p']), float(t['q']), 1 if t['m'] else 0])
                    success = True
                    print(f"    [OK] {pd.to_datetime(curr, unit='ms')} | Baleias: {len(data) if data else 0}")
                    break
                elif r.status_code == 429:
                    print("⚠️ Rate limit atingido. Trocando endpoint...")
                    time.sleep(10)
            except:
                continue
        
        if not success:
            print(f"❌ Falha na janela {curr}. Pulando...")
        
        curr = next_stop
        time.sleep(0.5) # Delay de segurança
            
    return np.array(all_trades) if all_trades else None

def main():
    if not os.path.exists(PASTA_DADOS):
        try: os.makedirs(PASTA_DADOS)
        except: pass

    server_time = get_binance_time()
    start_ms = server_time - (DAYS_BACK * 24 * 3600 * 1000)
    
    # 1. Download de Trades
    trades_np = download_agg_trades_resilient(SIMBOLO, start_ms, server_time, MIN_VAL_USD)
    
    # 2. Download de Klines (OHLCV)
    print(f"📊 Baixando OHLCV {TIMEFRAME}...")
    klines_url = f"{ENDPOINTS[0]}/fapi/v1/klines"
    params = {"symbol": SIMBOLO, "interval": TIMEFRAME, "startTime": start_ms, "limit": 1500}
    rows = requests.get(klines_url, params=params).json()
    
    if not rows:
        print("❌ Erro ao baixar OHLCV."); return

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","qv","tr","tbb","tbq","ignore","ignore2"])
    df = df[["ts","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')

    # 3. Injeção de Microestrutura
    if trades_np is not None and len(trades_np) > 0:
        print(f"💉 Injetando {len(trades_np)} trades de baleias...")
        tdf = pd.DataFrame(trades_np, columns=['ts','p','q','m'])
        tdf['ts'] = pd.to_datetime(tdf['ts'], unit='ms')
        tdf['buy'] = np.where(tdf['m']==0, tdf['q'], 0)
        tdf['sell'] = np.where(tdf['m']==1, tdf['q'], 0)
        
        res = tdf.set_index('ts').resample(TIMEFRAME).agg({'buy':'sum','sell':'sum','q':'count'})
        res['delta'] = res['buy'] - res['sell']
        res['cum_delta'] = res['delta'].cumsum()
        
        df = df.set_index('ts').join(res).fillna(0).reset_index()
    
    # 4. Salvar
    caminho = f"{PASTA_DADOS}/{SIMBOLO}_{TIMEFRAME}.csv"
    df.to_csv(caminho, index=False)
    print(f"✅ SUCESSO! Arquivo salvo: {caminho} | Linhas: {len(df)}")

if __name__ == "__main__":
    main()

