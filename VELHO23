# DATA MANAGER INSTITUCIONAL ELITE V16 - CORRIGIDO V2
# Otimizado para Microestrutura e Alta Performance
# CORREÇÃO: Lógica de download de AggTrades refeita para evitar "Zero Trades".
# VERSÃO FIXA PARA RENDER (NÃO INTERATIVA)

import requests
import pandas as pd
import time
from datetime import datetime
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

CONTRATO = [
    "ts","open","high","low","close","volume",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","close_time"
]

# -----------------------------------------------------------------------------
# DOWNLOADER DE ALTA PERFORMANCE
# -----------------------------------------------------------------------------

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    limit = 1500
    ms_per_kline = 60000 
    
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
        except Exception:
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

def download_agg_trades_parallel(symbol, start_ms, end_ms, min_val_usd=500):
    """Download robusto de AggTrades com garantia de captura."""
    print(f"[INFO] Baixando AggTrades para {symbol} (Filtro: >${min_val_usd})...")
    
    all_chunks = []
    curr = start_ms
    total_whales = 0
    
    # Janelas de 1 hora para evitar timeouts e garantir precisão
    hour_ms = 3600 * 1000
    
    with requests.Session() as session:
        while curr < end_ms:
            next_stop = min(curr + hour_ms, end_ms)
            try:
                # URL baseada em tempo para a janela atual
                url = "https://fapi.binance.com/fapi/v1/aggTrades"
                params = {
                    "symbol": symbol,
                    "startTime": int(curr),
                    "endTime": int(next_stop),
                    "limit": 1000
                }
                
                trades_in_window = 0
                whales_in_window = 0
                
                while True:
                    res = session.get(url, params=params, timeout=20)
                    if res.status_code == 429:
                        time.sleep(20)
                        continue
                    
                    data = res.json()
                    if not data or not isinstance(data, list):
                        break
                    
                    chunk = []
                    for t in data:
                        trades_in_window += 1
                        p = float(t['p'])
                        q = float(t['q'])
                        val = p * q
                        
                        if val >= min_val_usd:
                            whales_in_window += 1
                            # [Timestamp, Preço, Quantidade, É Venda]
                            chunk.append([int(t['T']), p, q, 1 if t['m'] else 0])
                    
                    if chunk:
                        all_chunks.append(np.array(chunk))
                        total_whales += len(chunk)
                    
                    # Se pegamos 1000 trades, pode haver mais no mesmo milissegundo ou logo após
                    if len(data) == 1000:
                        # Atualiza o startTime para o último trade recebido + 1ms
                        params["startTime"] = int(data[-1]['T']) + 1
                        # Se o novo startTime passou do endTime da janela, paramos
                        if params["startTime"] >= next_stop:
                            break
                    else:
                        break
                
                # Log de progresso
                dt_str = pd.to_datetime(curr, unit='ms').strftime('%Y-%m-%d %H:%M')
                print(f"    [OK] {dt_str} | Trades: {trades_in_window} | Baleias: {whales_in_window}")
                
                curr = next_stop
                time.sleep(0.05) # Pequeno delay para evitar rate limit
                
            except Exception as e:
                print(f"    [!] Erro na janela {curr}: {e}. Retentando...")
                time.sleep(5)
                continue
                
    if not all_chunks: return None
    return np.vstack(all_chunks)

# -----------------------------------------------------------------------------
# PROCESSAMENTO E AGREGAÇÃO
# -----------------------------------------------------------------------------

def normalize_klines(rows):
    if not rows: return None
    df = pd.DataFrame(rows, columns=[
        "ts","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["ts","open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df[CONTRATO]

def processar_microestrutura_dinamica(df_klines_1m, trades_np, tf_str):
    if trades_np is None or df_klines_1m is None:
        return df_klines_1m

    ms_1m = 60000
    ts_target_1m = (trades_np[:, 0] // ms_1m) * ms_1m
    q = trades_np[:, 2]
    is_sell = trades_np[:, 3].astype(bool)
    
    buy_vol = np.where(~is_sell, q, 0)
    sell_vol = np.where(is_sell, q, 0)
    
    df_agg = pd.DataFrame({
        'ts': ts_target_1m,
        'cum_delta': buy_vol - sell_vol,
        'total_vol_agg': q,
        'buy_vol_agg': buy_vol,
        'sell_vol_agg': sell_vol
    })
    
    micro_1m = df_agg.groupby('ts').sum().reset_index()
    df_klines_1m = pd.merge(df_klines_1m, micro_1m, on='ts', how='left').fillna(0)
    
    tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D'
    }
    rule = tf_map.get(tf_str, '15min')
    
    df_klines_1m['datetime'] = pd.to_datetime(df_klines_1m['ts'], unit='ms')
    df_klines_1m.set_index('datetime', inplace=True)
    
    agg_logic = {
        'ts': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'quote_volume': 'sum', 'trades': 'sum',
        'taker_buy_base': 'sum', 'taker_buy_quote': 'sum', 'close_time': 'last',
        'cum_delta': 'sum', 'total_vol_agg': 'sum', 'buy_vol_agg': 'sum', 'sell_vol_agg': 'sum'
    }
    
    agg_logic = {k: v for k, v in agg_logic.items() if k in df_klines_1m.columns}
    df_res = df_klines_1m.resample(rule).agg(agg_logic).dropna()
    
    if 'total_vol_agg' in df_res.columns:
        df_res['vpin'] = (df_res['buy_vol_agg'] - df_res['sell_vol_agg']).abs() / (df_res['total_vol_agg'] + 1e-9)
        df_res['price_range'] = (df_res['high'] - df_res['low']) / (df_res['close'] + 1e-9)
        df_res['absorcao'] = (df_res['total_vol_agg'] / (df_res['price_range'] + 1e-9)) * (1 - df_res['vpin'])
        window = 100
        df_res['absorcao'] = (df_res['absorcao'] - df_res['absorcao'].rolling(window).mean()) / (df_res['absorcao'].rolling(window).std() + 1e-9)
    
    return df_res.reset_index(drop=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    simbolo = "PENDLEUSDT"
    tf = "15m"
    start_str = "2024-01-01"
    end_str = "2024-06-30"
    pasta = "/data"
    min_val = 500.0

    start_ms = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end_str).timestamp() * 1000)

    if not os.path.isdir(pasta): os.makedirs(pasta)

    # PASSO 1: DOWNLOAD DE AGRESSÃO
    trades_np = download_agg_trades_parallel(simbolo, start_ms, end_ms, min_val_usd=min_val)

    # PASSO 2: DOWNLOAD KLINES 1M E AGREGAÇÃO
    rows_1m = download_futures_klines(simbolo, '1m', start_ms, end_ms)
    if not rows_1m: return
        
    df_klines_1m = normalize_klines(rows_1m)
    
    if trades_np is not None:
        df_final = processar_microestrutura_dinamica(df_klines_1m, trades_np, tf)
    else:
        print("[AVISO] Nenhum trade de baleia encontrado. Verifique o filtro ou o ativo.")
        return

    saida_csv = os.path.join(pasta, f"{simbolo}_{tf}.csv")
    df_final.to_csv(saida_csv, index=False)
    print(f"\n[SUCESSO] Arquivo gerado: {saida_csv} | Linhas: {len(df_final)}")

if __name__ == "__main__":
    main()
