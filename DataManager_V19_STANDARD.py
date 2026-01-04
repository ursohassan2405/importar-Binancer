import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# ========================================================================
# DATA MANAGER V19 — STANDARD & PERSISTENT (SPOT)
# ========================================================================
# Foco: Consistência de logs, velocidade e sobrevivência a reinicializações.

SYMBOL = "PENDLEUSDT"
INTERVAL = "15m"
START_DATE = "2024-01-01"
END_DATE = "2024-06-30"
MIN_VAL_USD = 500
BASE_URL = "https://api.binance.com/api/v3"

def get_klines(symbol, interval, start_str, end_str):
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    all_klines = []
    curr_ts = start_ts
    while curr_ts < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": curr_ts, "endTime": end_ts, "limit": 1000}
        res = requests.get(f"{BASE_URL}/klines", params=params).json()
        if not res: break
        all_klines.extend(res)
        curr_ts = res[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(all_klines, columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["ts"] = df["ts"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]: df[col] = df[col].astype(float)
    return df[["ts", "open", "high", "low", "close", "volume"]]

def download_trades_chunk(symbol, start_ts, end_ts):
    trades = []
    res = requests.get(f"{BASE_URL}/aggTrades", params={"symbol": symbol, "startTime": start_ts, "limit": 1}).json()
    if not res: return None
    curr_id = res[0]['a']
    last_log_time = time.time()
    
    while True:
        try:
            res = requests.get(f"{BASE_URL}/aggTrades", params={"symbol": symbol, "fromId": curr_id, "limit": 1000}).json()
            if not res: break
            for t in res:
                ts, p, q = int(t['T']), float(t['p']), float(t['q'])
                if ts > end_ts: return trades
                if p * q >= MIN_VAL_USD: trades.append([ts, p, q, 1 if not t['m'] else -1])
            curr_id = res[-1]['a'] + 1
            
            # Batida de Coração (Log a cada 5 minutos)
            if time.time() - last_log_time > 300:
                print(f"   [STATUS] {datetime.fromtimestamp(res[-1]['T']/1000).strftime('%Y-%m-%d %H:%M')} | Capturados: {len(trades)} baleias...")
                last_log_time = time.time()
                
            if res[-1]['T'] > end_ts: break
            time.sleep(0.05)
        except:
            time.sleep(2); continue
    return trades

def main():
    print(f"\n===============================================================")
    print(f">>> MOTOR LIGADO: DATA MANAGER V19 STANDARD")
    print(f">>> ATIVO: {SYMBOL} SPOT | FILTRO: ${MIN_VAL_USD}")
    print(f"===============================================================\n")
    
    os.makedirs("progress", exist_ok=True)
    months = [
        ("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-31"), ("2024-04-01", "2024-04-30"),
        ("2024-05-01", "2024-05-31"), ("2024-06-01", "2024-06-30"),
    ]
    
    all_monthly_data = []
    for start_m, end_m in months:
        file_path = f"progress/data_{start_m}.csv"
        if os.path.exists(file_path):
            print(f"✔ Mês {start_m} já concluído. Carregando do disco...")
            all_monthly_data.append(pd.read_csv(file_path))
            continue
        
        print(f"⏳ Iniciando download: {start_m} até {end_m}...")
        df_klines = get_klines(SYMBOL, INTERVAL, start_m, end_m)
        start_ts = int(datetime.strptime(start_m, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_m, "%Y-%m-%d").timestamp() * 1000) + 86399999
        
        raw_trades = download_trades_chunk(SYMBOL, start_ts, end_ts)
        if raw_trades:
            trades_np = np.array(raw_trades)
            ms_15m = 15 * 60 * 1000
            ts_target = (trades_np[:, 0] // ms_15m) * ms_15m
            df_agg = pd.DataFrame({'ts': ts_target.astype(np.int64), 'cum_delta': trades_np[:, 2] * trades_np[:, 3], 'total_vol_agg': trades_np[:, 2], 'buy_vol_agg': np.where(trades_np[:, 3] == 1, trades_np[:, 2], 0), 'sell_vol_agg': np.where(trades_np[:, 3] == -1, trades_np[:, 2], 0)})
            micro = df_agg.groupby('ts').sum().reset_index()
            df_final = pd.merge(df_klines, micro, on='ts', how='left').fillna(0)
            df_final['vpin'] = (df_final['buy_vol_agg'] - df_final['sell_vol_agg']).abs() / (df_final['volume'] + 1e-9)
            df_final['absorcao'] = df_final['total_vol_agg'] / (df_final['volume'] + 1e-9)
            df_final.to_csv(file_path, index=False)
            all_monthly_data.append(df_final)
            print(f"✅ Mês {start_m} FINALIZADO com sucesso.\n")
        else:
            print(f"⚠ Mês {start_m} sem dados de baleias.")

    if all_monthly_data:
        full_df = pd.concat(all_monthly_data).drop_duplicates(subset=['ts']).sort_values('ts')
        final_name = f"{SYMBOL}_{INTERVAL}.csv"
        full_df.to_csv(final_name, index=False)
        print(f"\n🚀 PROCESSO CONCLUÍDO! Arquivo final: {final_name}")
        print(f">>> Localização: {os.path.abspath(final_name)}")
    else:
        print("❌ Erro: Nenhum dado foi processado.")

if __name__ == "__main__":
    main()
