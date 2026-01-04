import os
import time
import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# ========================================================================
# DATA MANAGER V21 — INSTANT PROOF & DAILY PERSISTENCE (SPOT)
# ========================================================================
# Foco: Prova de funcionamento nos primeiros segundos e salvamento diário.

SYMBOL = "PENDLEUSDT"
INTERVAL = "15m"
START_DATE = "2024-01-01"
END_DATE = "2024-06-30"
MIN_VAL_USD = 500
BASE_URL = "https://api.binance.com/api/v3"

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def get_klines(symbol, interval, start_ts, end_ts):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ts, "endTime": end_ts, "limit": 1000}
    res = requests.get(f"{BASE_URL}/klines", params=params).json()
    if not res or "code" in res: return None
    df = pd.DataFrame(res, columns=["ts", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["ts"] = df["ts"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]: df[col] = df[col].astype(float)
    return df[["ts", "open", "high", "low", "close", "volume"]]

def download_trades_day(symbol, start_ts, end_ts):
    trades = []
    res = requests.get(f"{BASE_URL}/aggTrades", params={"symbol": symbol, "startTime": start_ts, "limit": 1}).json()
    if not res or len(res) == 0: return None
    curr_id = res[0]['a']
    
    # PROVA INSTANTÂNEA: Mostrar os primeiros trades capturados
    first_batch = True
    
    while True:
        try:
            res = requests.get(f"{BASE_URL}/aggTrades", params={"symbol": symbol, "fromId": curr_id, "limit": 1000}).json()
            if not res: break
            
            for t in res:
                ts, p, q = int(t['T']), float(t['p']), float(t['q'])
                if ts > end_ts: return trades
                if p * q >= MIN_VAL_USD:
                    trades.append([ts, p, q, 1 if not t['m'] else -1])
            
            if first_batch and len(trades) > 0:
                print_flush(f"\n✅ PROVA DE DADOS (Primeiras Baleias):")
                temp_df = pd.DataFrame(trades[:5], columns=['ts', 'price', 'qty', 'side'])
                print_flush(temp_df.to_string())
                first_batch = False

            curr_id = res[-1]['a'] + 1
            if res[-1]['T'] > end_ts: break
            time.sleep(0.02)
        except:
            time.sleep(1); continue
    return trades

def main():
    print_flush("\n===============================================================")
    print_flush(f">>> MOTOR LIGADO: DATA MANAGER V21 INSTANT PROOF")
    print_flush(f">>> ATIVO: {SYMBOL} SPOT | FILTRO: ${MIN_VAL_USD}")
    print_flush("===============================================================\n")
    
    os.makedirs("progress_daily", exist_ok=True)
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    current_dt = start_dt

    all_data = []

    while current_dt <= end_dt:
        day_str = current_dt.strftime("%Y-%m-%d")
        file_path = f"progress_daily/data_{day_str}.csv"
        
        if os.path.exists(file_path):
            print_flush(f"✔ Dia {day_str} já existe. Pulando...")
            all_data.append(pd.read_csv(file_path))
            current_dt += timedelta(days=1)
            continue

        print_flush(f"⏳ Processando Dia: {day_str}...")
        s_ts = int(current_dt.timestamp() * 1000)
        e_ts = s_ts + 86399999
        
        df_klines = get_klines(SYMBOL, INTERVAL, s_ts, e_ts)
        raw_trades = download_trades_day(SYMBOL, s_ts, e_ts)
        
        if raw_trades and df_klines is not None:
            trades_np = np.array(raw_trades)
            ms_15m = 15 * 60 * 1000
            ts_target = (trades_np[:, 0] // ms_15m) * ms_15m
            
            df_agg = pd.DataFrame({'ts': ts_target.astype(np.int64), 'cum_delta': trades_np[:, 2] * trades_np[:, 3], 'total_vol_agg': trades_np[:, 2], 'buy_vol_agg': np.where(trades_np[:, 3] == 1, trades_np[:, 2], 0), 'sell_vol_agg': np.where(trades_np[:, 3] == -1, trades_np[:, 2], 0)})
            micro = df_agg.groupby('ts').sum().reset_index()
            df_final = pd.merge(df_klines, micro, on='ts', how='left').fillna(0)
            
            df_final['vpin'] = (df_final['buy_vol_agg'] - df_final['sell_vol_agg']).abs() / (df_final['volume'] + 1e-9)
            df_final['absorcao'] = df_final['total_vol_agg'] / (df_final['volume'] + 1e-9)
            
            # PROVA DE CÁLCULO: Mostrar indicadores da primeira vela do dia
            if not df_final.empty:
                print_flush(f"📊 INDICADORES (Primeira Vela do Dia):")
                print_flush(f"   VPIN: {df_final['vpin'].iloc[0]:.6f} | ABSORÇÃO: {df_final['absorcao'].iloc[0]:.6f} | DELTA: {df_final['cum_delta'].iloc[0]:.2f}")

            df_final.to_csv(file_path, index=False)
            all_data.append(df_final)
            print_flush(f"✅ Dia {day_str} FINALIZADO.\n")
        else:
            print_flush(f"⚠ Dia {day_str} sem dados suficientes.")
            
        current_dt += timedelta(days=1)

    if all_data:
        full_df = pd.concat(all_data).drop_duplicates(subset=['ts']).sort_values('ts')
        full_df.to_csv(f"{SYMBOL}_{INTERVAL}.csv", index=False)
        print_flush(f"\n🚀 PROCESSO CONCLUÍDO! Arquivo final gerado.")
    else:
        print_flush("❌ Erro crítico: Nenhum dado processado.")

if __name__ == "__main__":
    main()
