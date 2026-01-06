# ============================================================
# DataManager_FINAL_CATBOX.py
# PENDLEUSDT – Binance Data Vision
# Upload automático para CatBox
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
import random

sys.stdout.reconfigure(line_buffering=True)

SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 6, 30, 23, 59, 59)

OUT_DIR = "./pendle_agg_2025_01_01__2025_06_30"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_PATH = OUT_DIR + ".zip"

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
]

def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                wait = min(5 * (2 ** attempt), 60)
                time.sleep(wait)
            
            response = session.get(url, headers=get_headers(), timeout=90)
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    files = z.namelist()
                    if not files:
                        return None
                    
                    csv_filename = files[0]
                    
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any('transact_time' in str(val) for val in df_test.iloc[0])
                    
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=0 if has_header else None)
                        return df
            
            elif response.status_code == 404:
                return None
            elif response.status_code in [418, 429]:
                continue
            else:
                continue
        
        except Exception:
            if attempt == retry_count - 1:
                return None
            continue
    
    return None

def process_binance_data(df):
    if df is None or df.empty:
        return None
    
    if 'transact_time' not in df.columns:
        df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                      'last_trade_id', 'transact_time', 'is_buyer_maker']
    
    def convert_side(val):
        return 1 if (val is True or val == 'True' or val == 'true') else 0
    
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('Int64'),
        'price': pd.to_numeric(df['price'], errors='coerce').astype(float),
        'qty': pd.to_numeric(df['quantity'], errors='coerce').astype(float),
        'side': df['is_buyer_maker'].apply(convert_side)
    })
    
    return df_processed.dropna()

def upload_catbox(filepath):
    """Faz upload do arquivo para CatBox e retorna o link"""
    print(">>> Enviando para CatBox...", flush=True)
    url = "https://catbox.moe/user/api.php"
    with open(filepath, "rb") as f:
        r = requests.post(
            url,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
            timeout=300
        )
    r.raise_for_status()
    return r.text.strip()

def main():
    print(">>> Iniciando download...", flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    
    session = requests.Session()
    success_count = 0
    first_write = True
    
    for i, date in enumerate(dates, 1):
        print(f"[{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
        
        df = download_daily_file(SYMBOL, date, session, retry_count=5)
        
        if df is not None:
            df_processed = process_binance_data(df)
            
            if df_processed is not None and not df_processed.empty:
                df_processed.to_csv(CSV_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                print(f"✓ {len(df_processed):,}", flush=True)
                del df, df_processed
            else:
                print("⚠️", flush=True)
        else:
            print("⚠️", flush=True)
        
        time.sleep(random.uniform(0.5, 2.0))
    
    session.close()
    
    print(f"\n>>> Download: {success_count}/{total_dates} dias", flush=True)
    
    if success_count == 0:
        print(">>> ERRO: Nenhum dado!", flush=True)
        return
    
    # Cria ZIP
    print(">>> Criando ZIP...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
    
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f">>> ZIP: {zip_size:.2f} MB", flush=True)
    
    # Upload para CatBox e mostra link
    try:
        link = upload_catbox(ZIP_PATH)
        print("=" * 80, flush=True)
        print(">>> ZIP pronto para download:", flush=True)
        print(link, flush=True)
        print("=" * 80, flush=True)
        print(">>> FINALIZADO.", flush=True)
    except Exception as e:
        print(f">>> ERRO ao enviar: {e}", flush=True)
        return
    
    # Mantém vivo
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()