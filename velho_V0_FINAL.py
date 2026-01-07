# ============================================================
# DataManager_V39_FINAL.py
# PENDLEUSDT – Binance Data Vision
# GARANTIDO: Cria ZIP para download do CatBot
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
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =========================
# CONFIGURAÇÃO - EXATAMENTE IGUAL AO ORIGINAL
# =========================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 6, 30, 23, 59, 59)

OUT_DIR = "./pendle_agg_2025_01_01__2025_06_30"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_PATH = OUT_DIR + ".zip"

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# =========================
# FUNÇÕES
# =========================
def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }

def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                time.sleep(min(5 * (2 ** attempt), 60))
            
            response = session.get(url, headers=get_headers(), timeout=90)
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=None)
                        return df
            
            elif response.status_code == 404:
                return None
        except Exception:
            continue
    
    return None

def process_binance_data(df):
    if df is None or df.empty:
        return None
    
    df.columns = [
        'agg_trade_id', 'price', 'quantity',
        'first_trade_id', 'last_trade_id',
        'transact_time', 'is_buyer_maker'
    ]
    
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('Int64'),
        'price': pd.to_numeric(df['price'], errors='coerce'),
        'qty': pd.to_numeric(df['quantity'], errors='coerce'),
        'side': df['is_buyer_maker'].astype(int)
    })
    
    return df_processed.dropna()

# =========================
# SERVIDOR HTTP
# =========================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):

        # ---------- CATBOT ----------
        if self.path == '/catbot':
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header(
                    'Content-Disposition',
                    'attachment; filename="PENDLEUSDT_aggTrades.zip"'
                )
                self.end_headers()
                with open(ZIP_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        # ---------- DOWNLOAD PADRÃO ----------
        if self.path == '/download':
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header(
                    'Content-Disposition',
                    'attachment; filename="PENDLEUSDT_aggTrades.zip"'
                )
                self.end_headers()
                with open(ZIP_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    server.serve_forever()

# =========================
# MAIN
# =========================
def main():
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)

    print(">>> Iniciando download completo...", flush=True)

    dates = generate_date_range(START_DT, END_DT)

    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    session = requests.Session()
    first_write = True

    for date in dates:
        df = download_daily_file(SYMBOL, date, session)
        if df is not None:
            df_p = process_binance_data(df)
            if df_p is not None and not df_p.empty:
                df_p.to_csv(CSV_PATH, mode='a', header=first_write, index=False)
                first_write = False
        time.sleep(random.uniform(0.5, 1.5))

    session.close()

    print(">>> Criando ZIP...", flush=True)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")

    print(">>> ZIP pronto.", flush=True)
    print(">>> CatBot pode baixar em /catbot", flush=True)

    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
