# ============================================================
# DataManager_V40_WITH_DOWNLOAD_LINK.py
# CRIA LINK HTTP PARA DOWNLOAD DO ZIP
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
# CONFIGURAÇÃO
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
# SERVIDOR HTTP
# =========================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download' or self.path == '/download/':
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="PENDLEUSDT_aggTrades.zip"')
                self.end_headers()
                
                with open(ZIP_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Arquivo nao encontrado')
        else:
            # Página inicial
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = f"""
            <html>
            <head><title>Download PENDLEUSDT</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>PENDLEUSDT Data Download</h1>
                <p>Status: {'PRONTO!' if os.path.exists(ZIP_PATH) else 'Processando...'}</p>
                {'<a href="/download" style="font-size: 24px; padding: 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">📥 BAIXAR ZIP</a>' if os.path.exists(ZIP_PATH) else '<p>Aguarde o processamento...</p>'}
            </body>
            </html>
            """
            self.wfile.write(html.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    print(f">>> Servidor HTTP iniciado na porta {port}", flush=True)
    server.serve_forever()

# =========================
# FUNÇÕES DE DOWNLOAD
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

# =========================
# MAIN
# =========================
def process_data():
    """Função que processa os dados"""
    print(">>> Iniciando download completo...", flush=True)
    print(f">>> Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}", flush=True)
    print("=" * 80, flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    
    print(f">>> Total de dias: {total_dates}", flush=True)
    
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
    
    print(f"\n>>> Download concluído: {success_count}/{total_dates} dias", flush=True)
    
    if success_count == 0:
        print(">>> ERRO: Nenhum dado coletado!", flush=True)
        return
    
    # CRIA ZIP IMEDIATAMENTE
    print(">>> Criando ZIP...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
    
    if os.path.exists(ZIP_PATH):
        zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        print(f">>> ZIP CRIADO: {zip_size:.2f} MB", flush=True)
        print("=" * 80, flush=True)
        print(">>> ✅ DOWNLOAD DISPONÍVEL!", flush=True)
        print(">>> Acesse: https://SEU-APP.onrender.com/download", flush=True)
        print("=" * 80, flush=True)

def main():
    # Inicia servidor HTTP em thread separada
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    time.sleep(2)  # Aguarda servidor iniciar
    
    # Processa dados
    process_data()
    
    # Mantém servidor vivo
    print(">>> Servidor HTTP mantido ativo...", flush=True)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
