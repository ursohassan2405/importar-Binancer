# ============================================================
# DataManager_V38_ANTI_BLOCK.py
# PENDLEUSDT – Binance Data Vision (ANTI-BLOQUEIO)
# Salva em /opt/render/project/src/output
# Após terminar, arquivos ficam disponíveis por SSH/SFTP
# ============================================================

import os
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
import random

# =========================
# CONFIGURAÇÃO
# =========================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT = datetime(2025, 6, 30, 23, 59, 59)

OUT_DIR = "/opt/render/project/src/output"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.zip")

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# Lista de User Agents para rotacionar
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
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
    """Retorna headers com User-Agent aleatório"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/zip,*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'DNT': '1',
    }

def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    for attempt in range(retry_count):
        try:
            # Delay progressivo entre tentativas
            if attempt > 0:
                wait = min(5 * (2 ** attempt), 60)
                print(f"      Tentativa {attempt + 1}/{retry_count} após {wait}s...")
                time.sleep(wait)
            
            response = session.get(url, headers=get_headers(), timeout=90)
            
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    files = z.namelist()
                    if not files:
                        return None
                    
                    csv_filename = files[0]
                    
                    # Detecta cabeçalho
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any('transact_time' in str(val) for val in df_test.iloc[0])
                    
                    # Lê dados
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f, header=0 if has_header else None)
                        return df
            
            elif response.status_code == 404:
                return None  # Arquivo não existe, pula
            
            elif response.status_code in [418, 429]:
                # Bloqueio - aguarda mais tempo
                continue
            
            else:
                print(f"      Status {response.status_code}")
                continue
        
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"      Erro após {retry_count} tentativas: {e}")
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
def main():
    print("\n" + "=" * 80)
    print("🚀 BINANCE DATA DOWNLOAD - ANTI-BLOCK VERSION")
    print("=" * 80)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} → {END_DT.strftime('%Y-%m-%d')}")
    print(f"Destino: {OUT_DIR}")
    print("=" * 80)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    
    print(f"\n📅 Total: {total_dates} dias")
    print("=" * 80)
    
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    
    session = requests.Session()
    success_count = 0
    fail_count = 0
    first_write = True
    
    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ")
        
        df = download_daily_file(SYMBOL, date, session, retry_count=5)
        
        if df is not None:
            df_processed = process_binance_data(df)
            
            if df_processed is not None and not df_processed.empty:
                df_processed.to_csv(CSV_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                print(f"✓ {len(df_processed):,}")
                del df, df_processed
            else:
                print("⚠️  Sem dados válidos")
                fail_count += 1
        else:
            print("⚠️  Não encontrado")
            fail_count += 1
        
        # Delay aleatório entre 0.5s e 2s
        time.sleep(random.uniform(0.5, 2.0))
    
    session.close()
    
    print("\n" + "=" * 80)
    print(f"✓ {success_count} dias | ⚠️  {fail_count} dias sem dados")
    print("=" * 80)
    
    if success_count == 0:
        print("\n❌ Nenhum dado coletado! IP pode estar bloqueado.")
        return
    
    print(f"\n🔧 Processando arquivo final...")
    
    chunks = []
    for chunk in pd.read_csv(CSV_PATH, chunksize=100000):
        chunks.append(chunk)
    
    df_final = pd.concat(chunks, ignore_index=True)
    del chunks
    
    print(f"   Total: {len(df_final):,}")
    df_final = df_final.drop_duplicates(subset=['ts'], keep='first')
    print(f"   Sem duplicatas: {len(df_final):,}")
    df_final = df_final.sort_values('ts').reset_index(drop=True)
    
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(END_DT.timestamp() * 1000)
    df_final = df_final[(df_final['ts'] >= start_ms) & (df_final['ts'] <= end_ms)]
    print(f"   No período: {len(df_final):,}")
    
    df_final.to_csv(CSV_PATH, index=False)
    
    if not df_final.empty:
        print(f"\n📈 STATS:")
        print(f"   {datetime.fromtimestamp(df_final['ts'].min()/1000)} → {datetime.fromtimestamp(df_final['ts'].max()/1000)}")
        print(f"   ${df_final['price'].min():.4f} - ${df_final['price'].max():.4f}")
        print(f"   Volume: {df_final['qty'].sum():,.2f}")
    
    del df_final
    
    print(f"\n📦 Criando ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
    
    csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024)
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    
    print(f"\n📁 ARQUIVOS:")
    print(f"   {CSV_PATH} ({csv_size:.2f} MB)")
    print(f"   {ZIP_PATH} ({zip_size:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("✅ FINALIZADO!")
    print("=" * 80)
    print(f"\n💾 Arquivos salvos em: {OUT_DIR}")
    print("   Acesse via SFTP ou configure storage persistente no Render\n")

if __name__ == "__main__":
    main()
