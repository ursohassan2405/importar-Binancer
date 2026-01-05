# ============================================================
# DataManager_V35_WEB_SERVICE.py
# PENDLEUSDT – Binance Historical Data (Data Vision)
# Período: 01/01/2025 → 30/06/2025
# Fonte: https://data.binance.vision (OFICIAL)
# Modo: WEB SERVICE - acesse /download para iniciar
# ============================================================

import os
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from flask import Flask, send_file, jsonify

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

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Flag para saber se já está processando
processing = False
download_ready = False

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# FUNÇÕES AUXILIARES
# =========================
def generate_date_range(start_dt, end_dt):
    """Gera lista de datas entre start e end"""
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def download_daily_file(symbol, date, session):
    """Baixa arquivo diário do Binance Data Vision"""
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    print(f"   📥 Baixando: {filename}")
    
    try:
        response = session.get(url, headers=HEADERS, timeout=60)
        
        if response.status_code == 404:
            print(f"   ⚠️  Arquivo não encontrado (404)")
            return None
        
        if response.status_code != 200:
            print(f"   ⚠️  Erro {response.status_code}")
            return None
        
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            files = z.namelist()
            if not files:
                print(f"   ⚠️  ZIP vazio")
                return None
            
            csv_filename = files[0]
            with z.open(csv_filename) as f:
                # Detecta se tem cabeçalho
                df = pd.read_csv(f, header=None, nrows=1)
                first_row = df.iloc[0]
                has_header = any('transact_time' in str(val) for val in first_row)
                
            # Lê novamente com a configuração correta
            with z.open(csv_filename) as f:
                if has_header:
                    df = pd.read_csv(f, header=0)
                else:
                    df = pd.read_csv(f, header=None)
                
                print(f"   ✓ {len(df):,} registros")
                return df
    
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def process_binance_data(df):
    """Converte dados do formato Binance para nosso formato"""
    if df is None or df.empty:
        return None
    
    # Verifica se já tem os nomes de colunas corretos
    if 'transact_time' not in df.columns:
        df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                      'last_trade_id', 'transact_time', 'is_buyer_maker']
    
    def convert_side(val):
        if val is True or val == 'True' or val == 'true':
            return 1
        else:
            return 0
    
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('Int64'),
        'price': pd.to_numeric(df['price'], errors='coerce').astype(float),
        'qty': pd.to_numeric(df['quantity'], errors='coerce').astype(float),
        'side': df['is_buyer_maker'].apply(convert_side)
    })
    
    df_processed = df_processed.dropna()
    return df_processed

def process_download():
    """Executa o download completo"""
    global processing, download_ready
    
    print("\n" + "=" * 80)
    print("🚀 BINANCE DATA VISION - DOWNLOAD HISTÓRICO")
    print("=" * 80)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    dates = generate_date_range(START_DT, END_DT)
    print(f"\n📅 Total de dias a processar: {len(dates)}")
    print("=" * 80)
    
    all_data = []
    success_count = 0
    fail_count = 0
    
    session = requests.Session()
    
    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] {date.strftime('%Y-%m-%d')}")
        
        df = download_daily_file(SYMBOL, date, session)
        
        if df is not None:
            df_processed = process_binance_data(df)
            if df_processed is not None:
                all_data.append(df_processed)
                success_count += 1
        else:
            fail_count += 1
        
        time.sleep(0.2)
    
    session.close()
    
    print("\n" + "=" * 80)
    print(f"📊 RESUMO: ✓ {success_count} dias | ⚠️  {fail_count} dias sem dados")
    print("=" * 80)
    
    if not all_data:
        print("\n❌ Nenhum dado coletado!")
        processing = False
        return False
    
    print(f"\n💾 Consolidando {len(all_data)} arquivos...")
    df_final = pd.concat(all_data, ignore_index=True)
    
    print(f"   Registros antes da limpeza: {len(df_final):,}")
    df_final = df_final.drop_duplicates(subset=['ts'], keep='first')
    df_final = df_final.sort_values('ts').reset_index(drop=True)
    print(f"   Registros após limpeza: {len(df_final):,}")
    
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(END_DT.timestamp() * 1000)
    df_final = df_final[(df_final['ts'] >= start_ms) & (df_final['ts'] <= end_ms)]
    print(f"   Registros no período: {len(df_final):,}")
    
    print(f"\n💾 Salvando CSV...")
    df_final.to_csv(CSV_PATH, index=False)
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Primeiro: {datetime.fromtimestamp(df_final['ts'].min()/1000)}")
    print(f"   Último: {datetime.fromtimestamp(df_final['ts'].max()/1000)}")
    print(f"   Preço min: ${df_final['price'].min():.4f}")
    print(f"   Preço max: ${df_final['price'].max():.4f}")
    print(f"   Volume total: {df_final['qty'].sum():,.2f}")
    
    print(f"\n📦 Criando ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
    
    csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024)
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    
    print(f"\n📁 CSV: {csv_size:.2f} MB | ZIP: {zip_size:.2f} MB")
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD FINALIZADO!")
    print("=" * 80 + "\n")
    
    processing = False
    download_ready = True
    return True

# =========================
# ROTAS WEB
# =========================
@app.route('/')
def home():
    """Página inicial"""
    return '''
    <html>
    <head><title>Binance Data Downloader</title></head>
    <body style="font-family: Arial; padding: 50px; text-align: center;">
        <h1>🚀 Binance PENDLEUSDT Data Downloader</h1>
        <p>Período: 01/01/2025 até 30/06/2025</p>
        <hr>
        <h3>Endpoints disponíveis:</h3>
        <ul style="list-style: none;">
            <li><a href="/start" style="font-size: 20px;">➡️ /start</a> - Inicia o download</li>
            <li><a href="/status" style="font-size: 20px;">➡️ /status</a> - Verifica status</li>
            <li><a href="/download" style="font-size: 20px;">➡️ /download</a> - Baixa o ZIP (após processar)</li>
        </ul>
    </body>
    </html>
    '''

@app.route('/start')
def start_download():
    """Inicia o processamento"""
    global processing
    
    if processing:
        return jsonify({"status": "error", "message": "Já está processando!"}), 400
    
    if download_ready and os.path.exists(ZIP_PATH):
        return jsonify({"status": "ready", "message": "Download já está pronto! Use /download"}), 200
    
    processing = True
    
    # Processa em background (em produção, use Celery ou similar)
    import threading
    thread = threading.Thread(target=process_download)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Download iniciado! Use /status para acompanhar."
    }), 200

@app.route('/status')
def check_status():
    """Verifica o status do processamento"""
    if processing:
        return jsonify({
            "status": "processing",
            "message": "Download em andamento... aguarde."
        }), 200
    
    if download_ready and os.path.exists(ZIP_PATH):
        zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        return jsonify({
            "status": "ready",
            "message": "Download pronto!",
            "file_size_mb": round(zip_size, 2),
            "download_url": "/download"
        }), 200
    
    return jsonify({
        "status": "idle",
        "message": "Nenhum download em andamento. Use /start para iniciar."
    }), 200

@app.route('/download')
def download_file():
    """Faz o download do arquivo ZIP"""
    if not os.path.exists(ZIP_PATH):
        return jsonify({
            "status": "error",
            "message": "Arquivo não encontrado. Use /start para processar primeiro."
        }), 404
    
    return send_file(
        ZIP_PATH,
        mimetype='application/zip',
        as_attachment=True,
        download_name='PENDLEUSDT_aggTrades_2025_H1.zip'
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
