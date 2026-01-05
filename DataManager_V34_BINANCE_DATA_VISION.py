# ============================================================
# DataManager_V34_BINANCE_DATA_VISION.py
# PENDLEUSDT – Binance Historical Data (Data Vision)
# Período: 01/01/2025 → 30/06/2025
# Fonte: https://data.binance.vision (OFICIAL)
# Estrutura: ts, price, qty, side
# ============================================================

import os
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

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

# Base URL dos dados históricos oficiais da Binance
BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

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
    """
    Baixa arquivo diário do Binance Data Vision
    URL: https://data.binance.vision/data/futures/um/daily/aggTrades/SYMBOL/SYMBOL-aggTrades-YYYY-MM-DD.zip
    """
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    print(f"   📥 Baixando: {filename}")
    
    try:
        response = session.get(url, headers=HEADERS, timeout=60)
        
        if response.status_code == 404:
            print(f"   ⚠️  Arquivo não encontrado (404) - pode não ter dados neste dia")
            return None
        
        if response.status_code != 200:
            print(f"   ⚠️  Erro {response.status_code}")
            return None
        
        # Descompacta o arquivo ZIP em memória
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            # Lista arquivos no ZIP
            files = z.namelist()
            if not files:
                print(f"   ⚠️  ZIP vazio")
                return None
            
            # Lê o primeiro arquivo CSV (SEM CABEÇALHO)
            csv_filename = files[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, header=None)
                print(f"   ✓ {len(df):,} registros")
                return df
    
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Erro de rede: {e}")
        return None
    except zipfile.BadZipFile:
        print(f"   ❌ Arquivo ZIP corrompido")
        return None
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def process_binance_data(df):
    """
    Converte dados do formato Binance para nosso formato
    CSV Binance (SEM CABEÇALHO): agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
    Nosso: ts, price, qty, side
    """
    if df is None or df.empty:
        return None
    
    # Os CSVs da Binance NÃO têm cabeçalho, então precisamos nomear as colunas
    df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 
                  'last_trade_id', 'transact_time', 'is_buyer_maker']
    
    # Converte is_buyer_maker para nosso formato de side
    # is_buyer_maker = True → comprador é maker → VENDA AGRESSIVA → side = 1
    # is_buyer_maker = False → comprador é taker → COMPRA AGRESSIVA → side = 0
    df_processed = pd.DataFrame({
        'ts': df['transact_time'].astype(int),
        'price': df['price'].astype(float),
        'qty': df['quantity'].astype(float),
        'side': df['is_buyer_maker'].map({True: 1, 'True': 1, False: 0, 'False': 0}).astype(int)
    })
    
    return df_processed

# =========================
# MAIN
# =========================
def main():
    print("\n" + "=" * 80)
    print("🚀 BINANCE DATA VISION - DOWNLOAD HISTÓRICO")
    print("   Fonte: https://data.binance.vision (OFICIAL)")
    print("=" * 80)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    # Gera lista de datas
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
        
        # Pequeno delay entre downloads
        time.sleep(0.2)
    
    session.close()
    
    print("\n" + "=" * 80)
    print(f"📊 RESUMO DO DOWNLOAD:")
    print(f"   ✓ Dias com dados: {success_count}")
    print(f"   ⚠️  Dias sem dados: {fail_count}")
    print("=" * 80)
    
    if not all_data:
        print("\n❌ ERRO: Nenhum dado foi coletado!")
        print("\nPossíveis causas:")
        print("   1. PENDLEUSDT pode não existir no período especificado (2025)")
        print("   2. Os dados de 2025 ainda não estão disponíveis")
        print("   3. Símbolo incorreto")
        print("\n💡 SUGESTÃO: Verifique os dados disponíveis em:")
        print("   https://data.binance.vision/?prefix=data/futures/um/daily/aggTrades/")
        return
    
    # Concatena todos os dataframes
    print(f"\n💾 Consolidando {len(all_data)} arquivos...")
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicatas e ordena
    print(f"   Registros antes da limpeza: {len(df_final):,}")
    df_final = df_final.drop_duplicates(subset=['ts'], keep='first')
    df_final = df_final.sort_values('ts').reset_index(drop=True)
    print(f"   Registros após limpeza: {len(df_final):,}")
    
    # Filtra pelo período exato
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(END_DT.timestamp() * 1000)
    df_final = df_final[(df_final['ts'] >= start_ms) & (df_final['ts'] <= end_ms)]
    print(f"   Registros no período especificado: {len(df_final):,}")
    
    # Salva CSV
    print(f"\n💾 Salvando CSV...")
    df_final.to_csv(CSV_PATH, index=False)
    print(f"   ✓ {CSV_PATH}")
    
    # Estatísticas
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Primeiro registro: {datetime.fromtimestamp(df_final['ts'].min()/1000)}")
    print(f"   Último registro: {datetime.fromtimestamp(df_final['ts'].max()/1000)}")
    print(f"   Preço mínimo: ${df_final['price'].min():.4f}")
    print(f"   Preço máximo: ${df_final['price'].max():.4f}")
    print(f"   Volume total: {df_final['qty'].sum():,.2f}")
    print(f"   Trades de compra: {(df_final['side'] == 0).sum():,}")
    print(f"   Trades de venda: {(df_final['side'] == 1).sum():,}")
    
    # Cria ZIP
    print(f"\n📦 Criando arquivo ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
    
    csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024)
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    
    print(f"   ✓ {ZIP_PATH}")
    print(f"\n📁 Tamanhos:")
    print(f"   CSV: {csv_size:.2f} MB")
    print(f"   ZIP: {zip_size:.2f} MB")
    print(f"   Compressão: {(1 - zip_size/csv_size)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD FINALIZADO COM SUCESSO!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
