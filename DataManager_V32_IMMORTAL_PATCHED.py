# ============================================================
# DataManager_V32_FIXED.py
# PENDLEUSDT – Binance Futures aggTrades
# Período: 01/01/2025 00:00:00 → 30/06/2025 23:59:59
# Modo: FAST & FIDELITY (COM ANTI-BLOCK)
# Estrutura PRESERVADA: ts, price, qty, side
# ============================================================

import os
import time
import zipfile
import requests
import pandas as pd
from datetime import datetime

# =========================
# CONFIGURAÇÃO FIXA
# =========================
SYMBOL = "PENDLEUSDT"
BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
LIMIT = 1000

START_DT = datetime(2025, 1, 1, 0, 0, 0)
END_DT   = datetime(2025, 6, 30, 23, 59, 59)

START_MS = int(START_DT.timestamp() * 1000)
END_MS   = int(END_DT.timestamp() * 1000)

OUT_DIR = "./pendle_agg_2025_01_01__2025_06_30"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_PATH = OUT_DIR + ".zip"

os.makedirs(OUT_DIR, exist_ok=True)

# Headers para evitar bloqueio
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive'
}

# =========================
# RETRY COM BACKOFF
# =========================
def make_request_with_retry(session, params, max_retries=5):
    """Faz requisição com retry exponencial e backoff"""
    for attempt in range(max_retries):
        try:
            r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            
            if r.status_code == 200:
                return r
            
            if r.status_code == 418:
                wait_time = min(2 ** attempt * 2, 60)  # Max 60s
                print(f"⚠️  Erro 418 (bloqueio). Aguardando {wait_time}s antes de tentar novamente...")
                time.sleep(wait_time)
                continue
            
            if r.status_code == 429:
                wait_time = min(2 ** attempt * 3, 120)  # Max 120s
                print(f"⚠️  Rate limit (429). Aguardando {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            r.raise_for_status()
            return r
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = min(2 ** attempt, 30)
            print(f"⚠️  Erro de conexão: {e}. Tentando novamente em {wait_time}s...")
            time.sleep(wait_time)
    
    raise RuntimeError("Falha após múltiplas tentativas")

# =========================
# UTIL – PRIMEIRO fromId
# =========================
def get_first_id(symbol, start_ms, session):
    """Obtém o primeiro ID disponível para o período"""
    print(f"🔍 Buscando primeiro ID para {datetime.fromtimestamp(start_ms/1000)}...")
    params = {"symbol": symbol, "startTime": start_ms, "limit": 1}
    
    r = make_request_with_retry(session, params)
    data = r.json()
    
    if isinstance(data, list) and data:
        first_id = int(data[0]["a"])
        print(f"✓ Primeiro ID encontrado: {first_id}")
        return first_id
    
    print("⚠️  Nenhum dado encontrado para o período especificado")
    return None

# =========================
# DOWNLOAD CONTÍNUO (fromId)
# =========================
def download_all_aggtrades(symbol, start_ms, end_ms):
    """Download completo usando fromId para continuidade"""
    rows = []
    request_count = 0
    
    with requests.Session() as session:
        first_id = get_first_id(symbol, start_ms, session)
        if first_id is None:
            raise RuntimeError("Não foi possível obter fromId inicial.")

        curr_id = first_id
        last_ts = 0
        
        print(f"\n📥 Iniciando download de {datetime.fromtimestamp(start_ms/1000)} até {datetime.fromtimestamp(end_ms/1000)}")
        print("=" * 80)

        while True:
            params = {"symbol": symbol, "fromId": curr_id, "limit": LIMIT}
            r = make_request_with_retry(session, params)
            data = r.json()
            
            request_count += 1

            if not isinstance(data, list) or not data:
                print("\n✓ Fim dos dados disponíveis")
                break

            batch_rows = []
            for t in data:
                ts = int(t["T"])
                
                # Se passou do período final, retorna
                if ts > end_ms:
                    if batch_rows:
                        rows.extend(batch_rows)
                    print(f"\n✓ Período final alcançado: {datetime.fromtimestamp(ts/1000)}")
                    return rows

                price = float(t["p"])
                qty   = float(t["q"])
                side  = 1 if t["m"] else 0

                batch_rows.append([ts, price, qty, side])
                last_ts = ts

            rows.extend(batch_rows)
            
            # Progress update a cada 10 requisições
            if request_count % 10 == 0:
                print(f"📊 Requisições: {request_count:,} | Registros: {len(rows):,} | Último timestamp: {datetime.fromtimestamp(last_ts/1000)}")

            curr_id = int(data[-1]["a"]) + 1
            
            # Delay para evitar rate limit
            time.sleep(0.15)  # 150ms entre requisições

    return rows

# =========================
# MAIN
# =========================
def main():
    print("\n" + "=" * 80)
    print("🚀 INICIANDO DOWNLOAD - PENDLEUSDT aggTrades")
    print("=" * 80)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DT} até {END_DT}")
    print(f"Diretório de saída: {OUT_DIR}")
    print("=" * 80 + "\n")

    try:
        rows = download_all_aggtrades(SYMBOL, START_MS, END_MS)
        
        if not rows:
            raise RuntimeError("Nenhum dado retornado.")

        print(f"\n💾 Salvando dados em CSV...")
        df = pd.DataFrame(rows, columns=["ts", "price", "qty", "side"])
        df.to_csv(CSV_PATH, index=False)

        print(f"\n✓ CSV salvo: {CSV_PATH}")
        print(f"✓ Total de registros: {len(df):,}")
        
        # Estatísticas básicas
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"   Primeiro registro: {datetime.fromtimestamp(df['ts'].min()/1000)}")
        print(f"   Último registro: {datetime.fromtimestamp(df['ts'].max()/1000)}")
        print(f"   Preço mínimo: ${df['price'].min():.4f}")
        print(f"   Preço máximo: ${df['price'].max():.4f}")
        print(f"   Volume total: {df['qty'].sum():,.2f}")

        print(f"\n📦 Criando arquivo ZIP...")
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")

        print(f"✓ ZIP pronto: {ZIP_PATH}")
        
        # Tamanho dos arquivos
        csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024)  # MB
        zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)  # MB
        print(f"\n📁 Tamanhos:")
        print(f"   CSV: {csv_size:.2f} MB")
        print(f"   ZIP: {zip_size:.2f} MB")
        print(f"   Compressão: {(1 - zip_size/csv_size)*100:.1f}%")
        
        print("\n" + "=" * 80)
        print("✅ DOWNLOAD FINALIZADO COM SUCESSO!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        raise

if __name__ == "__main__":
    main()