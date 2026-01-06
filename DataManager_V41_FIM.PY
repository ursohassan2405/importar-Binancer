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

# MESMO caminho do script original!
OUT_DIR = "./pendle_agg_2025_01_01__2025_06_30"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_PATH = OUT_DIR + ".zip"  # Cria: ./pendle_agg_2025_01_01__2025_06_30.zip

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
# 15M TRATADO (SEM ESTOURAR RAM)
# =========================
def gerar_15m_tratado_incremental(csv_agg_path, csv_15m_path, timeframe_min=15, min_val_usd=500, chunksize=200_000):
    """
    Converte o CSV de aggTrades (ts,price,qty,side) para um dataset 15m:
    colunas: ts, open, high, low, close, volume, buy_vol, sell_vol, delta

    Regras:
    - bucket = floor(datetime UTC, 15min)
    - OHLCV do preço/qty
    - baleias: val_usd = price*qty >= min_val_usd
      buy_vol: soma qty das baleias com side == 0
      sell_vol: soma qty das baleias com side == 1
      delta = buy_vol - sell_vol
    - SEM carregar o arquivo inteiro em memória (processa em chunks)
    """

    print(">>> Gerando dataset 15m tratado...", flush=True)

    # Estrutura incremental por bucket
    buckets = {}  # key: bucket_start_ms -> dict(o,h,l,c,vol,buy,sell)

    # Lê em chunks
    for chunk in pd.read_csv(csv_agg_path, chunksize=chunksize):
        # Garante tipos
        chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")
        chunk["side"] = pd.to_numeric(chunk["side"], errors="coerce")

        chunk = chunk.dropna(subset=["ts", "price", "qty", "side"])
        if chunk.empty:
            continue

        # Bucket UTC 15m
        dt = pd.to_datetime(chunk["ts"].astype("int64"), unit="ms", utc=True)
        bucket_dt = dt.dt.floor(f"{timeframe_min}min")
        bucket_ms = (bucket_dt.astype("int64") // 10**6).astype("int64")
        chunk = chunk.assign(bucket_ms=bucket_ms)

        # Whale flag
        val_usd = chunk["price"] * chunk["qty"]
        is_whale = val_usd >= float(min_val_usd)

        # Itera em linhas (baixo RAM, mais CPU; mas no Render não mata)
        # Obs: mantém open como primeiro preço visto no bucket e close como último.
        for ts_ms, price, qty, side, bms, whale in zip(
            chunk["ts"].astype("int64"),
            chunk["price"].astype(float),
            chunk["qty"].astype(float),
            chunk["side"].astype(int),
            chunk["bucket_ms"].astype("int64"),
            is_whale.astype(bool),
        ):
            st = buckets.get(bms)
            if st is None:
                st = {
                    "ts": int(bms),
                    "open": float(price),
                    "high": float(price),
                    "low": float(price),
                    "close": float(price),
                    "volume": 0.0,
                    "buy_vol": 0.0,
                    "sell_vol": 0.0,
                }
                buckets[bms] = st
            else:
                # OHLC
                if price > st["high"]:
                    st["high"] = float(price)
                if price < st["low"]:
                    st["low"] = float(price)
                st["close"] = float(price)

            st["volume"] += float(qty)

            # Baleias
            if whale:
                if side == 0:
                    st["buy_vol"] += float(qty)
                else:
                    st["sell_vol"] += float(qty)

    if not buckets:
        raise RuntimeError("Nenhum bucket 15m gerado (buckets vazio).")

    # Ordena por tempo e salva
    rows = []
    for bms in sorted(buckets.keys()):
        st = buckets[bms]
        rows.append([
            st["ts"],
            st["open"],
            st["high"],
            st["low"],
            st["close"],
            st["volume"],
            st["buy_vol"],
            st["sell_vol"],
            st["buy_vol"] - st["sell_vol"],
        ])
    df_15m = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close",
        "volume", "buy_vol", "sell_vol", "delta"
    ])

    # ============================
    # CORREÇÃO + SANEAMENTO (PADRÃO ELITE)
    # ============================

    df_15m["taker_buy_base"]  = df_15m["buy_vol"]
    df_15m["taker_sell_base"] = df_15m["sell_vol"]
    df_15m["total_vol_agg"]   = df_15m["buy_vol"] - df_15m["sell_vol"]

    df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
    df_15m["trades"] = 0
    df_15m["close_time"] = df_15m["ts"] + (timeframe_min * 60 * 1000) - 1

    # saneamento numérico
    for c in [
        "open","high","low","close",
        "volume","quote_volume",
        "taker_buy_base","taker_sell_base",
        "total_vol_agg","delta"
    ]:
        df_15m[c] = pd.to_numeric(df_15m[c], errors="coerce").fillna(0.0)

    df_15m["ts"] = pd.to_numeric(df_15m["ts"], errors="coerce").fillna(0).astype("int64")

    df_15m = df_15m.sort_values("ts").reset_index(drop=True)

    df_15m.to_csv(csv_15m_path, index=False)

    print(f">>> 15m tratado gerado: {csv_15m_path}", flush=True)

# ============================================================
# SAÍDA FINAL — CONTRATO V1 + AGRESSÃO 1m REAL
# ============================================================

df_15m["taker_buy_base"]  = df_15m.get("buy_vol_agg", 0.0)
df_15m["taker_sell_base"] = df_15m.get("sell_vol_agg", 0.0)
df_15m["total_vol_agg"]   = df_15m["taker_buy_base"] + df_15m["taker_sell_base"]

df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
df_15m["trades"]       = df_15m.get("trades", 0)
df_15m["close_time"]   = df_15m["ts"] + (15 * 60 * 1000) - 1

assert_schema_v1(df_15m)

COLUNAS_V1 = [
    "ts",
    "open","high","low","close",
    "volume","quote_volume","trades",
    "taker_buy_base","taker_sell_base",
    "total_vol_agg",
    "delta","cum_delta"
]

df_15m[COLUNAS_V1].to_csv(CSV_PATH, index=False)

print(f">>> CSV 15m institucional salvo: {CSV_PATH}", flush=True)


# ============================================================
# SAÍDA FINAL — CONTRATO V1 + AGRESSÃO 1m REAL
# ============================================================

# Aliases (conteúdo novo, nomes do V1)
df_15m["taker_buy_base"]  = df_15m.get("buy_vol_agg", 0.0)
df_15m["taker_sell_base"] = df_15m.get("sell_vol_agg", 0.0)
df_15m["total_vol_agg"]   = df_15m["taker_buy_base"] + df_15m["taker_sell_base"]

# Campos do V1
df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
df_15m["trades"]       = df_15m.get("trades", 0)
df_15m["close_time"]   = df_15m["ts"] + (15 * 60 * 1000) - 1

assert_schema_v1(df_15m)

COLUNAS_V1 = [
    "ts",
    "open","high","low","close",
    "volume","quote_volume","trades",
    "taker_buy_base","taker_sell_base",
    "total_vol_agg",
    "delta","cum_delta"
]

df_15m[COLUNAS_V1].to_csv(CSV_PATH, index=False)

print(f">>> CSV 15m institucional salvo: {CSV_PATH}", flush=True)



# =========================
# SERVIDOR HTTP PARA DOWNLOAD
# =========================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download' or self.path == '/download/':
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', 'attachment; filename="PENDLEUSDT_aggTrades.zip"')
                self.end_headers()
                with open(ZIP_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP ainda nao criado')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            status = 'PRONTO!' if os.path.exists(ZIP_PATH) else 'Processando...'
            link = '<a href="/download" style="font-size:24px;padding:20px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;">BAIXAR ZIP</a>' if os.path.exists(ZIP_PATH) else '<p>Aguarde...</p>'
            html = f'<html><body style="font-family:Arial;padding:50px;text-align:center;"><h1>PENDLEUSDT Download</h1><p>{status}</p>{link}</body></html>'
            self.wfile.write(html.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    print(f">>> Servidor HTTP na porta {port}", flush=True)
    server.serve_forever()

# =========================
# MAIN
# =========================
def main():
    # Inicia servidor HTTP em background
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    print(">>> Iniciando download completo...", flush=True)
    print(f">>> Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}", flush=True)
    print(f">>> Destino: {ZIP_PATH}", flush=True)
    print("=" * 80, flush=True)
    
    dates = generate_date_range(START_DT, END_DT)
    total_dates = len(dates)
    
    print(f">>> Total de dias: {total_dates}", flush=True)
    print("=" * 80, flush=True)
    
    # Limpa arquivo anterior
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
    
    session = requests.Session()
    success_count = 0
    first_write = True
    
    # DOWNLOAD INCREMENTAL (baixa RAM)
    for i, date in enumerate(dates, 1):
        print(f"[{i}/{total_dates}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
        
        df = download_daily_file(SYMBOL, date, session, retry_count=5)
        
        if df is not None:
            df_processed = process_binance_data(df)
            
            if df_processed is not None and not df_processed.empty:
                # SALVA IMEDIATAMENTE - não acumula na RAM
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
    
    print("\n" + "=" * 80, flush=True)
    print(f">>> Download concluído: {success_count}/{total_dates} dias", flush=True)
    print("=" * 80, flush=True)
    
    if success_count == 0:
        print(">>> ERRO: Nenhum dado coletado!", flush=True)
        return
    
    # ============================================================
    # GERA DATASET 15M TRATADO E CRIA ZIP (ÚNICO DOWNLOAD)
    # ============================================================

    CSV_15M_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_15m.csv")
    if os.path.exists(CSV_15M_PATH):
        os.remove(CSV_15M_PATH)

    # Geração 15m SEM estourar RAM (incremental)
    gerar_15m_tratado_incremental(
        CSV_PATH,
        CSV_15M_PATH,
        timeframe_min=15,
        min_val_usd=500,
        chunksize=200_000
    )

    print(">>> Criando ZIP do dataset 15m tratado...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_15M_PATH, arcname="PENDLEUSDT_15m.csv")

    if os.path.exists(ZIP_PATH):
        zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
        print(f">>> ZIP CRIADO: {ZIP_PATH} ({zip_size:.2f} MB)", flush=True)
        print(">>> ✅ ZIP 15m PRONTO PARA DOWNLOAD!", flush=True)

    print(">>> Enviando ZIP para CatBox...", flush=True)
    try:
        link = upload_catbox(ZIP_PATH)
        print(">>> LINK FINAL PARA DOWNLOAD:", flush=True)
        print(link, flush=True)
    except Exception as e:
        print(f">>> ERRO AO ENVIAR PARA CATBOX: {e}", flush=True)

        print(">>> Serviço mantido ativo para download...", flush=True)
        while True:
            time.sleep(3600)


    else:
        print(">>> ERRO: ZIP 15m não foi criado!", flush=True)

    return
    
    # PROCESSAMENTO FINAL (opcional - já temos o ZIP)
    # Se estourar memória aqui, tudo bem - o ZIP já foi criado!
    try:
        print(">>> Processando arquivo final (limpeza)...", flush=True)
        
        # Lê em chunks MUITO MENORES para economizar RAM
        chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(CSV_PATH, chunksize=50000):  # 50k em vez de 100k
            chunks.append(chunk)
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"    Lendo chunk {chunk_count}...", flush=True)
    
        df_final = pd.concat(chunks, ignore_index=True)
        del chunks
        
        print(f">>> Total de registros: {len(df_final):,}", flush=True)
        
        # Limpa duplicatas
        df_final = df_final.drop_duplicates(subset=['ts'], keep='first')
        df_final = df_final.sort_values('ts').reset_index(drop=True)
        
        # Filtra período exato
        start_ms = int(START_DT.timestamp() * 1000)
        end_ms = int(END_DT.timestamp() * 1000)
        df_final = df_final[(df_final['ts'] >= start_ms) & (df_final['ts'] <= end_ms)]
        
        # ============================================================
        # GERAÇÃO FINAL 15M — MESMA LÓGICA DO SCRIPT PRINCIPAL (ELITE)
        # ============================================================

        # df_1m JÁ EXISTE NO SCRIPT (klines 1m carregados)
        # trades_np JÁ EXISTE NO SCRIPT (aggTrades processados)

        # 1) Normalização do 1m (como no ELITE)
        df_1m = normalize_klines(df_1m)

        # 2) Microestrutura real calculada no 1m
        df_1m = processar_microestrutura_dinamica(
            df_1m,
            trades_np,
            timeframe="1m"
        )


        # ============================================================
        # COMPATIBILIDADE TOTAL COM V1 (MESMO CONTRATO DE SAÍDA)
        # ============================================================

        # Aliases de agressão (conteúdo NOVO, nome LEGADO)
        if "buy_vol_agg" in df_15m.columns:
            df_15m["taker_buy_base"] = df_15m["buy_vol_agg"]
        else:
            df_15m["taker_buy_base"] = 0.0

        if "sell_vol_agg" in df_15m.columns:
            df_15m["taker_sell_base"] = df_15m["sell_vol_agg"]
        else:
            df_15m["taker_sell_base"] = 0.0

        # Volume total agressivo
        df_15m["total_vol_agg"] = df_15m["taker_buy_base"] + df_15m["taker_sell_base"]

        # Quote volume (igual V1)
        df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]

        # Trades (se não existir, mantém compatível)
        if "trades" not in df_15m.columns:
            df_15m["trades"] = 0

        # Close time (padrão Binance)
        df_15m["close_time"] = df_15m["ts"] + (15 * 60 * 1000) - 1

        # 3) Resample institucional 1m → 15m (preserva TODAS as colunas)
        df_15m = smart_resample(df_1m, "15m")

        # 4) Salva CSV final 15m
        df_15m.to_csv(CSV_PATH, index=False)
        print(f">>> CSV 15m institucional salvo: {CSV_PATH}", flush=True)
        
        del df_final
        
        # Recria ZIP com dados limpos
        print(">>> Atualizando ZIP com dados limpos...", flush=True)
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")
        
        print(">>> ZIP atualizado com dados limpos!", flush=True)
        
    except MemoryError:
        print(">>> AVISO: Memória insuficiente para processar dados.", flush=True)
        print(">>> Mas não tem problema - o ZIP já foi criado com os dados brutos!", flush=True)
    except Exception as e:
        print(f">>> AVISO: Erro no processamento final: {e}", flush=True)
        print(">>> Mas não tem problema - o ZIP já foi criado!", flush=True)
    
    print(">>> FINALIZADO.", flush=True)
    print("=" * 80, flush=True)
    print(">>> 🔗 LINK PARA DOWNLOAD:", flush=True)
    print(">>> https://SEU-APP.onrender.com/download", flush=True)
    print("=" * 80, flush=True)
    
    # MANTÉM VIVO (não reinicia)
    print(">>> Serviço mantido ativo para download...", flush=True)
    while True:
        time.sleep(3600)

import requests

def upload_catbox(filepath):
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


if __name__ == "__main__":
    main()
