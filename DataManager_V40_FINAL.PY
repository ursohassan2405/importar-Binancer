# ============================================================
# DataManager_V40_FINAL.py
# Coletor aggTrades + Agregador 15m + Download único (Render)
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
CSV_15M_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_15m.csv")
ZIP_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_15m.zip")

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
]

# =========================
# FUNÇÕES AUXILIARES
# =========================
def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}


def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"

    for attempt in range(retry_count):
        try:
            response = session.get(url, headers=get_headers(), timeout=60)
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any("transact_time" in str(v) for v in df_test.iloc[0])
                    with z.open(csv_filename) as f:
                        return pd.read_csv(f, header=0 if has_header else None)
            elif response.status_code == 404:
                return None
        except Exception:
            time.sleep(2)
    return None


def process_binance_data(df):
    if df is None or df.empty:
        return None

    if "transact_time" not in df.columns:
        df.columns = [
            "agg_trade_id", "price", "quantity",
            "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"
        ]

    def convert_side(v):
        return 1 if (v is True or str(v).lower() == "true") else 0

    out = pd.DataFrame({
        "ts": pd.to_numeric(df["transact_time"], errors="coerce"),
        "price": pd.to_numeric(df["price"], errors="coerce"),
        "qty": pd.to_numeric(df["quantity"], errors="coerce"),
        "side": df["is_buyer_maker"].apply(convert_side),
    })

    return out.dropna()


# =========================
# AGREGADOR 15M (OBRIGATÓRIO)
# =========================
def gerar_15m_tratado(csv_agg_path, csv_15m_path):
    TIMEFRAME_MIN = 15
    MIN_VAL_USD = 500

    print(">>> Gerando dataset 15m tratado...", flush=True)

    df = pd.read_csv(csv_agg_path)

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["bucket"] = df["datetime"].dt.floor(f"{TIMEFRAME_MIN}T")

    ohlc = (
        df.groupby("bucket")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("qty", "sum"),
        )
        .reset_index()
    )

    df["val_usd"] = df["price"] * df["qty"]
    baleias = df[df["val_usd"] >= MIN_VAL_USD]

    buy_flow = baleias[baleias["side"] == 0].groupby("bucket")["qty"].sum()
    sell_flow = baleias[baleias["side"] == 1].groupby("bucket")["qty"].sum()

    ohlc["buy_vol"] = ohlc["bucket"].map(buy_flow).fillna(0.0)
    ohlc["sell_vol"] = ohlc["bucket"].map(sell_flow).fillna(0.0)
    ohlc["delta"] = ohlc["buy_vol"] - ohlc["sell_vol"]

    ohlc["ts"] = (ohlc["bucket"].astype("int64") // 10**6)

    ohlc[
        ["ts", "open", "high", "low", "close", "volume",
         "buy_vol", "sell_vol", "delta"]
    ].to_csv(csv_15m_path, index=False)

    print(f">>> 15m tratado gerado: {csv_15m_path}", flush=True)


# =========================
# HTTP SERVER (DOWNLOAD)
# =========================
class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if not os.path.exists(ZIP_PATH):
            self.send_response(404)
            self.end_headers()
            return

        size = os.path.getsize(ZIP_PATH)

        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header(
            "Content-Disposition",
            'attachment; filename="PENDLEUSDT_15m.zip"'
        )
        self.send_header("Content-Length", str(size))
        self.end_headers()

        with open(ZIP_PATH, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)


def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), DownloadHandler)
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
    success = 0

    for d in dates:
        df = download_daily_file(SYMBOL, d, session)
        if df is not None:
            out = process_binance_data(df)
            if out is not None and not out.empty:
                out.to_csv(CSV_PATH, mode="a", header=first_write, index=False)
                first_write = False
                success += 1
        time.sleep(1)

    session.close()

    if success == 0:
        print(">>> ERRO: nenhum dado coletado", flush=True)
        return

    gerar_15m_tratado(CSV_PATH, CSV_15M_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_15M_PATH, arcname="PENDLEUSDT_15m.csv")

    print(">>> ZIP FINAL 15M CRIADO", flush=True)
    print(">>> Serviço mantido ativo...", flush=True)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
