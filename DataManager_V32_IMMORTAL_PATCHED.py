# ============================================================
# DataManager_V32_IMMORTAL_PATCHED.py
# PENDLEUSDT — Binance Futures aggTrades
# Período: 01/01/2025 00:00:00 → 30/06/2025 23:59:59
# Modo: FAST & FIDELITY (SEM CHUNK / SEM SKIP / SEM LOOP)
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

# =========================
# UTIL — PRIMEIRO fromId
# =========================
def get_first_id(symbol, start_ms, session):
    params = {"symbol": symbol, "startTime": start_ms, "limit": 1}
    r = session.get(BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return int(data[0]["a"])
    return None

# =========================
# DOWNLOAD CONTÍNUO (fromId)
# =========================
def download_all_aggtrades(symbol, start_ms, end_ms):
    rows = []
    with requests.Session() as session:
        first_id = get_first_id(symbol, start_ms, session)
        if first_id is None:
            raise RuntimeError("Não foi possível obter fromId inicial.")

        curr_id = first_id

        while True:
            params = {"symbol": symbol, "fromId": curr_id, "limit": LIMIT}
            r = session.get(BASE_URL, params=params, timeout=20)

            if r.status_code == 429:
                time.sleep(1)
                continue

            r.raise_for_status()
            data = r.json()

            if not isinstance(data, list) or not data:
                break

            for t in data:
                ts = int(t["T"])
                if ts > end_ms:
                    return rows

                price = float(t["p"])
                qty   = float(t["q"])
                side  = 1 if t["m"] else 0  # mesma lógica de agressão

                rows.append([ts, price, qty, side])

            curr_id = int(data[-1]["a"]) + 1
            time.sleep(0.01)  # mínimo para evitar 429

    return rows

# =========================
# MAIN
# =========================
def main():
    print(">>> Iniciando download completo (SEM CHUNK / SEM SKIP)...")

    rows = download_all_aggtrades(SYMBOL, START_MS, END_MS)
    if not rows:
        raise RuntimeError("Nenhum dado retornado.")

    df = pd.DataFrame(rows, columns=["ts", "price", "qty", "side"])
    df.to_csv(CSV_PATH, index=False)

    print(f">>> CSV salvo: {CSV_PATH}")
    print(f">>> Total de registros: {len(df):,}")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CSV_PATH, arcname="PENDLEUSDT_aggTrades.csv")

    print(f">>> ZIP pronto para download: {ZIP_PATH}")
    print(">>> FINALIZADO.")

if __name__ == "__main__":
    main()
