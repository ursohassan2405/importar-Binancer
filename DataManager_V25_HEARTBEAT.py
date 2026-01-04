import os
import sys
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import gc
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuração de Timeout Global para evitar conexões zumbis
socket.setdefaulttimeout(15)

MAX_RETRIES = 5
CHUNK_HOURS = 1 # Fatiamento do dia em blocos de 1 hora
HEARTBEAT_INTERVAL = 500 # Imprimir um ponto a cada 500 trades processados

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

class DataManagerV25:
    def __init__(self, symbol="PENDLEUSDT", min_val_usd=2000):
        self.symbol = symbol
        self.min_val_usd = min_val_usd
        self.base_url = "https://api.binance.com/api/v3"
        self.progress_dir = "progress_spot"
        if not os.path.exists(self.progress_dir):
            os.makedirs(self.progress_dir)
        
        # Configuração de Sessão com Retentativa
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_first_id_of_time(self, timestamp_ms):
        url = f"{self.base_url}/aggTrades"
        params = {"symbol": self.symbol, "startTime": timestamp_ms, "limit": 1}
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data: return data[0]['a']
        except Exception as e:
            print_flush(f"⚠️ Erro ao buscar ID inicial: {e}")
            return None

    def download_chunk(self, start_dt, end_dt):
        date_str = start_dt.strftime("%Y-%m-%d")
        chunk_id = start_dt.strftime("%H%M")
        file_path = os.path.join(self.progress_dir, f"data_{date_str}_{chunk_id}.csv")
        
        if os.path.exists(file_path):
            print_flush(f"✅ Chunk {date_str} {chunk_id} já existe. Pulando...")
            return True

        print_flush(f"⏳ Processando Chunk: {date_str} {chunk_id} ({start_dt.strftime('%H:%M')} a {end_dt.strftime('%H:%M')})...")
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        curr_id = self.get_first_id_of_time(start_ts)
        if not curr_id:
            print_flush(f"❌ Não foi possível encontrar ID inicial para o chunk {date_str} {chunk_id}")
            return False

        all_trades = []
        retries = 0
        total_trades_processed = 0
        
        while True:
            if retries >= MAX_RETRIES:
                print_flush(f"❌ Falha persistente no chunk {date_str} {chunk_id}. Pulando...")
                return False

            try:
                url = f"{self.base_url}/aggTrades"
                params = {"symbol": self.symbol, "fromId": curr_id, "limit": 1000} 
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status() 
                trades = r.json()
                
                if not trades: break
                
                new_trades_count = 0
                for t in trades:
                    total_trades_processed += 1
                    
                    # HEARTBEAT FORÇADO
                    if total_trades_processed % HEARTBEAT_INTERVAL == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    
                    ts = int(t['T'])
                    if ts >= end_ts:
                        break 
                    
                    p = float(t['p'])
                    q = float(t['q'])
                    
                    if p * q >= self.min_val_usd:
                        all_trades.append([ts, p, q, -1 if t['m'] else 1])
                        new_trades_count += 1
                
                if int(trades[-1]['T']) >= end_ts: break
                
                curr_id = trades[-1]['a'] + 1
                retries = 0

            except requests.exceptions.RequestException as e:
                retries += 1
                print_flush(f"\n⚠️ Erro de Requisição (Tentativa {retries}/{MAX_RETRIES}): {e}. Retentando...")
                time.sleep(5)
                continue
            except Exception as e:
                retries += 1
                print_flush(f"\n⚠️ Erro Inesperado (Tentativa {retries}/{MAX_RETRIES}): {e}. Retentando...")
                time.sleep(5)
                continue

        if all_trades:
            df = pd.DataFrame(all_trades, columns=['ts', 'price', 'qty', 'side'])
            df.to_csv(file_path, index=False)
            print_flush(f"\n✅ Chunk {date_str} {chunk_id} FINALIZADO. ({len(all_trades)} baleias)")
            del all_trades
            del df
            gc.collect()
            return True
        
        print_flush(f"\nℹ️ Chunk {date_str} {chunk_id} FINALIZADO. (0 baleias)")
        return True

    def run(self, start_date="2024-01-01", end_date="2024-06-30"):
        print_flush("===============================================================")
        print_flush(">>> MOTOR LIGADO: DATA MANAGER V25 HEARTBEAT (1H CHUNKS)")
        print_flush(f">>> ATIVO: {self.symbol} | FILTRO: ${self.min_val_usd}")
        print_flush("===============================================================")
        
        current_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_dt <= end_dt:
            for i in range(24 // CHUNK_HOURS):
                chunk_start = current_dt + timedelta(hours=i * CHUNK_HOURS)
                chunk_end = current_dt + timedelta(hours=(i + 1) * CHUNK_HOURS)
                
                if chunk_end > end_dt + timedelta(days=1):
                    chunk_end = end_dt + timedelta(days=1)
                
                self.download_chunk(chunk_start, chunk_end)
            
            current_dt += timedelta(days=1)
            
        print_flush("\n🚀 TODOS OS CHUNKS CONCLUÍDOS! Gerando arquivo final...")

if __name__ == "__main__":
    # O usuário deve mudar a data de início para 2024-03-25 para recomeçar
    dm = DataManagerV25()
    dm.run(start_date="2024-03-25") # Forçando o início no dia do travamento
