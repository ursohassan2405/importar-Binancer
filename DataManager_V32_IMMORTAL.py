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

# Configuração de Timeout Global
socket.setdefaulttimeout(60)

MAX_RETRIES = 5
CHUNK_HOURS = 1 
# O script vai se "matar" e reiniciar após processar este número de chunks
CHUNKS_BEFORE_RESTART = 12 
HEARTBEAT_INTERVAL = 500 
STATE_FILE = "last_state.txt"

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

class DataManagerV32:
    def __init__(self, symbol="PENDLEUSDT", min_val_usd=2000):
        self.symbol = symbol
        self.min_val_usd = min_val_usd
        self.base_url = "https://api.binance.com/api/v3"
        self.progress_dir = "progress_spot"
        if not os.path.exists(self.progress_dir):
            os.makedirs(self.progress_dir)
        
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

    def _read_state(self):
        """Lê o último estado salvo (last_date, last_chunk_id, last_trade_id)"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = f.read().split(',')
                if len(data) == 3:
                    return data[0], data[1], int(data[2])
        return "2024-01-01", "0000", None

    def _write_state(self, current_date, current_chunk_id, last_trade_id):
        """Salva o estado atual para que o script possa recomeçar."""
        with open(STATE_FILE, 'w') as f:
            f.write(f"{current_date},{current_chunk_id},{last_trade_id}")

    def get_first_id_of_time(self, timestamp_ms):
        """Busca o primeiro tradeId para um dado timestamp. Usado apenas no início do dia."""
        url = f"{self.base_url}/aggTrades"
        params = {"symbol": self.symbol, "startTime": timestamp_ms, "limit": 1}
        try:
            r = self.session.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if data: return data[0]['a']
        except Exception as e:
            print_flush(f"⚠️ Erro ao buscar ID inicial: {e}")
            return None

    def download_chunk(self, start_dt, end_dt, last_trade_id=None):
        date_str = start_dt.strftime("%Y-%m-%d")
        chunk_id = start_dt.strftime("%H%M")
        file_path = os.path.join(self.progress_dir, f"data_{date_str}_{chunk_id}.csv")
        
        if os.path.exists(file_path):
            print_flush(f"✅ Chunk {date_str} {chunk_id} já existe. Pulando...")
            return True, None

        print_flush(f"⏳ Processando Chunk: {date_str} {chunk_id} ({start_dt.strftime('%H:%M')} a {end_dt.strftime('%H:%M')})...")
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # Lógica de ID Sequencial
        curr_id = last_trade_id
        if curr_id is None:
            curr_id = self.get_first_id_of_time(start_ts)
            if not curr_id:
                print_flush(f"❌ Não foi possível encontrar ID inicial para o chunk {date_str} {chunk_id}")
                return False, None

        all_trades = []
        retries = 0
        total_trades_processed = 0
        
        while True:
            if retries >= MAX_RETRIES:
                print_flush(f"\n❌ Falha persistente no chunk {date_str} {chunk_id}. Pulando...")
                return False, None

            try:
                url = f"{self.base_url}/aggTrades"
                params = {"symbol": self.symbol, "fromId": curr_id, "limit": 1000} 
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status() 
                trades = r.json()
                
                if not trades: break
                
                last_trade_id_in_chunk = None
                
                for t in trades:
                    total_trades_processed += 1
                    
                    if total_trades_processed % HEARTBEAT_INTERVAL == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    
                    ts = int(t['T'])
                    if ts >= end_ts:
                        break 
                    
                    p = float(t['p'])
                    q = float(t['q'])
                    
                    if p * q >= self.min_val_usd:
                        side = -1 if t['m'] else 1
                        all_trades.append([ts, p, q, side])
                    
                    last_trade_id_in_chunk = t['a']

                if int(trades[-1]['T']) >= end_ts:
                    next_id = last_trade_id_in_chunk + 1 if last_trade_id_in_chunk else trades[-1]['a'] + 1
                    break
                
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

        next_id = curr_id if 'next_id' not in locals() else next_id

        if all_trades:
            df = pd.DataFrame(all_trades, columns=['ts', 'price', 'qty', 'side'])
            df.to_csv(file_path, index=False)
            print_flush(f"\n✅ Chunk {date_str} {chunk_id} FINALIZADO. ({len(all_trades)} baleias)")
            del all_trades
            del df
            gc.collect()
            return True, next_id
        
        print_flush(f"\nℹ️ Chunk {date_str} {chunk_id} FINALIZADO. (0 baleias)")
        return True, next_id

    def run(self, end_date="2024-06-30"):
        print_flush("===============================================================")
        print_flush(">>> MOTOR LIGADO: DATA MANAGER V32 THE IMMORTAL (AUTO-RESTART)")
        print_flush(f">>> ATIVO: {self.symbol} | FILTRO: ${self.min_val_usd}")
        print_flush("===============================================================")
        
        # 1. Recuperar o estado
        start_date_str, start_chunk_id_str, last_trade_id = self._read_state()
        
        current_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Ajustar o current_dt para o chunk correto
        start_hour = int(start_chunk_id_str[:2])
        current_dt = current_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        chunks_processed_in_session = 0
        
        while current_dt <= end_dt:
            for i in range(24 // CHUNK_HOURS):
                chunk_start = current_dt + timedelta(hours=i * CHUNK_HOURS)
                chunk_end = current_dt + timedelta(hours=(i + 1) * CHUNK_HOURS)
                
                if chunk_end > end_dt + timedelta(days=1):
                    chunk_end = end_dt + timedelta(days=1)
                
                # Se o chunk atual for anterior ao ponto de partida, pulamos
                if chunk_start.strftime("%Y-%m-%d %H%M") < current_dt.strftime("%Y-%m-%d %H%M"):
                    continue

                # 2. Download do Chunk
                success, next_id = self.download_chunk(chunk_start, chunk_end, last_trade_id)
                
                if next_id is not None:
                    last_trade_id = next_id
                
                if not success:
                    print_flush("\n🚨 FALHA CRÍTICA NO CHUNK. SALVANDO ESTADO E REINICIANDO.")
                    self._write_state(chunk_start.strftime("%Y-%m-%d"), chunk_start.strftime("%H%M"), last_trade_id)
                    sys.exit(1) # Força o Render a reiniciar o serviço

                # 3. Lógica de Auto-Suicídio
                chunks_processed_in_session += 1
                if chunks_processed_in_session >= CHUNKS_BEFORE_RESTART:
                    print_flush(f"\n♻️ LIMITE DE {CHUNKS_BEFORE_RESTART} CHUNKS ATINGIDO. SALVANDO ESTADO E REINICIANDO.")
                    # Salva o estado do PRÓXIMO chunk a ser processado
                    next_chunk_start = chunk_end
                    self._write_state(next_chunk_start.strftime("%Y-%m-%d"), next_chunk_start.strftime("%H%M"), last_trade_id)
                    sys.exit(0) # Saída limpa para forçar o Render a reiniciar

                # 4. Atualizar o estado após o sucesso
                self._write_state(chunk_end.strftime("%Y-%m-%d"), chunk_end.strftime("%H%M"), last_trade_id)

            current_dt += timedelta(days=1)
            
        print_flush("\n🚀 TODOS OS CHUNKS CONCLUÍDOS! Processamento finalizado.")

if __name__ == "__main__":
    dm = DataManagerV32()
    dm.run()
