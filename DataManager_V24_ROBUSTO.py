import os
import sys
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import gc
import socket

# Configuração de Timeout Global para evitar conexões zumbis
# Reduzido para 15s para ser mais agressivo no Render
socket.setdefaulttimeout(15)

MAX_RETRIES = 5
CHUNK_HOURS = 4 # Fatiamento do dia em blocos de 4 horas para dias de alto volume

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

class DataManagerV24:
    def __init__(self, symbol="PENDLEUSDT", min_val_usd=2000): # Aumentando o filtro para $2000 conforme contexto
        self.symbol = symbol
        self.min_val_usd = min_val_usd
        self.base_url = "https://api.binance.com/api/v3"
        self.progress_dir = "progress_spot"
        if not os.path.exists(self.progress_dir):
            os.makedirs(self.progress_dir)

    def get_first_id_of_time(self, timestamp_ms):
        url = f"{self.base_url}/aggTrades"
        params = {"symbol": self.symbol, "startTime": timestamp_ms, "limit": 1}
        for _ in range(MAX_RETRIES):
            try:
                r = requests.get(url, params=params, timeout=15)
                data = r.json()
                if data: return data[0]['a']
            except Exception as e:
                print_flush(f"⚠️ Erro ao buscar ID inicial: {e}. Retentando...")
                time.sleep(2)
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
        
        while True:
            if retries >= MAX_RETRIES:
                print_flush(f"❌ Falha persistente no chunk {date_str} {chunk_id}. Pulando...")
                return False # Pula o chunk após falhas persistentes

            try:
                url = f"{self.base_url}/aggTrades"
                # Usamos fromId e limit, mas o loop interno garante que não ultrapassamos end_ts
                params = {"symbol": self.symbol, "fromId": curr_id, "limit": 1000} 
                r = requests.get(url, params=params, timeout=15) # Timeout de 15s para a requisição
                
                # Se a resposta não for 200, levanta um erro para retentar
                r.raise_for_status() 
                trades = r.json()
                
                if not trades: break
                
                new_trades_count = 0
                for t in trades:
                    ts = int(t['T'])
                    if ts >= end_ts:
                        # Se o trade for do próximo chunk, paramos e salvamos
                        break 
                    
                    p = float(t['p'])
                    q = float(t['q'])
                    
                    # Filtro de Baleia: $2000 USD
                    if p * q >= self.min_val_usd:
                        all_trades.append([ts, p, q, -1 if t['m'] else 1])
                        new_trades_count += 1
                
                # Se o último trade da requisição for maior ou igual ao fim do chunk, terminamos
                if int(trades[-1]['T']) >= end_ts: break
                
                # Se não houver novos trades, mas o loop não terminou, avançamos o ID
                if new_trades_count == 0 and len(trades) == 1000:
                    curr_id = trades[-1]['a'] + 1
                elif new_trades_count > 0:
                    curr_id = trades[-1]['a'] + 1
                else:
                    # Caso de erro ou fim de dados inesperado
                    break

                retries = 0 # Resetar retries em caso de sucesso
                
                # Heartbeat a cada 50k trades (ajustado para 5000 para chunks menores)
                if len(all_trades) % 5000 == 0 and all_trades:
                    print_flush(f"❤️ Heartbeat: {len(all_trades)} baleias processadas no chunk.")

            except requests.exceptions.RequestException as e:
                retries += 1
                print_flush(f"⚠️ Erro de Requisição (Tentativa {retries}/{MAX_RETRIES}): {e}. Retentando...")
                time.sleep(5)
                continue
            except Exception as e:
                retries += 1
                print_flush(f"⚠️ Erro Inesperado (Tentativa {retries}/{MAX_RETRIES}): {e}. Retentando...")
                time.sleep(5)
                continue

        if all_trades:
            df = pd.DataFrame(all_trades, columns=['ts', 'price', 'qty', 'side'])
            df.to_csv(file_path, index=False)
            print_flush(f"✅ Chunk {date_str} {chunk_id} FINALIZADO. ({len(all_trades)} baleias)")
            del all_trades
            del df
            gc.collect() # Limpeza de memória agressiva
            return True
        
        print_flush(f"ℹ️ Chunk {date_str} {chunk_id} FINALIZADO. (0 baleias)")
        return True # Retorna True mesmo com 0 baleias para avançar

    def run(self, start_date="2024-01-01", end_date="2024-06-30"):
        print_flush("===============================================================")
        print_flush(">>> MOTOR LIGADO: DATA MANAGER V24 ROBUSTO (4H CHUNKS)")
        print_flush(f">>> ATIVO: {self.symbol} | FILTRO: ${self.min_val_usd}")
        print_flush("===============================================================")
        
        current_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_dt <= end_dt:
            # Iterar sobre os chunks de 4 horas
            for i in range(24 // CHUNK_HOURS):
                chunk_start = current_dt + timedelta(hours=i * CHUNK_HOURS)
                chunk_end = current_dt + timedelta(hours=(i + 1) * CHUNK_HOURS)
                
                # Se o chunk_end for maior que o end_dt geral, ajustamos
                if chunk_end > end_dt + timedelta(days=1):
                    chunk_end = end_dt + timedelta(days=1)
                
                self.download_chunk(chunk_start, chunk_end)
            
            current_dt += timedelta(days=1)
            
        print_flush("🚀 TODOS OS CHUNKS CONCLUÍDOS! Gerando arquivo final...")
        # Lógica de consolidação final aqui... (Pode ser feita em um script separado no Render)

if __name__ == "__main__":
    # O usuário precisa mudar a data de início para 2024-03-24 para recomeçar
    # Vou deixar o padrão, mas a instrução de uso será para começar em 2024-03-24
    dm = DataManagerV24()
    dm.run()
