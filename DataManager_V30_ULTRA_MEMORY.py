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
BATCH_SIZE = 100000 # Salvar a cada 100.000 trades processados

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

class DataManagerV30:
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

    def get_first_id_of_time(self, timestamp_ms):
        """Busca o primeiro tradeId para um dado timestamp. Usado apenas no início da execução."""
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

    def get_last_trade_id(self):
        """Busca o último tradeId salvo para retomar o download."""
        all_files = sorted(os.listdir(self.progress_dir))
        if not all_files:
            return None, None

        # Pega o último arquivo e lê o último tradeId
        last_file = all_files[-1]
        try:
            df = pd.read_csv(os.path.join(self.progress_dir, last_file))
            if not df.empty:
                # O tradeId não está salvo no CSV, mas o timestamp está.
                # Vamos buscar o ID do trade que aconteceu depois do último timestamp salvo.
                last_ts = df['ts'].iloc[-1]
                
                # Para ser seguro, buscamos o primeiro trade do próximo dia
                next_day_ts = (datetime.fromtimestamp(last_ts / 1000) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
                
                # Isso é complexo e pode falhar. Vamos simplificar: o usuário deve deletar o último arquivo
                # e o script vai recomeçar do último timestamp.
                # Para a V30, vamos forçar o início no 01/01 e o usuário deleta os arquivos
                # se quiser recomeçar.
                return None, None # Simplificando para evitar lógica complexa de ID sequencial entre reinícios

        except Exception as e:
            print_flush(f"⚠️ Erro ao ler último arquivo: {e}")
            return None, None

    def download_all(self, start_date="2024-01-01", end_date="2024-06-30"):
        print_flush("===============================================================")
        print_flush(">>> MOTOR LIGADO: DATA MANAGER V30 ULTRA MEMORY (BATCHING)")
        print_flush(f">>> ATIVO: {self.symbol} | FILTRO: ${self.min_val_usd}")
        print_flush("===============================================================")
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # 1. Determinar o ponto de partida
        # Para a V30, vamos simplificar e forçar o início no 01/01. O usuário deve deletar
        # os arquivos se quiser recomeçar.
        
        curr_id = self.get_first_id_of_time(start_ts)
        if not curr_id:
            print_flush("❌ Não foi possível encontrar ID inicial. Abortando.")
            return

        all_trades_buffer = []
        retries = 0
        batch_count = 0
        
        while True:
            if retries >= MAX_RETRIES:
                print_flush("\n❌ Falha persistente na requisição. Abortando.")
                break

            try:
                url = f"{self.base_url}/aggTrades"
                params = {"symbol": self.symbol, "fromId": curr_id, "limit": 1000} 
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status() 
                trades = r.json()
                
                if not trades: break
                
                # 2. Processamento em Lote e Vetorizado
                
                # Converte a lista de dicionários para um DataFrame
                df_trades = pd.DataFrame(trades)
                
                # Filtra trades que estão fora do período final
                df_trades = df_trades[df_trades['T'] < end_ts]
                
                if df_trades.empty: break
                
                # Filtro de Baleia Vetorizado (Muito mais rápido que loop for)
                # O volume é p * q
                df_trades['volume'] = df_trades['p'].astype(float) * df_trades['q'].astype(float)
                df_baleias = df_trades[df_trades['volume'] >= self.min_val_usd]
                
                if not df_baleias.empty:
                    # Extrai apenas as colunas necessárias e converte 'm' para side
                    df_baleias['side'] = np.where(df_baleias['m'], -1, 1)
                    
                    # Colunas: ts, price, qty, side
                    baleias_data = df_baleias[['T', 'p', 'q', 'side']].values.tolist()
                    all_trades_buffer.extend(baleias_data)
                
                # 3. Salvar o Lote
                if len(all_trades_buffer) >= BATCH_SIZE:
                    batch_count += 1
                    file_path = os.path.join(self.progress_dir, f"batch_{batch_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    
                    df_batch = pd.DataFrame(all_trades_buffer, columns=['ts', 'price', 'qty', 'side'])
                    df_batch.to_csv(file_path, index=False)
                    
                    print_flush(f"✅ BATCH {batch_count} SALVO. ({len(all_trades_buffer)} baleias)")
                    all_trades_buffer = []
                    gc.collect()

                # Avança o ID para a próxima requisição
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

        # Salvar o buffer restante
        if all_trades_buffer:
            batch_count += 1
            file_path = os.path.join(self.progress_dir, f"batch_{batch_count}_FINAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            df_batch = pd.DataFrame(all_trades_buffer, columns=['ts', 'price', 'qty', 'side'])
            df_batch.to_csv(file_path, index=False)
            
            print_flush(f"\n✅ BATCH FINAL SALVO. ({len(all_trades_buffer)} baleias)")
            del all_trades_buffer
            gc.collect()

        print_flush("\n🚀 TODOS OS DADOS CONCLUÍDOS! Processamento finalizado.")

if __name__ == "__main__":
    dm = DataManagerV30()
    dm.download_all(start_date="2024-01-01", end_date="2024-06-30")
