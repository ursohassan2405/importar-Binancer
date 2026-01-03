#!/usr/bin/env python3
"""
DATA MANAGER RENDER 1M - Otimizado para Download Rápido, Cronometragem e Upload Único via GitHub (Repositório Existente)
"""

import requests
import pandas as pd
import time
import json
import os
import sys
import subprocess
from datetime import datetime
import base64
from concurrent.futures import ThreadPoolExecutor

# --- Configuração ---
CONFIG_FILE = 'config_datamanager_1m.json'
OUTPUT_DIR = '/data'
ZIP_PATH = '/tmp/dados_1m_FINAL.zip'
GITHUB_TOKEN = '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' # Token fornecido pelo usuário
REPO_FULL_NAME = 'ursohassan2405/importar-Binancer' # Repositório existente do usuário
# --------------------

def load_config():
    """Carrega a configuração do arquivo JSON."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de configuração '{CONFIG_FILE}' não encontrado.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERRO] Erro ao decodificar o arquivo JSON '{CONFIG_FILE}'.")
        sys.exit(1)

def calculate_time_range(days_back):
    """Calcula o range de tempo para o download."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days_back * 24 * 3600 * 1000)
    return start_ms, end_ms

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    """Download de um chunk de klines com tratamento de rate limit."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    limit = 1500
    ms_per_kline = 60000  # 1 minuto
    
    while curr < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(curr),
            "endTime": int(end_ms),
            "limit": limit
        }
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data:
                    break
                rows.extend(data)
                curr = data[-1][0] + ms_per_kline
                if len(data) < limit:
                    break
            elif r.status_code in [418, 429]:
                print(f"[AVISO] Rate limit. Aguardando 15s...")
                time.sleep(15)
            else:
                time.sleep(2)
        except Exception as e:
            print(f"[ERRO Klines] {e}")
            time.sleep(5)
    
    return rows

def download_futures_klines(symbol, interval, days_back):
    """Download paralelo de Klines em 1m."""
    start_ms, end_ms = calculate_time_range(days_back)
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD Klines: {symbol} - {interval}")
    print(f"{'='*70}")
    
    slice_ms = 15 * 24 * 3600 * 1000
    chunks = []
    curr = start_ms
    while curr < end_ms:
        chunks.append((curr, min(curr + slice_ms, end_ms)))
        curr += slice_ms
    
    print(f"[INFO] Total de chunks a processar: {len(chunks)}")
    
    start_time = time.time()
    all_rows = []
    klines_downloaded = 0
    
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    download_klines_chunk,
                    symbol,
                    interval,
                    c[0],
                    c[1],
                    session
                ) for c in chunks
            ]
            
            for i, f in enumerate(futures, 1):
                res = f.result()
                if res:
                    all_rows.extend(res)
                    klines_downloaded += len(res)
                    elapsed = time.time() - start_time
                    rate = klines_downloaded / elapsed if elapsed > 0 else 0
                    print(f"    [Chunk {i}/{len(chunks)}] {klines_downloaded} klines | Tempo: {elapsed:.1f}s | Taxa: {rate:.0f} klines/s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if not all_rows:
        print("[ERRO] Nenhum kline foi baixado!")
        return None, total_time
    
    df_temp = pd.DataFrame(all_rows).drop_duplicates(subset=0).sort_values(by=0)
    
    print(f"\n{'='*70}")
    print(f"RESUMO DE DOWNLOAD Klines")
    print(f"Tempo Total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Klines Baixados: {len(df_temp)}")
    print(f"Taxa de Download: {len(df_temp)/total_time:.0f} klines/segundo")
    print(f"{'='*70}\n")
    
    return df_temp, total_time

def download_agg_trades(symbol, days_back, min_val_usd):
    """Simula o download de AggTrades (Whale Trades)."""
    print(f"\n{'='*70}")
    print(f"DOWNLOAD AggTrades (Whale Trades): {symbol}")
    print(f"{'='*70}")
    print(f"[AVISO] Usando simulação de tempo para download de AggTrades.")
    
    start_time = time.time()
    time.sleep(5) # Simula 5 segundos de download
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Tempo Total (Simulado): {total_time:.2f} segundos")
    print(f"{'='*70}\n")
    
    return None, total_time

def processar_microestrutura(df_klines, trades_np, interval):
    """Simula o processamento de microestrutura."""
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO DE MICROESTRUTURA")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    if df_klines is not None:
        process_time = len(df_klines) / 10000
        time.sleep(min(process_time, 10)) # Limite de 10s
    else:
        time.sleep(1)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Tempo Total (Simulado): {total_time:.2f} segundos")
    print(f"{'='*70}\n")
    
    return df_klines, total_time

def upload_file_to_github(token, repo_full_name, file_path):
    """Faz o upload de um arquivo para o repositório via API de Conteúdo."""
    print(f"[GITHUB] Tentando fazer upload de {os.path.basename(file_path)} para {repo_full_name}")
    
    # 1. Ler e codificar o arquivo
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        encoded_content = base64.b64encode(content).decode('utf-8')
    except Exception as e:
        print(f"[ERRO GITHUB] Falha ao ler o arquivo: {e}")
        return None

    # 2. Tentar obter o SHA do arquivo existente (para atualização)
    file_name = os.path.basename(file_path)
    url_get = f"https://api.github.com/repos/{repo_full_name}/contents/{file_name}"
    headers_get = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response_get = requests.get(url_get, headers=headers_get)
    sha = None
    if response_get.status_code == 200:
        sha = response_get.json().get('sha')
        print(f"[GITHUB] Arquivo existente encontrado. SHA: {sha}")

    # 3. Fazer o upload/atualização
    url_put = f"https://api.github.com/repos/{repo_full_name}/contents/{file_name}"
    headers_put = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "message": f"Upload de dados de backtest: {file_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": encoded_content,
        "sha": sha # Inclui o SHA se for uma atualização
    }
    
    response_put = requests.put(url_put, headers=headers_put, json=data)
    
    if response_put.status_code in [200, 201]:
        print("[GITHUB] Upload/Atualização concluído com sucesso.")
        return response_put.json()['content']['html_url']
    else:
        print(f"[ERRO GITHUB] Falha ao fazer upload ({response_put.status_code}): {response_put.json().get('message', 'Erro desconhecido')}")
        return None

def main():
    config = load_config()
    symbols = config.get('symbols', [])
    interval = config.get('timeframe', '1m')
    days_back = config.get('days_back', 180)
    min_val_usd = config.get('min_val_usd', 500)
    
    total_time_klines = 0
    total_time_trades = 0
    total_time_process = 0
    
    files_to_zip = []
    
    for symbol in symbols:
        # ... (Processamento de dados)
        print(f"\n\n{'#'*80}")
        print(f"INICIANDO PROCESSAMENTO PARA {symbol}")
        print(f"{'#'*80}")
        
        # 1. Download Klines
        df_klines, time_klines = download_futures_klines(symbol, interval, days_back)
        total_time_klines += time_klines
        
        # 2. Download AggTrades (Simulado)
        trades_np, time_trades = download_agg_trades(symbol, days_back, min_val_usd)
        total_time_trades += time_trades
        
        # 3. Processamento de Microestrutura (Simulado)
        df_final, time_process = processar_microestrutura(df_klines, trades_np, interval)
        total_time_process += time_process
        
        # 4. Salvar
        if df_final is not None:
            output_path = os.path.join(OUTPUT_DIR, f"{symbol}_{interval}.csv")
            df_final.to_csv(output_path, index=False)
            print(f"[SUCESSO] Dados salvos em: {output_path}")
            files_to_zip.append(output_path)
        
        print(f"\n{'#'*80}")
        print(f"FIM DO PROCESSAMENTO PARA {symbol}")
        print(f"{'#'*80}\n")
        
    print(f"\n\n{'='*80}")
    print(f"RESUMO GERAL DE TEMPO")
    print(f"Tempo Total Klines: {total_time_klines:.2f}s")
    print(f"Tempo Total Trades (Simulado): {total_time_trades:.2f}s")
    print(f"Tempo Total Processamento (Simulado): {total_time_process:.2f}s")
    print(f"Tempo Total Estimado: {total_time_klines + total_time_trades + total_time_process:.2f}s")
    print(f"{'='*80}\n")
    
    # --- Etapa Final: Compactar e Upload para GitHub ---
    if files_to_zip:
        try:
            # 1. Compacta os arquivos
            # Usamos 'zip -j' para não incluir a estrutura de diretórios /data/
            zip_command = ["zip", "-j", ZIP_PATH] + files_to_zip
            subprocess.run(zip_command, check=True, capture_output=True)
            print(f"[SUCESSO] Arquivos compactados em: {ZIP_PATH}")
            
            # 2. Faz o upload do ZIP
            upload_url = upload_file_to_github(GITHUB_TOKEN, REPO_FULL_NAME, ZIP_PATH)
            
            if upload_url:
                # 3. Imprime o link de download
                raw_download_url = f"https://raw.githubusercontent.com/{REPO_FULL_NAME}/main/{os.path.basename(ZIP_PATH)}"
                
                print(f"\n\n{'*'*80}")
                print(f"LINK DE DOWNLOAD ÚNICO (VIA GITHUB):")
                print(f"  Repositório: https://github.com/{REPO_FULL_NAME}")
                print(f"  Link Direto para Download: {raw_download_url}")
                print(f"  AVISO: O arquivo foi salvo no seu repositório '{REPO_FULL_NAME}'.")
                print(f"{'*'*80}\n")
            else:
                print("[ERRO FATAL] Falha no upload para o GitHub.")
            
        except subprocess.CalledProcessError as e:
            print(f"[ERRO ZIP] Falha ao compactar os arquivos: {e.stderr.decode().strip()}")
            
if __name__ == "__main__":
    main()
