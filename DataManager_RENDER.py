import requests
import pandas as pd
import time
import os
import numpy as np
import json

print("[DEBUG 1] Script iniciado...")

def main():
    print("[DEBUG 2] Entrou na função main...")
    
    # Teste de conexão simples
    try:
        print("[DEBUG 3] Testando conexão com Binance...")
        r = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=10 )
        print(f"[DEBUG 4] Resposta Binance: {r.status_code} | Time: {r.json()['serverTime']}")
    except Exception as e:
        print(f"[DEBUG 4] ERRO de Conexão: {e}")

    config_path = "config_render.json"
    print(f"[DEBUG 5] Procurando config em: {os.path.abspath(config_path)}")
    
    if not os.path.exists(config_path):
        print("[DEBUG 6] ERRO: config_render.json NÃO ENCONTRADO!")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
        print("[DEBUG 7] Configuração carregada com sucesso.")

    dm_cfg = config.get('data_manager', {})
    simbolo = dm_cfg.get('symbol', 'RUNEUSDT')
    print(f"[DEBUG 8] Símbolo alvo: {simbolo}")

    # Teste de escrita no disco
    try:
        print("[DEBUG 9] Testando escrita no disco /data...")
        if not os.path.exists("/data"):
            print("[DEBUG 10] Pasta /data não existe, tentando criar...")
            os.makedirs("/data")
        
        with open("/data/teste.txt", "w") as f:
            f.write("teste")
        print("[DEBUG 11] Teste de disco OK!")
    except Exception as e:
        print(f"[DEBUG 11] ERRO de Disco: {e}")

    print("[DEBUG 12] Iniciando download de OHLCV simplificado...")
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={simbolo}&interval=15m&limit=100"
    data = requests.get(url ).json()
    print(f"[DEBUG 13] Download OK! Recebidos {len(data)} candles.")
    
    print("="*50)
    print("DIAGNÓSTICO CONCLUÍDO - O SISTEMA PODE RODAR")
    print("="*50)

if __name__ == "__main__":
    main()
