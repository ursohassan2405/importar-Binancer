import ccxt
import pandas as pd
import time
import os
import json
import numpy as np
from datetime import datetime, timedelta

def get_config():
    config_path = "config_render.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f).get('data_manager', {})
    return {}

cfg = get_config()
SIMBOLO = cfg.get('symbol', 'RUNEUSDT')
TIMEFRAME = cfg.get('timeframe', '15m')
DAYS_BACK = cfg.get('days_back', 30)
PASTA_DADOS = "/data"

def main():
    print(f"🚀 INICIANDO MODO RESILIENTE (CCXT): {SIMBOLO}")
    
    # Inicializa a Binance via CCXT (Mais robusto contra bloqueios)
    exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000, 'enableRateLimit': True})

    # 1. Calcular tempo
    since = exchange.milliseconds() - (DAYS_BACK * 24 * 60 * 60 * 1000)
    
    all_klines = []
    print(f"📅 Baixando histórico desde: {pd.to_datetime(since, unit='ms')}")

    try:
        # 2. Download de OHLCV (Candles)
        while since < exchange.milliseconds():
            print(f"  -> Coletando candles a partir de: {pd.to_datetime(since, unit='ms')}")
            klines = exchange.fetch_ohlcv(SIMBOLO, TIMEFRAME, since, limit=1000)
            if not klines: break
            all_klines.extend(klines)
            since = klines[-1][0] + 1
            time.sleep(0.5) # Respeitar rate limit
            if len(klines) < 1000: break

        df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')

        # 3. Download de Trades (Baleias) via CCXT
        # Como a Binance bloqueia aggTrades longos, vamos pegar os trades mais recentes
        print(f"🐳 Capturando Trades Institucionais recentes...")
        trades = exchange.fetch_trades(SIMBOLO, limit=1000)
        
        if trades:
            tdf = pd.DataFrame(trades, columns=['timestamp', 'price', 'amount', 'side'])
            print(f"✅ {len(tdf)} trades capturados com sucesso!")
            # Aqui injetamos uma lógica simplificada de volume para não travar
            df['n_trades_whale'] = len(tdf) / len(df) # Média ilustrativa para manter compatibilidade
        
        # 4. Salvar no Disco
        caminho = f"{PASTA_DADOS}/{SIMBOLO}_{TIMEFRAME}.csv"
        df.to_csv(caminho, index=False)
        print(f"✅ SUCESSO TOTAL! Arquivo salvo: {caminho} | Linhas: {len(df)}")

    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")

if __name__ == "__main__":
    main()
