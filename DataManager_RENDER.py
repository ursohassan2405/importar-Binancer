import ccxt
import pandas as pd
import time
import os
import json
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DE AMBIENTE ---
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
MIN_VAL_USD = cfg.get('min_val_usd', 500)
PASTA_DADOS = "/data"

def main():
    print(f"🚀 INICIANDO DATA MANAGER ELITE V16 (CCXT): {SIMBOLO}")
    
    # Inicializa a Binance via CCXT (Modo Futuros)
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'},
        'timeout': 30000,
        'enableRateLimit': True
    })

    # 1. Sincronização de Tempo
    now_ms = exchange.milliseconds()
    since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
    
    print(f"📅 Período: {pd.to_datetime(since, unit='ms')} até agora")

    try:
        # 2. Download de OHLCV (Candles)
        all_klines = []
        temp_since = since
        while temp_since < now_ms:
            print(f"  -> Baixando Candles: {pd.to_datetime(temp_since, unit='ms')}")
            klines = exchange.fetch_ohlcv(SIMBOLO, TIMEFRAME, temp_since, limit=1000)
            if not klines: break
            all_klines.extend(klines)
            temp_since = klines[-1][0] + 1
            if len(klines) < 1000: break
            time.sleep(0.1)

        df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts_dt'] = pd.to_datetime(df['ts'], unit='ms')

        # 3. Download de Trades Institucionais (Whale Hunter)
        # Usamos fetch_trades para contornar bloqueios de aggTrades
        print(f"🐳 Capturando Fluxo de Agressão (Whale Hunter Mode)...")
        all_trades = []
        # Pegamos os trades mais recentes para injetar a microestrutura atual
        trades = exchange.fetch_trades(SIMBOLO, limit=1000)
        
        if trades:
            tdf = pd.DataFrame(trades, columns=['timestamp', 'price', 'amount', 'side'])
            # Filtro de Baleias Real
            tdf['val_usd'] = tdf['price'] * tdf['amount']
            baleias = tdf[tdf['val_usd'] >= MIN_VAL_USD].copy()
            
            print(f"✅ {len(baleias)} trades institucionais identificados!")
            
            # Injeção de Microestrutura (Lógica Simplificada para Estabilidade)
            df['buy_vol'] = baleias[baleias['side'] == 'buy']['amount'].sum() / len(df)
            df['sell_vol'] = baleias[baleias['side'] == 'sell']['amount'].sum() / len(df)
            df['delta'] = df['buy_vol'] - df['sell_vol']
        
        # 4. Salvar no Disco Persistente
        if not os.path.exists(PASTA_DADOS):
            try: os.makedirs(PASTA_DADOS)
            except: pass

        caminho = f"{PASTA_DADOS}/{SIMBOLO}_{TIMEFRAME}.csv"
        # Removemos a coluna auxiliar antes de salvar
        df_save = df.drop(columns=['ts_dt']) if 'ts_dt' in df.columns else df
        df_save.to_csv(caminho, index=False)
        
        print(f"✅ SUCESSO! Arquivo salvo: {caminho} | Linhas: {len(df)}")

        # 5. Listagem de Segurança (Para você ver no Log)
        print("\n" + "="*30)
        print("📂 CONTEÚDO DO DISCO /data:")
        arquivos = os.listdir(PASTA_DADOS)
        for arq in arquivos:
            tamanho = os.path.getsize(f"{PASTA_DADOS}/{arq}") / (1024*1024)
            print(f"📄 {arq} | Tamanho: {tamanho:.2f} MB")
        print("="*30 + "\n")

    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")

if __name__ == "__main__":
    main()
