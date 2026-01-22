
# DATA MANAGER INSTITUCIONAL ELITE V16
# Arquitetura de Agregação Inteligente (1m -> Multi-TF)
# Otimizado para Microestrutura e Alta Performance

import requests
import pandas as pd
import time
from datetime import datetime
import os
import sys
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# 🔧 CONFIGURAÇÃO AUTOMÁTICA RENDER
# ============================================================================
RUNNING_ON_RENDER = os.path.exists("/opt/render/project")

if RUNNING_ON_RENDER:
    # Procurar config em múltiplos lugares
    CONFIG_PATHS = [
        "/data/config_elite_render.json",
        "/opt/render/project/src/config_elite_render.json",
        "./config_elite_render.json",
    ]
    
    CONFIG_PATH = None
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            CONFIG_PATH = path
            break
    
    if CONFIG_PATH:
        with open(CONFIG_PATH, 'r') as f:
            CONFIG = json.load(f)
        print("="*60)
        print("🌐 RENDER: Usando configuração automática")
        print(f"   Config: {CONFIG_PATH}")
        print("="*60)
    else:
        # Config padrão se não encontrar arquivo
        CONFIG = {
            "simbolo": "PENDLEUSDT",
            "timeframes": "15m,30m,1h,4h,8h,1d",
            "data_inicial": "2025-12-01",
            "data_final": "2025-12-30",
            "pasta_saida": "/data/PENDLEUSDT_DATA",
            "ativar_micro": "s",
            "min_val_usd": 500
        }
        print("="*60)
        print("🌐 RENDER: Usando configuração padrão")
        print("="*60)
else:
    CONFIG = None
    print("💻 MODO LOCAL: Inputs interativos")

def get_config(key, prompt, default):
    """Função universal para inputs - Render usa CONFIG, Local usa input()"""
    if RUNNING_ON_RENDER and CONFIG:
        value = CONFIG.get(key, default)
        print(f"{prompt}{value}")
        return value
    else:
        result = input(prompt).strip()
        return result if result else default
# ============================================================================

CONTRATO = [
    "ts","open","high","low","close","volume",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","close_time"
]

def banner():
    print("\n" + "="*50)
    print("   DATA MANAGER INSTITUCIONAL ELITE V16")
    print("   >>> Smart Aggregation & Microstructure <<<")
    print("="*50 + "\n")

def ask(msg):
    return input(msg).strip()

# -----------------------------------------------------------------------------
# DOWNLOADER DE ALTA PERFORMANCE
# -----------------------------------------------------------------------------

def download_klines_chunk(symbol, interval, start_ms, end_ms, session):
    url = "https://fapi.binance.com/fapi/v1/klines"
    rows = []
    curr = start_ms
    limit = 1500
    ms_per_kline = 60000 # Focado em 1m
    
    while curr < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": int(curr), "endTime": int(end_ms), "limit": limit}
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data: break
                rows.extend(data)
                curr = data[-1][0] + ms_per_kline
                if len(data) < limit: break # Chegou ao fim dos dados disponíveis
            elif r.status_code in [418, 429]:
                print(f"[AVISO Klines] Rate limit. Aguardando 15s...")
                time.sleep(15)
            else:
                time.sleep(2)
        except Exception as e:
            print(f"[ERRO Klines Chunk] {e}")
            time.sleep(5)
    return rows

def download_futures_klines(symbol, interval, start_ms, end_ms):
    """Download paralelo de Klines para máxima velocidade (Nitro Mode)."""
    print(f"[INFO] Baixando Klines {interval} para {symbol} (Nitro Mode)...")
    
    # Fatias de 15 dias para Klines de 1m
    slice_ms = 15 * 24 * 3600000
    chunks = []
    curr = start_ms
    while curr < end_ms:
        chunks.append((curr, min(curr + slice_ms, end_ms)))
        curr += slice_ms
    
    all_rows = []
    with requests.Session() as session:
        # 🚀 5 workers simultâneos para acelerar 5x o download
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_klines_chunk, symbol, interval, c[0], c[1], session) for c in chunks]
            for f in futures:
                res = f.result()
                if res:
                    all_rows.extend(res)
                    if len(all_rows) % 1500 == 0 or len(all_rows) > 500000:
                        print(f"    - {len(all_rows)} klines acumuladas...")
            
    if not all_rows: return []
    # Ordenar e remover duplicatas
    df_temp = pd.DataFrame(all_rows).drop_duplicates(subset=0).sort_values(by=0)
    return df_temp.values.tolist()

def download_agg_trades_chunk(symbol, start_ms, end_ms, session, min_val_usd=100):
    """Baixa um pedaço de trades com estratégia de baixo ruído (Low Noise)."""
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    trades = []
    curr = start_ms
    # 🚀 Limite de tentativas para evitar loop infinito em caso de erro
    max_requests_per_chunk = 50 
    req_count = 0
    
    while curr < end_ms and req_count < max_requests_per_chunk:
        params = {"symbol": symbol, "startTime": int(curr), "endTime": int(end_ms), "limit": 1000}
        req_count += 1
        try:
            # 🛡️ Delay aumentado para estabilidade total
            time.sleep(0.2) 
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if not data: break
                
                if data:
                    p = np.array([t['p'] for t in data], dtype=np.float32)
                    q = np.array([t['q'] for t in data], dtype=np.float32)
                    m = np.array([t['m'] for t in data], dtype=bool)
                    T = np.array([t['T'] for t in data], dtype=np.int64)
                    
                    mask = (p * q) >= min_val_usd
                    if np.any(mask):
                        chunk_data = np.column_stack((T[mask], p[mask], q[mask], m[mask]))
                        trades.append(chunk_data)
                
                curr = data[-1]['T'] + 1
            elif r.status_code in [418, 429]:
                # Pausa progressiva para limpar reputação de IP
                wait_time = 60
                print(f"\n[AVISO AggTrades] Rate limit detectado. Pausa de {wait_time}s para limpeza de IP...")
                time.sleep(wait_time)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(5)
    return trades

def download_agg_trades_parallel(symbol, start_ms, end_ms, min_val_usd=100):
    """Download sequencial puro (Monolithic Mode) para estabilidade absoluta."""
    print(f"[INFO] Baixando AggTrades (Monolithic Mode) para {symbol}...")
    
    all_chunks = []
    curr = start_ms
    total_trades = 0
    
    # Estratégia Industrial: Janelas de 1 Hora
    curr = start_ms
    hour_ms = 3600 * 1000
    
    with requests.Session() as session:
        while curr < end_ms:
            try:
                next_stop = min(curr + hour_ms, end_ms)
                temp_curr = curr
                
                # Loop de exaustão dentro da hora (Paginação por ID)
                last_id = None
                trades_in_hour = 0
                whales_in_hour = 0
                
                while True:
                    # 🛡️ Proteção contra Rate Limit
                    time.sleep(0.1) 
                    
                    # 1. Monta a URL
                    if last_id is None:
                        # Primeira requisição da hora (baseada em tempo)
                        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&startTime={temp_curr}&endTime={next_stop}&limit=1000"
                    else:
                        # Requisições subsequentes (baseadas em ID)
                        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&fromId={last_id + 1}&limit=1000"
                    
                    # 2. Faz a requisição
                    res = session.get(url, timeout=20)
                    
                    if res.status_code in [418, 429]:
                        print(f"\n[AVISO AggTrades] Rate limit detectado. Pausa de 30s para limpeza de IP...")
                        time.sleep(30)
                        continue # Tenta a mesma requisição novamente
                    
                    res_raw = res.json()
                    
                    if not isinstance(res_raw, list) or len(res_raw) == 0:
                        break # Fim dos trades para esta hora
                    
                    # 3. Filtra e processa
                    
                    # Se for uma requisição fromId, o primeiro trade pode estar fora da janela de tempo
                    # O fromId não respeita o endTime, então precisamos checar
                    if last_id is not None and int(res_raw[0]['T']) >= next_stop:
                        break
                        
                    chunk = []
                    for t in res_raw:
                        trade_ts = int(t['T'])
                        
                        # Se o trade for do futuro (próxima hora), ignoramos e paramos
                        if trade_ts >= next_stop:
                            break
                            
                        trades_in_hour += 1
                        val = float(t['p']) * float(t['q'])
                        
                        if val >= min_val_usd:
                            whales_in_hour += 1
                            chunk.append([trade_ts, float(t['p']), float(t['q']), 1 if t['m'] else 0])
                    
                    # 4. Atualiza o estado
                    if chunk:
                        all_chunks.append(np.array(chunk))
                        total_trades += len(chunk)
                    
                    last_id = int(res_raw[-1]['a']) # Pega o ID do último trade
                    
                    # Se o loop interno parou por causa do trade_ts >= next_stop
                    if trade_ts >= next_stop:
                        break
                        
                    # Se a requisição retornou menos que o limite, chegamos ao fim
                    if len(res_raw) < 1000:
                        break
                
                # Log de progresso a cada hora processada
                print(f"    [OK] {pd.to_datetime(curr, unit='ms')} -> {pd.to_datetime(next_stop, unit='ms')} | Trades Brutos: {trades_in_hour} | Baleias Capturadas: {whales_in_hour}")
                
                curr = next_stop
                
            except Exception as e:
                print(f"\n[!] Erro CRÍTICO na janela {pd.to_datetime(curr, unit='ms')}. Erro: {e}. Retentando em 10s...")
                time.sleep(10)
                continue
                
        if total_trades % 10000 < 1000:
            print(f"    - {total_trades} trades institucionais capturados...")
            
    if not all_chunks: return None
    print(f"[INFO] Total de {total_trades} trades institucionais baixados.")
    return np.vstack(all_chunks)

# -----------------------------------------------------------------------------
# PROCESSAMENTO E AGREGAÇÃO INTELIGENTE
# -----------------------------------------------------------------------------

def normalize_klines(rows):
    if not rows: return None
    df = pd.DataFrame(rows, columns=[
        "ts","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["ts","open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df[CONTRATO]

def processar_microestrutura_dinamica(df_target, trades_np, tf_str):
    """Calcula a microestrutura alinhando diretamente ao TF solicitado."""
    if trades_np is None or df_target is None:
        return df_target

    print(f"[INFO] Alinhando Microestrutura diretamente ao TF {tf_str}...")
    
    # Mapeamento de milissegundos por TF
    tf_ms_map = {
        '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000,
        '1h': 3600000, '2h': 7200000, '4h': 14400000, '6h': 21600000, '8h': 28800000,
        '12h': 43200000, '1d': 86400000
    }
    ms = tf_ms_map.get(tf_str, 60000)
    
    # trades_np: [0:T, 1:p, 2:q, 3:m]
    ts_target = (trades_np[:, 0] // ms) * ms
    q = trades_np[:, 2]
    is_sell = trades_np[:, 3].astype(bool)
    
    buy_vol = np.where(~is_sell, q, 0)
    sell_vol = np.where(is_sell, q, 0)
    delta = buy_vol - sell_vol
    
    df_t = pd.DataFrame({
        'ts': ts_target,
        'cum_delta': delta,
        'total_vol_agg': q,
        'buy_vol_agg': buy_vol,
        'sell_vol_agg': sell_vol
    })
    
    micro = df_t.groupby('ts').sum().reset_index()
    df_final = pd.merge(df_target, micro, on='ts', how='left').fillna(0)
    
    # Recalcular indicadores derivados
    if 'total_vol_agg' in df_final.columns:
        df_final['vpin'] = (df_final['buy_vol_agg'] - df_final['sell_vol_agg']).abs() / (df_final['total_vol_agg'] + 1e-9)
        df_final['price_range'] = (df_final['high'] - df_final['low']) / (df_final['close'] + 1e-9)
        df_final['absorcao'] = (df_final['total_vol_agg'] / (df_final['price_range'] + 1e-9)) * (1 - df_final['vpin'])
        
        # Z-Score da Absorção
        window = 100
        df_final['absorcao'] = (df_final['absorcao'] - df_final['absorcao'].rolling(window).mean()) / (df_final['absorcao'].rolling(window).std() + 1e-9)
        
    return df_final

def smart_resample(df_1m, target_tf):
    """Transforma dados de 1m em qualquer TF superior instantaneamente."""
    print(f"[INFO] Agregando 1m -> {target_tf}...")
    
    # Mapeamento de TFs para Pandas
    tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D'
    }
    rule = tf_map.get(target_tf, '15min')
    
    df = df_1m.copy()
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # Lógica de Agregação OHLCV + Microestrutura
    agg_logic = {
        'ts': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum',
        'close_time': 'last',
        'cum_delta': 'sum',
        'total_vol_agg': 'sum',
        'buy_vol_agg': 'sum',
        'sell_vol_agg': 'sum'
    }
    
    # Remove colunas de microestrutura se não existirem
    agg_logic = {k: v for k, v in agg_logic.items() if k in df.columns}
    
    df_res = df.resample(rule).agg(agg_logic).dropna()
    
    # Recalcular indicadores derivados no TF maior
    if 'total_vol_agg' in df_res.columns:
        df_res['vpin'] = (df_res['buy_vol_agg'] - df_res['sell_vol_agg']).abs() / (df_res['total_vol_agg'] + 1e-9)
        df_res['price_range'] = (df_res['high'] - df_res['low']) / (df_res['close'] + 1e-9)
        df_res['absorcao'] = (df_res['total_vol_agg'] / (df_res['price_range'] + 1e-9)) * (1 - df_res['vpin'])
        
        # Z-Score da Absorção (Janela adaptativa baseada no TF)
        window = 100
        df_res['absorcao'] = (df_res['absorcao'] - df_res['absorcao'].rolling(window).mean()) / (df_res['absorcao'].rolling(window).std() + 1e-9)
    
    return df_res.reset_index(drop=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("="*60)
    print("   DATA MANAGER INSTITUCIONAL ELITE V16 (WHALE-HUNTER)")
    print("   >>> High-Value Aggression Only <<<")
    print("="*60)

    simbolo = get_config("simbolo", "\nSímbolo (ex: BTCUSDT): ", "PENDLEUSDT").strip().upper()
    tfs_input = get_config("timeframes", "Lista de timeframes desejados (ex: 15m,30m,1h): ", "15m,30m,1h,4h,8h,1d").strip().lower()
    tfs = [x.strip() for x in tfs_input.split(',')]
    
    start_str = get_config("data_inicial", "Data inicial (YYYY-MM-DD): ", "2025-12-01").strip()
    end_str = get_config("data_final", "Data final   (YYYY-MM-DD): ", "2025-12-30").strip()
    
    # Pasta de saída: Render usa /data, Local usa input
    if RUNNING_ON_RENDER:
        pasta = CONFIG.get("pasta_saida", "/data/PENDLEUSDT_DATA")
        print(f"Pasta de saída: {pasta}")
    else:
        pasta = input("Pasta de saída: ").strip() or "./datasets"
    
    ativar_micro = get_config("ativar_micro", "Ativar Microestrutura (AggTrades)? (s/n): ", "s").strip().lower()
    
    # 🚀 FILTRO DE BALEIAS
    min_val = float(get_config("min_val_usd", "Valor mínimo do trade (USD) [Sugerido: 500]: ", "500"))

    start_ms = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end_str).timestamp() * 1000)

    if not os.path.isdir(pasta): os.makedirs(pasta)

    # 🚀 PASSO 1: DOWNLOAD DE AGRESSÃO (SE SOLICITADO)
    trades_np = None
    if ativar_micro == 's':
        print(f"\n>>> PASSO 1: Baixando Agressão Institucional (Whale Hunter Mode: >${min_val})")
        trades_np = download_agg_trades_parallel(simbolo, start_ms, end_ms + 60000, min_val_usd=min_val)
        if trades_np is None:
            print("    [AVISO] Nenhum trade encontrado com este filtro. Continuando apenas com OHLCV.")

    # 🚀 PASSO 2: DOWNLOAD E INJEÇÃO NOS TFS ESCOLHIDOS
    print(f"\n>>> PASSO 2: Processando Timeframes Escolhidos")
    for tf in tfs:
        print(f"\n[*] Baixando Klines {tf}...")
        rows = download_futures_klines(simbolo, tf, start_ms, end_ms)
        if not rows:
            print(f"    [ERRO] Falha ao baixar {tf}. Pulando.")
            continue
            
        df_final = normalize_klines(rows)
        
        # Injeta a agressão se os trades foram baixados
        if trades_np is not None:
            print(f"    [+] Injetando Agressão Institucional em {tf}...")
            df_final = processar_microestrutura_dinamica(df_final, trades_np, tf)
        
        # 🚀 SALVAMENTO OBRIGATÓRIO (FORA DE QUALQUER IF)
        saida_csv = os.path.join(pasta, f"{simbolo}_{tf}.csv")
        df_final.to_csv(saida_csv, index=False)
        
        # Confirmação das colunas para o usuário
        cols_finais = list(df_final.columns)
        tem_agressao = 'cum_delta' in cols_finais
        print(f"    [OK] Arquivo gerado: {saida_csv}")
        print(f"    [OK] Linhas: {len(df_final)} | Agressão Incluída: {'SIM ✅' if tem_agressao else 'NÃO ❌'}")

    print("\n" + "="*60)
    print("   DATA MANAGER V16 CONCLUÍDO COM SUCESSO!")
    print(f"   Timeframes processados: {tfs_input}")
    print(f"   Agressão incluída: {'SIM' if ativar_micro == 's' else 'NÃO'}")
    print("="*60)
if __name__ == "__main__":
    main()
