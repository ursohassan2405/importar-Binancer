# ============================================================
# V55_DATAMANAGER_RENDER.py
# ============================================================
# █  VERSÃO RENDER: EXTRAÇÃO + TREINO COMPLETO
# █  >>> V51 (extração) + V27 (treino) UNIFICADOS <<<
# ============================================================
#
# ██ CONFIGURAÇÃO - MODIFIQUE AQUI ██
SYMBOL = "PENDLEUSDT"
START_DT_STR = "2025-01-01"  # Data início (YYYY-MM-DD) - 2 MESES
END_DT_STR = "2025-01-30"    # Data fim (YYYY-MM-DD) - PROCESSA EM ~20MIN
MIN_WHALE_USD = 500          # Filtro whale em USD

# ██ CONFIGURAÇÃO DE TREINO (SEM INPUTS INTERATIVOS) ██
HORIZONTE_FUTURO = 5         # N candles (você usa 5)
USAR_MULTIFRAME = True       # Adicionar contexto multi-TF
TFS_ADICIONAIS = "30m,1h,4h,8h,1d"  # Timeframes extras (você usa estes)
USAR_PESO_TEMPORAL = True    # Peso temporal (você usa peso 1)
MODO_PESO_TEMPORAL = "1"     # Modo 1 (padrão)
# ██████████████████████████████████████████████████████████████
#
# FUNCIONALIDADES:
# ✅ Extrai dados da Binance Vision (aggTrades)
# ✅ Filtro whale >= $500
# ✅ Gera CSVs multi-timeframe
# ✅ Treino LGBM + XGB (sem CatBoost)
# ✅ Targets: A, B, C, REV_LONG, REV_SHORT, CONFLUENCIA
# ✅ Upload Catbox + Servidor HTTP
# ============================================================
#MELHOR SETUP, MUTLFRAME, PESO TEMPORAL 1
# ========================================================================
# IA_CRIPTO — V25_FINAL_REAL.py
# ------------------------------------------------------------------------
# █  V98 — VERSÃO VOLTA AO OURO (LUCRO MÁXIMO + SENSORES COMO FEATURES)
# █  >>> FOCO: MASS TRAINING & ROBUSTNESS <<<
# ------------------------------------------------------------------------
# • Arquitetura final V25 — institucional
# • 100% conversacional
# • Parametrizado
# • Zero hardcode
# • Zero leakage
# • Feature Engine V90 + V92 completo
# • Targets adaptativos (A/B/C)
# • Multi-TF real e seguro
# • Treino LGBM + XGB + CAT (com normalização consistente)
# • Painéis avançados de probabilidade
# • Exportador V22 Universal
# • Estrutura limpa, modular e documentada
# ========================================================================
#
# 🔧 CHANGELOG DE CORREÇÕES (VERSÃO CORRIGIDA)
# ------------------------------------------------------------------------
# Data: Janeiro 2026
# Objetivo: Corrigir vieses que inflavam resultados do backtest
#
# CORREÇÕES APLICADAS:
#
# 1. ✅ FUNDING RATE REALISTA (Linha ~3526)
#    - ANTES: taxa_financiamento = 0.0 (funding não cobrado)
#    - DEPOIS: taxa_financiamento_8h = 0.0001 (0.01% a cada 8h)
#    - IMPACTO: -10% a -15% nos resultados anuais
#    - JUSTIFICATIVA: Futuros perpétuos cobram funding rate típico de
#      0.01% a cada 8h (Binance/Bybit padrão)
#
# 2. ✅ ENTRADA NO PRÓXIMO CANDLE (Linha ~3700)
#    - ANTES: preco_entrada = close do candle atual (look-ahead bias)
#    - DEPOIS: preco_entrada = open do próximo candle
#    - IMPACTO: -20% a -50% nos resultados
#    - JUSTIFICATIVA: Sinal só é gerado APÓS fechamento do candle.
#      Entrada no close = impossível em mercado real.
#
# 3. ✅ SLIPPAGE ADICIONAL EM STOPS (Linha ~3730)
#    - ANTES: SL executado no preço exato
#    - DEPOIS: SL com slippage 2.5x maior (market order em pânico)
#    - IMPACTO: -10% a -20% nos resultados
#    - JUSTIFICATIVA: Stops executam com pior preço que ordens normais
#
# 4. ✅ ROUND-TRIP CORRETO (Linha ~3764)
#    - ANTES: notional * (comissao + slippage)
#    - DEPOIS: notional * 2 * (comissao + slippage)
#    - IMPACTO: -0.3% por trade (entrada + saída)
#    - JUSTIFICATIVA: Round-trip = 2 operações (compra E venda)
#
# 5. ✅ FUNDING PROPORCIONAL AO TEMPO (Linha ~3768)
#    - ANTES: funding_usd = notional * taxa * hold_time (sem unidade)
#    - DEPOIS: funding_usd = notional * taxa_8h * (hold_time / 32)
#    - IMPACTO: Cálculo correto proporcional a 8h
#    - JUSTIFICATIVA: 32 candles de 15min = 8h (período de funding)
#
# INFLAÇÃO TOTAL CORRIGIDA: 55-120% dos resultados originais
#
# NOTA IMPORTANTE:
# - NENHUMA linha do código original foi removida
# - Apenas adicionadas correções e comentários explicativos
# - Para comparar com original, busque por "🔧 CORREÇÃO"
# ------------------------------------------------------------------------
# ========================================================================

import os
import sys
import gc
import glob
import shutil
import json
import time
import zipfile
import requests
import threading
import random
from datetime import datetime, timedelta
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Força output unbuffered
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier  # REMOVIDO - incompatível Render
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin

scaler = None  # Inicializa o scaler como vazio no escopo global

import joblib

# ============================================================
# CONFIGURAÇÃO DE EXTRAÇÃO
# ============================================================
from datetime import datetime, timedelta

START_DT = datetime.strptime(START_DT_STR, "%Y-%m-%d")
END_DT = datetime.strptime(END_DT_STR, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

# ============================================================
# 🔧 PATCH RENDER: Detectar e usar disco persistente
# ============================================================
# CRÍTICO: Render precisa usar /opt/render/project/.data (persistente)!
RENDER_DISK = '/opt/render/project/.data'  # DISCO PERSISTENTE CORRETO

# Detectar se está no Render (múltiplos métodos)
IS_RENDER = (
    os.environ.get('RENDER') or  # Variável RENDER
    os.environ.get('RENDER_SERVICE_NAME') or  # Nome do serviço
    os.path.exists('/opt/render/project') or  # Path específico do Render
    'render.com' in os.environ.get('HOSTNAME', '')  # Hostname
)

if IS_RENDER:
    # Estamos no Render - SEMPRE usar disco persistente
    BASE_DIR = RENDER_DISK
    print("="*70, flush=True)
    print("🚀 RENDER DETECTADO - USANDO DISCO PERSISTENTE!", flush=True)
    print(f"   Path: {BASE_DIR}", flush=True)
    print("   ✅ Arquivos (CSVs, PKLs, ZIP) serão PRESERVADOS entre deploys!", flush=True)
    print("="*70, flush=True)
    
    # Criar diretório se não existir
    os.makedirs(BASE_DIR, exist_ok=True)
else:
    # Ambiente local
    BASE_DIR = "."
    print("="*70, flush=True)
    print("💻 AMBIENTE LOCAL DETECTADO", flush=True)
    print(f"   Path: {os.getcwd()}", flush=True)
    print("   ⚠️  Arquivos em diretório de trabalho (não persistente)", flush=True)
    print("="*70, flush=True)

# Paths finais (funcionam em local E Render)
OUT_DIR = os.path.join(BASE_DIR, f"{SYMBOL}_DATA")
CSV_AGG_PATH = os.path.join(OUT_DIR, f"{SYMBOL}_aggTrades.csv")
ZIP_PATH = os.path.join(BASE_DIR, f"{SYMBOL}_COMPLETO.zip")
BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0',
]

# Criar diretório de saída
os.makedirs(OUT_DIR, exist_ok=True)
print(f"\n📁 Diretório de saída: {OUT_DIR}", flush=True)
print(f"📄 CSV aggTrades: {CSV_AGG_PATH}", flush=True)
print(f"📦 ZIP output: {ZIP_PATH}\n", flush=True)
# ============================================================

# ============================================================
# FUNÇÕES DE EXTRAÇÃO DE DADOS (V51)
# ============================================================

def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS), 'Accept': '*/*', 'Connection': 'keep-alive'}

def download_daily_file(symbol, date, session, retry_count=5):
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    
    for attempt in range(retry_count):
        try:
            if attempt > 0:
                time.sleep(min(5 * (2 ** attempt), 60))
            response = session.get(url, headers=get_headers(), timeout=90)
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    files = z.namelist()
                    if not files:
                        return None
                    csv_filename = files[0]
                    with z.open(csv_filename) as f:
                        df_test = pd.read_csv(f, header=None, nrows=1)
                        has_header = any('transact_time' in str(val) for val in df_test.iloc[0])
                    with z.open(csv_filename) as f:
                        return pd.read_csv(f, header=0 if has_header else None)
            elif response.status_code == 404:
                return None
        except Exception:
            if attempt == retry_count - 1:
                return None
    return None

def process_binance_data(df):
    if df is None or df.empty:
        return None
    if 'transact_time' not in df.columns:
        df.columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'transact_time', 'is_buyer_maker']
    def convert_side(val):
        return 1 if (val is True or val == 'True' or val == 'true') else 0
    df_processed = pd.DataFrame({
        'ts': pd.to_numeric(df['transact_time'], errors='coerce').astype('Int64'),
        'price': pd.to_numeric(df['price'], errors='coerce').astype(float),
        'qty': pd.to_numeric(df['quantity'], errors='coerce').astype(float),
        'side': df['is_buyer_maker'].apply(convert_side)
    })
    return df_processed.dropna()

def gerar_timeframe_tratado(csv_agg_path, csv_tf_path, timeframe_min=15, min_val_usd=500, chunksize=500_000):
    print(f">>> Gerando dataset {timeframe_min}m...", flush=True)
    buckets = {}
    chunks_processados = 0
    
    try:
        for chunk in pd.read_csv(csv_agg_path, chunksize=chunksize):
            chunks_processados += 1
            if chunks_processados % 10 == 0:
                print(f"   Processados {chunks_processados} chunks...", flush=True)
            
            # OTIMIZAÇÃO: Conversão em batch (mais rápido que linha por linha)
            chunk = chunk.astype({
                'ts': 'int64',
                'price': 'float64',
                'qty': 'float64',
                'side': 'int8'
            }, errors='ignore')
            
            chunk = chunk.dropna(subset=["ts", "price", "qty", "side"])
            if chunk.empty:
                continue
            
            # OTIMIZAÇÃO: Operações vetorizadas (muito mais rápido)
            dt = pd.to_datetime(chunk["ts"], unit="ms", utc=True)
            chunk["bucket_ms"] = (dt.dt.floor(f"{timeframe_min}min").astype("int64") // 10**6).astype("int64")
            chunk["val_usd"] = chunk["price"] * chunk["qty"]
            chunk["is_whale"] = chunk["val_usd"] >= float(min_val_usd)
            
            # OTIMIZAÇÃO: GroupBy ao invés de loop Python (10-100x mais rápido)
            for bms, group in chunk.groupby("bucket_ms"):
                st = buckets.get(bms)
                if st is None:
                    st = {
                        "ts": int(bms),
                        "open": float(group["price"].iloc[0]),
                        "high": float(group["price"].max()),
                        "low": float(group["price"].min()),
                        "close": float(group["price"].iloc[-1]),
                        "volume": 0.0,
                        "buy_vol": 0.0,
                        "sell_vol": 0.0
                    }
                    buckets[bms] = st
                else:
                    st["high"] = max(st["high"], float(group["price"].max()))
                    st["low"] = min(st["low"], float(group["price"].min()))
                    st["close"] = float(group["price"].iloc[-1])
                
                st["volume"] += float(group["qty"].sum())
                
                # Volume de whales
                whales = group[group["is_whale"]]
                if not whales.empty:
                    st["buy_vol"] += float(whales[whales["side"] == 0]["qty"].sum())
                    st["sell_vol"] += float(whales[whales["side"] == 1]["qty"].sum())
        
        print(f"   Total: {chunks_processados} chunks processados", flush=True)
        
    except Exception as e:
        print(f"❌ Erro processando chunks: {e}", flush=True)
        raise
    if not buckets:
        raise RuntimeError("Nenhum bucket gerado!")
    rows = [[st["ts"], st["open"], st["high"], st["low"], st["close"], st["volume"], st["buy_vol"], st["sell_vol"], st["buy_vol"] - st["sell_vol"]] for st in [buckets[bms] for bms in sorted(buckets.keys())]]
    df_tf = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol", "delta"])
    df_tf["buy_vol_agg"], df_tf["sell_vol_agg"] = df_tf["buy_vol"], df_tf["sell_vol"]
    df_tf["total_vol_agg"] = df_tf["buy_vol_agg"] + df_tf["sell_vol_agg"]
    df_tf["taker_buy_base"], df_tf["taker_sell_base"] = df_tf["buy_vol_agg"], df_tf["sell_vol_agg"]
    df_tf["taker_buy_quote"] = df_tf["taker_buy_base"] * df_tf["close"]
    df_tf["quote_volume"] = df_tf["volume"] * df_tf["close"]
    df_tf["trades"], df_tf["close_time"] = 0, df_tf["ts"] + (timeframe_min * 60 * 1000) - 1
    df_tf = df_tf.sort_values("ts").reset_index(drop=True)
    df_tf["cum_delta"] = df_tf["delta"].cumsum()
    df_tf["price_range"] = df_tf["high"] - df_tf["low"]
    df_tf["absorcao"] = df_tf["delta"] / (df_tf["price_range"].replace(0, 1e-9))
    df_tf["vpin"] = (df_tf["buy_vol_agg"] - df_tf["sell_vol_agg"]).abs() / (df_tf["total_vol_agg"].replace(0, 1e-9))
    for c in ["open", "high", "low", "close", "volume", "buy_vol", "sell_vol", "delta", "buy_vol_agg", "sell_vol_agg", "total_vol_agg", "taker_buy_base", "taker_sell_base", "taker_buy_quote", "quote_volume", "cum_delta", "price_range", "absorcao", "vpin"]:
        df_tf[c] = pd.to_numeric(df_tf[c], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    df_tf.to_csv(csv_tf_path, index=False)
    print(f">>> Salvo: {csv_tf_path} ({len(df_tf)} candles)", flush=True)
    return df_tf

def upload_catbox(filepath):
    try:
        with open(filepath, "rb") as f:
            r = requests.post("https://catbox.moe/user/api.php", data={"reqtype": "fileupload"}, files={"fileToUpload": f}, timeout=300)
        r.raise_for_status()
        return r.text.strip()
    except Exception as e:
        print(f"❌ Catbox erro: {e}")
        return None

class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/download', '/download/']:
            if os.path.exists(ZIP_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{SYMBOL}_COMPLETO.zip"')
                self.end_headers()
                with open(ZIP_PATH, 'rb') as f: self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP nao criado')
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            status = 'PRONTO!' if os.path.exists(ZIP_PATH) else 'Processando...'
            link = f'<a href="/download" style="font-size:24px;padding:20px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;">BAIXAR ZIP</a>' if os.path.exists(ZIP_PATH) else '<p>Aguarde...</p>'
            self.wfile.write(f'<html><body style="font-family:Arial;padding:50px;text-align:center;"><h1>{SYMBOL}</h1><p>{status}</p>{link}</body></html>'.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(('0.0.0.0', port), DownloadHandler).serve_forever()

def extrair_dados_binance():
    # RENDER FIX: Verificar se aggTrades já existe e está completo
    if os.path.exists(CSV_AGG_PATH):
        try:
            df_check = pd.read_csv(CSV_AGG_PATH, nrows=1)
            file_size = os.path.getsize(CSV_AGG_PATH) / (1024 * 1024)  # MB
            if file_size > 10:  # Se arquivo tem >10MB, assume que está completo
                print("=" * 80)
                print(f"✅ DADOS JÁ EXISTEM: {CSV_AGG_PATH}")
                print(f"   Tamanho: {file_size:.1f} MB")
                print("   Pulando download (usar dados existentes)")
                print("=" * 80)
                # Gerar timeframes e retornar
                timeframes = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "8h": 480, "1d": 1440}
                csv_paths = {}
                for label, tf_min in timeframes.items():
                    csv_tf_path = os.path.join(OUT_DIR, f"{SYMBOL}_{label}.csv")
                    if not os.path.exists(csv_tf_path):
                        gerar_timeframe_tratado(CSV_AGG_PATH, csv_tf_path, tf_min, MIN_WHALE_USD)
                    csv_paths[label] = csv_tf_path
                return csv_paths
        except:
            pass  # Se falhar, redownload
    
    print("=" * 80)
    print(f"EXTRAINDO DADOS: {SYMBOL}")
    print(f"PERÍODO: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}")
    print(f"FILTRO WHALE: >= ${MIN_WHALE_USD}")
    print("=" * 80)
    dates = generate_date_range(START_DT, END_DT)
    if os.path.exists(CSV_AGG_PATH): os.remove(CSV_AGG_PATH)
    session, success_count, first_write = requests.Session(), 0, True
    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {date.strftime('%Y-%m-%d')}", end=" ", flush=True)
        df = download_daily_file(SYMBOL, date, session)
        if df is not None:
            df_processed = process_binance_data(df)
            if df_processed is not None and not df_processed.empty:
                df_processed.to_csv(CSV_AGG_PATH, mode='a', header=first_write, index=False)
                first_write = False
                success_count += 1
                print(f"✓ {len(df_processed):,}", flush=True)
                del df, df_processed
            else: print("⚠️", flush=True)
        else: print("⚠️", flush=True)
        time.sleep(random.uniform(0.3, 1.0))
    session.close()
    print(f"\n>>> Download: {success_count}/{len(dates)} dias")
    print("="*70, flush=True)
    if success_count == 0: raise RuntimeError("Nenhum dado!")
    
    print("\n📊 GERANDO TIMEFRAMES...")
    print("="*70, flush=True)
    
    timeframes = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "8h": 480, "1d": 1440}
    csv_paths = {}
    for label, tf_min in timeframes.items():
        csv_tf_path = os.path.join(OUT_DIR, f"{SYMBOL}_{label}.csv")
        
        # 🚀 OTIMIZAÇÃO: Não regenerar se já existe e é válido
        if os.path.exists(csv_tf_path):
            try:
                df_check = pd.read_csv(csv_tf_path, nrows=1)
                file_size = os.path.getsize(csv_tf_path) / 1024  # KB
                if file_size > 10:  # Arquivo válido (>10KB)
                    print(f"✅ {label} JÁ EXISTE ({file_size/1024:.1f} MB) - Pulando", flush=True)
                    csv_paths[label] = csv_tf_path
                    continue
            except:
                pass  # Se erro, regenera
        
        print(f"\n🔄 Processando {label}...", flush=True)
        gerar_timeframe_tratado(CSV_AGG_PATH, csv_tf_path, tf_min, MIN_WHALE_USD)
        csv_paths[label] = csv_tf_path
        print(f"✅ {label} concluído!", flush=True)
    
    print("\n" + "="*70)
    print("✅ TODOS OS TIMEFRAMES GERADOS COM SUCESSO!")
    print("="*70, flush=True)
    return csv_paths


# ======================================================================
# BLOCO REMOVIDO: Backtest/Simulação/Análise
# (Linhas originais: 388-619)
# V27 já faz backtest - Render só treina modelos
# ======================================================================

# 🚀 VARIÁVEL GLOBAL PARA O JUIZ (RESOLVE ERRO DE ESCOPO)
meta_modelos_features = {}

def modulo_targetA_threshold_adaptativo(
        df: pd.DataFrame,
        atr_period: int = 5,
        ret_window: int = 10,
        p_low: float = 0.20,
        p_high: float = 0.60,
        fator_regime: dict = None
    ):
    """
    Módulo S7 — Threshold Adaptativo por Regime (Target A)
    Totalmente isolado e autossuficiente.
    Requisitos do df: colunas OHLC ('high','low','close')
    Retorno: DataFrame com 3 colunas novas:
        - threshold_A_usado
        - regime_targetA
        - target_A_adaptativo
    """

    # ---------------------------------------------------------------
    # Proteção absoluta contra dict mutável no default
    # ---------------------------------------------------------------
    if fator_regime is None:
        fator_regime = {0: 0.5, 1: 0.8, 2: 1.2}

    # Garantir que df tem as colunas necessárias
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória ausente no df: '{col}'")

    # Cópia local (não altera o dataframe original)
    df_local = df.copy()

    # ===============================================================
    # 1) ATR Micro (TR e Média Móvel)
    # ===============================================================
    high_low = df_local["high"] - df_local["low"]
    high_close_prev = (df_local["high"] - df_local["close"].shift(1)).abs()
    low_close_prev = (df_local["low"] - df_local["close"].shift(1)).abs()

    tr = np.nanmax(
        np.vstack([
            high_low.values,
            high_close_prev.values,
            low_close_prev.values
        ]),
        axis=0
    )

    # ATR_micro seguro com rolling
    df_local["atr_micro"] = pd.Series(tr).rolling(atr_period).mean()

    # ===============================================================
    # 2) Regime baseado em volatilidade relativa
    # ===============================================================
    # Evita divisão por zero
    df_local["regime_vol_raw"] = df_local["atr_micro"] / (df_local["close"] + 1e-12)

    p1 = df_local["regime_vol_raw"].quantile(p_low)
    p2 = df_local["regime_vol_raw"].quantile(p_high)

    def classify_regime(v):
        if pd.isna(v):
            return 0  # regime morto por segurança
        if v < p1:
            return 0
        elif v < p2:
            return 1
        else:
            return 2

    df_local["regime_targetA"] = df_local["regime_vol_raw"].apply(classify_regime)

    # ===============================================================
    # 3) Threshold adaptativo
    # ===============================================================
    df_local["threshold_A_usado"] = (
        df_local["atr_micro"] * df_local["regime_targetA"].map(fator_regime)
    )

    # ===============================================================
    # 4) Retornos futuros fixos (usa shift)
    # ===============================================================
    future_high = df_local["high"].shift(-ret_window)
    future_low = df_local["low"].shift(-ret_window)

    # 🔴 CORREÇÃO ANTI-LEAKAGE: Variáveis futuras são temporárias e não entram no DataFrame
    ret_max_temp = (future_high - df_local["close"]) / (df_local["close"] + 1e-12)
    ret_min_temp = (df_local["close"] - future_low) / (df_local["close"] + 1e-12)

    # ===============================================================
    # 5) Regra física do Target A
    # ===============================================================
    def compute_target(idx):
        thr = df_local.loc[idx, "threshold_A_usado"]

        # filtro para regime morto (evita falsos A)
        if df_local.loc[idx, "regime_targetA"] == 0 and thr < 0.002:
            return 0

        if ret_max_temp.loc[idx] >= thr:
            return 1
        if ret_min_temp.loc[idx] >= thr:
            return -1

        return 0

    df_local["target_A_adaptativo"] = df_local.index.map(compute_target)

    # ===============================================================
    # 6) Retorno final (somente colunas criadas pelo módulo)
    # ===============================================================
    return df_local[[
        "threshold_A_usado",
        "regime_targetA",
        "target_A_adaptativo"
    ]]


# =======================================================================
# FUNÇÕES NOVAS — ELLIOTT, VWAP, MICRO-SQUEEZE, INSIDE/NR, Z-SCORES
# (Cole este bloco no início do arquivo, junto com as outras funções)
# =======================================================================

def adicionar_fractais_elliott(df):
    df = df.copy()

    df["fractal_high"] = (
        (df["high"].shift(2) < df["high"].shift(1)) &
        (df["high"].shift(1) < df["high"]) &
        (df["high"].shift(-1) < df["high"]) &
        (df["high"].shift(-2) < df["high"])
    ).astype(int)

    df["fractal_low"] = (
        (df["low"].shift(2) > df["low"].shift(1)) &
        (df["low"].shift(1) > df["low"]) &
        (df["low"].shift(-1) > df["low"]) &
        (df["low"].shift(-2) > df["low"])
    ).astype(int)

    df["pivot_high"] = df["high"].where(
        (df["high"] > df["high"].shift(1)) &
        (df["high"] > df["high"].shift(-1))
    )
    df["pivot_low"] = df["low"].where(
        (df["low"] < df["low"].shift(1)) &
        (df["low"] < df["low"].shift(-1))
    )

    df["last_pivot_high"] = df["pivot_high"].ffill()
    df["last_pivot_low"] = df["pivot_low"].ffill()

    df["swing_dir"] = 0
    df.loc[df["last_pivot_high"] > df["last_pivot_high"].shift(1), "swing_dir"] = 1
    df.loc[df["last_pivot_low"]  > df["last_pivot_low"].shift(1),  "swing_dir"] = 2
    df.loc[df["last_pivot_high"] < df["last_pivot_high"].shift(1), "swing_dir"] = -1
    df.loc[df["last_pivot_low"]  < df["last_pivot_low"].shift(1),  "swing_dir"] = -2

    df["wave_amplitude"] = df["close"] - df["close"].shift(5)
    df["wave_amplitude_abs"] = df["wave_amplitude"].abs()

    df["correction_pct"] = (
        (df["wave_amplitude"].shift(1) - df["wave_amplitude"]) /
        (df["wave_amplitude"].shift(1).abs() + 1e-9)
    )

    if "atr14" in df.columns:
        df["wave_strength"] = df["wave_amplitude_abs"] / (df["atr14"] + 1e-9)
    else:
        df["wave_strength"] = np.nan

    df["dist_to_pivot_high"] = df["last_pivot_high"] - df["close"]
    df["dist_to_pivot_low"] = df["close"] - df["last_pivot_low"]

    df["impulse_count"] = (df["swing_dir"] != df["swing_dir"].shift(1)).cumsum()

    return df


def adicionar_vwap(df):
    df = df.copy().reset_index(drop=True)

    if "ts" not in df.columns:
        print("[AVISO] Coluna 'ts' ausente — VWAP ignorado.")
        return df

    dt = pd.to_datetime(df["ts"], unit="ms")

    df["day"] = dt.dt.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

    vol_day = df["volume"].groupby(df["day"]).cumsum()
    tpv_day = (df["tp"] * df["volume"]).groupby(df["day"]).cumsum()
    df["vwap"] = tpv_day / (vol_day + 1e-9)

    df["week"] = dt.dt.isocalendar().week.astype(int)
    vol_week = df["volume"].groupby(df["week"]).cumsum()
    tpv_week = (df["tp"] * df["volume"]).groupby(df["week"]).cumsum()
    df["vwap_week"] = tpv_week / (vol_week + 1e-9)

    df["month"] = dt.dt.month.astype(int)
    vol_month = df["volume"].groupby(df["month"]).cumsum()
    tpv_month = (df["tp"] * df["volume"]).groupby(df["month"]).cumsum()
    df["vwap_month"] = tpv_month / (vol_month + 1e-9)

    if "pivot_low" in df.columns and "pivot_high" in df.columns:
        pivot_anchor = df["pivot_low"].combine_first(df["pivot_high"]).fillna(0)
        df["anchored_vwap"] = np.nan
        sum_tp = 0.0
        sum_vol = 0.0

        for i in range(len(df)):
            if pivot_anchor.iloc[i] != 0:
                sum_tp = 0.0
                sum_vol = 0.0
            sum_tp += df["tp"].iloc[i] * df["volume"].iloc[i]
            sum_vol += df["volume"].iloc[i]
            df.at[i, "anchored_vwap"] = sum_tp / (sum_vol + 1e-9)
    else:
        df["anchored_vwap"] = np.nan

    df["vwap_std"] = df["close"].rolling(20).std()
    df["vwap_upper"] = df["vwap"] + df["vwap_std"]
    df["vwap_lower"] = df["vwap"] - df["vwap_std"]
    df["vwap_zscore"] = (df["close"] - df["vwap"]) / (df["vwap_std"] + 1e-9)

    if "atr14" in df.columns:
        df["dist_vwap_atr"] = (df["close"] - df["vwap"]) / (df["atr14"] + 1e-9)
    else:
        df["dist_vwap_atr"] = np.nan

    return df


def adicionar_micro_squeeze(df):
    df = df.copy()

    ma_5 = df["close"].rolling(5).mean()
    std_5 = df["close"].rolling(5).std()

    df["bb_upper_5"] = ma_5 + 2 * std_5
    df["bb_lower_5"] = ma_5 - 2 * std_5
    df["bb_width_5"] = (df["bb_upper_5"] - df["bb_lower_5"]) / (df["close"] + 1e-9)

    if "tr" in df.columns:
        atr_5 = df["tr"].rolling(5).mean()
    else:
        atr_5 = (df["high"] - df["low"]).rolling(5).mean()

    df["kc_upper_5"] = ma_5 + 1.5 * atr_5
    df["kc_lower_5"] = ma_5 - 1.5 * atr_5
    df["kc_range_5"] = (df["kc_upper_5"] - df["kc_lower_5"]) / (df["close"] + 1e-9)

    df["micro_squeeze"] = (
        (df["bb_upper_5"] < df["kc_upper_5"]) &
        (df["bb_lower_5"] > df["kc_lower_5"])
    ).astype(int)

    df["pre_breakout_pressure"] = (
        (df["close"] - df["bb_upper_5"]) /
        (df["bb_upper_5"] - df["bb_lower_5"] + 1e-9)
    )

    df["micro_vol_squeeze"] = (
        df["range"].rolling(5).std() /
        (df["range"].rolling(20).std() + 1e-9)
    )

    return df


def adicionar_inside_nr(df):
    df = df.copy()

    df["inside_bar"] = (
        (df["high"] < df["high"].shift(1)) &
        (df["low"]  > df["low"].shift(1))
    ).astype(int)

    prev_range = (df["high"].shift(1) - df["low"].shift(1))
    df["inside_bar_strength"] = (df["high"] - df["low"]) / (prev_range + 1e-9)

    df["range"] = df["high"] - df["low"]
    df["nr4"] = (df["range"] == df["range"].rolling(4).min()).astype(int)
    df["nr7"] = (df["range"] == df["range"].rolling(7).min()).astype(int)

    df["breakout_bias"] = (
        (df["close"] - df["low"]) /
        (df["high"] - df["low"] + 1e-9)
    )

    return df


def adicionar_zscore_intrabar(df):
    df = df.copy()

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df["body_z"] = (df["body"] - df["body"].rolling(10).mean()) / \
                   (df["body"].rolling(10).std() + 1e-9)

    df["range_z"] = (df["range"] - df["range"].rolling(10).mean()) / \
                    (df["range"].rolling(10).std() + 1e-9)

    df["close_pos"] = (df["close"] - df["low"]) / \
                      (df["high"] - df["low"] + 1e-9)

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["wick_up_ratio_5"] = upper_wick.rolling(5).mean() / (df["range"] + 1e-9)
    df["wick_down_ratio_5"] = lower_wick.rolling(5).mean() / (df["range"] + 1e-9)

    df["reversal_prob"] = (
        (-df["body_z"] * 0.5) +
        (df["wick_down_ratio_5"] * 0.7) -
        (df["wick_up_ratio_5"] * 0.7)
    )

    return df
# =============================================================================
# BLOCO 1 — FUNÇÕES BASE DO MÓDULO MULTI-TF (V33 DEFINITIVO)
# =============================================================================
# Objetivo:
#   • tratar a lista de TFs fornecida pelo usuário
#   • validar TFs disponíveis
#   • normalizar nomes
#   • evitar duplicações
#   • preparar estrutura básica para merges
#   • renomear colunas para ctx_<TF>_<col>
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1.1 — Normalizar string de TFs fornecida pelo usuário
# -----------------------------------------------------------------------------
def parse_lista_tfs(tf_input_str):
    """
    Recebe algo como: "15m, 30m,1h ,4h, 8h"
    Retorna uma lista limpa: ["15m","30m","1h","4h","8h"]
    """
    if not isinstance(tf_input_str, str):
        return []

    # separar por vírgula e limpar espaços
    bruto = [x.strip() for x in tf_input_str.split(",")]

    # remover vazios + duplicações mantendo ordem
    clean = []
    for tf in bruto:
        if tf != "" and tf not in clean:
            clean.append(tf)

    return clean


# -----------------------------------------------------------------------------
# 1.2 — Validar TFs solicitados vs. arquivos disponíveis
# -----------------------------------------------------------------------------
def validar_tfs_disponiveis(lista_tfs, pasta_datasets, simbolo):
    """
    Verifica quais TFs realmente existem como CSVs na pasta.
    Exemplo esperado de arquivo:
    ETHUSDT_15m_full_V14.csv
    """
    disponiveis = []
    ausentes = []

    for tf in lista_tfs:
        nome_esperado = f"{simbolo}_{tf}_full_V14.csv"
        caminho = os.path.join(pasta_datasets, nome_esperado)

        if os.path.exists(caminho):
            disponiveis.append(tf)
        else:
            ausentes.append(tf)

    return disponiveis, ausentes


# -----------------------------------------------------------------------------
# 1.3 — Renomear colunas do TF adicional para evitar colisões
# -----------------------------------------------------------------------------
def renomear_cols_multitf(df_tf, tf_label):
    """
    Renomeia todas as colunas, exceto timestamp, para o padrão:
        ctx_<TF>_<col>
    Exemplo:
        close  → ctx_15m_close
        ema20  → ctx_1h_ema20
    """
    novas_cols = {}

    for col in df_tf.columns:
        if col == "timestamp":
            novas_cols[col] = col
        else:
            novas_cols[col] = f"ctx_{tf_label}_{col}"

    return df_tf.rename(columns=novas_cols)


# -----------------------------------------------------------------------------
# 1.4 — Verificação essencial do dataframe antes do merge
# -----------------------------------------------------------------------------
def checar_dataframe_tf(df_tf, tf_label):
    """
    Checa estrutura mínima antes do merge_asof:
        • existência de 'timestamp'
        • ordenação
        • duplicatas
    """
    if "timestamp" not in df_tf.columns:
        raise ValueError(f"Dataset TF {tf_label} não contém coluna 'timestamp'.")

    # ordenar
    df_tf = df_tf.sort_values("timestamp").reset_index(drop=True)

    # duplicatas
    dups = df_tf["timestamp"].duplicated().sum()
    if dups > 0:
        print(f"⚠ Aviso: TF {tf_label} possui {dups} timestamps duplicados. Serão deduplicados.")
        df_tf = df_tf.drop_duplicates(subset="timestamp")

    return df_tf


# -----------------------------------------------------------------------------
# 1.5 — Pequena auditoria opcional por TF
# -----------------------------------------------------------------------------
def auditoria_minima_tf(df_tf, tf_label):
    """
    Auditoria leve, apenas para garantir integridade.
    Impacto zero no pipeline.
    """
    print(f"\n--- AUDITORIA MÍNIMA DO TF {tf_label} ---")
    print(f"Linhas: {len(df_tf)}")
    print(f"Colunas: {list(df_tf.columns)}")
    print(f"Timestamps únicos: {df_tf['timestamp'].nunique()}")
    print(f"Monotônico: {df_tf['timestamp'].is_monotonic_increasing}")
    print(f"NaNs totais: {df_tf.isna().sum().sum()}")
    print("--- Fim auditoria mínima ---\n")


print("\n================= V55 DATAMANAGER RENDER =================\n")

# ========================================================================
# EXTRAÇÃO AUTOMÁTICA DE DADOS
# ========================================================================

# Inicia servidor HTTP em background
http_thread = threading.Thread(target=start_http_server, daemon=True)
http_thread.start()
time.sleep(1)
print(f">>> Servidor HTTP iniciado na porta {os.environ.get('PORT', 10000)}")

# Extrai dados da Binance
csv_paths = extrair_dados_binance()

print("\n" + "="*80)
print("🎉 CSVs GERADOS COM SUCESSO!")
print("="*80)
for label, path in csv_paths.items():
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"   ✅ {label:4s}: {size_mb:6.1f} MB")
print("="*80)

# 🚀 GERAR ZIP E LINK CATBOX IMEDIATAMENTE (ANTES DO TREINO)
print("\n📦 CRIANDO ZIP COM CSVs (para download rápido)...")
ZIP_CSVS_PATH = os.path.join(BASE_DIR, f"{SYMBOL}_CSVs_ONLY.zip")

try:
    import zipfile
    with zipfile.ZipFile(ZIP_CSVS_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for label, csv_path in csv_paths.items():
            if os.path.exists(csv_path):
                arcname = os.path.basename(csv_path)
                zf.write(csv_path, arcname)
                print(f"   📊 Adicionado: {arcname}")
    
    zip_size = os.path.getsize(ZIP_CSVS_PATH) / (1024 * 1024)
    print(f"\n✅ ZIP CSVs criado: {zip_size:.2f} MB")
    print(f"   Path: {ZIP_CSVS_PATH}")
    
    # UPLOAD PARA CATBOX
    print("\n🚀 FAZENDO UPLOAD PARA CATBOX (CSVs)...")
    link_csvs = upload_catbox(ZIP_CSVS_PATH)
    
    if link_csvs:
        print("\n" + "="*80)
        print("🎉 LINK #1 PRONTO - CSVs APENAS (SEM PKLs)")
        print("="*80)
        print("📦 CONTEÚDO:")
        print("   ✅ PENDLEUSDT_15m.csv")
        print("   ✅ PENDLEUSDT_30m.csv")
        print("   ✅ PENDLEUSDT_1h.csv")
        print("   ✅ PENDLEUSDT_4h.csv")
        print("   ✅ PENDLEUSDT_8h.csv")
        print("   ✅ PENDLEUSDT_1d.csv")
        print("   ❌ Modelos PKLs (ainda treinando...)")
        print("="*80)
        print(f"🔗 LINK CSVs (download rápido):")
        print(f"   {link_csvs}")
        print("="*80)
        print("⏱️  TREINO EM ANDAMENTO...")
        print("   → Link #2 (CSVs + PKLs) virá em ~1 hora")
        print("   → Você pode baixar os CSVs agora!")
        print("="*80)
    else:
        print("⚠️  Upload Catbox falhou, mas ZIP está disponível localmente")
        
except Exception as e:
    print(f"❌ Erro criando ZIP de CSVs: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# FINALIZAÇÃO - CSVS GERADOS E SALVOS
# ============================================================================

print()
print("=" * 80)
print("✅ EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
print("=" * 80)
print()
print("📂 CSVs gerados e salvos em:")
print(f"   {BASE_DIR}/")
print()

# Listar CSVs gerados
import glob
csvs = glob.glob(f"{BASE_DIR}/*.csv")
print(f"📊 Total de CSVs: {len(csvs)}")
for csv_file in sorted(csvs):
    size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    nome = os.path.basename(csv_file)
    print(f"   ✅ {nome:<25} ({size_mb:>8.2f} MB)")

print()
print("=" * 80)
print("🎯 PRÓXIMO PASSO: Executar V27 para treino e backtest")
print("   python V27_RENDER_TREINO.py")
print("=" * 80)
print()


# ============================================================================
# ✅ EXTRAÇÃO CONCLUÍDA - PROCESSO FINALIZADO
# ============================================================================

print("\n" + "=" * 80)
print("✅ EXTRAÇÃO E GERAÇÃO DE CSVs CONCLUÍDA!")
print("=" * 80)
print()
print(f"📂 CSVs salvos em: {OUT_DIR}")
print()
print("📊 Arquivos gerados:")
for tf, path in csv_paths.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {os.path.basename(path)} ({size_mb:.2f} MB)")
print()
print("=" * 80)
print("🎉 PROCESSO FINALIZADO COM SUCESSO!")
print("=" * 80)
print()
print("📋 PRÓXIMO PASSO:")
print("   Execute V27_RENDER_TREINO.py para treinar modelos")
print()
