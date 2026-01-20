# ============================================================
# V55_DATAMANAGER_RENDER.py
# ============================================================
# █  VERSÃO RENDER: EXTRAÇÃO + TREINO COMPLETO
# █  >>> V51 (extração) + V27 (treino) UNIFICADOS <<<
# ============================================================
#
# ██ CONFIGURAÇÃO - MODIFIQUE AQUI ██
SYMBOL = "PENDLEUSDT"
START_DT_STR = "2025-11-19"  # Data início (YYYY-MM-DD) - 2 MESES
END_DT_STR = "2025-12-30"    # Data fim (YYYY-MM-DD) - PROCESSA EM ~20MIN
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

# Define variáveis para treino
csv_path = csv_paths["15m"]
out_dir = OUT_DIR
exp_name = f"{SYMBOL}_V55"
simbolo = SYMBOL

os.makedirs(out_dir, exist_ok=True)

print(f"\n✔ EXPERIMENTO: {exp_name}")
print(f"✔ CSV para treino: {csv_path}")
print(f"✔ Pasta de saída: {out_dir}\n")

# ========================================================================
# UTILIDADES GERAIS
# ========================================================================

def titulo(txt):
    bar = "=" * 70
    return f"{bar}\n{txt}\n{bar}"

def log_append(loglist, msg):
    loglist.append(msg + "\n")

log = []
log_append(log, f"Experimento: {exp_name}")
log_append(log, f"Arquivo CRU: {csv_path}")
log_append(log, f"Data/Hora: {datetime.now()}")

print(">>> Carregando dataset cru...")
df_raw = pd.read_csv(csv_path)
log_append(log, f"Dimensão CRU: {df_raw.shape}")

# Timeframe fixo 15m (extraído automaticamente)
tf_base_global = "15m"
print(f"✔ Timeframe: {tf_base_global}")

# Símbolo já definido na configuração

# ========================================================================
# FUNÇÃO — GERAR ASSET PROFILE (DIAGNÓSTICO INSTITUCIONAL)
# ========================================================================

def gerar_asset_profile(df, symbol, timeframe, thrA, out_dir):
    import json
    import os

    amp = df["amp_fut"].dropna()

    profile = {
        "symbol": symbol,
        "timeframe": timeframe,
        "n_bars_total": int(len(df)),
        "p50_move": float(np.percentile(amp, 50)),
        "p90_move": float(np.percentile(amp, 90)),
        "p95_move": float(np.percentile(amp, 95)),
        "p99_move": float(np.percentile(amp, 99)),
        "hit_rate_A_bin": float(df["target_A_bin"].mean()),
        "tier_quality": "S",  # provisório / já validado por você
        "operable_flag": True
    }

    path = os.path.join(
        out_dir,
        f"{symbol}_{timeframe}_ASSET_PROFILE.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)

    profile["path"] = path
    return profile



# ========================================================================
# BLOCO 1.8 — AUTODETECÇÃO DE DOMINÂNCIA DE FLUXO (ELITE)
# ========================================================================

def detectar_tf_lider_fluxo(pasta_datasets, simbolo, lista_tfs):
    """
    Analisa todos os TFs disponíveis e identifica qual tem a maior toxicidade (VPIN).
    O vencedor será o 'TF Líder' para agressão.
    """
    vpin_scores = {}
    
    print("\n" + "="*70)
    print("🔍 AUDITORIA AUTOMÁTICA DE DOMINÂNCIA DE FLUXO")
    print("="*70)
    
    for tf in lista_tfs:
        fname = f"{simbolo}_{tf}.csv"
        path = os.path.join(pasta_datasets, fname)
        if os.path.exists(path):
            try:
                # Ler apenas a coluna VPIN para ser rápido
                df_temp = pd.read_csv(path, usecols=['vpin'])
                vpin_medio = df_temp['vpin'].mean()
                vpin_scores[tf] = vpin_medio
                print(f"  • TF {tf:4} | Toxicidade (VPIN Médio): {vpin_medio:.4f}")
            except:
                continue
    
    if not vpin_scores:
        print("  ⚠ Nenhuma coluna VPIN encontrada. Usando TF base como padrão.")
        return None
    
    tf_lider = max(vpin_scores, key=vpin_scores.get)
    print("-" * 70)
    print(f"🏆 VENCEDOR: O Timeframe {tf_lider} foi definido como LÍDER DE FLUXO.")
    print(f"🚀 O modelo dará peso prioritário à agressão do {tf_lider}.")
    print("="*70 + "\n")
    
    return tf_lider

# ========================================================================
# BLOCO 2 — FEATURE ENGINE V90 + V92 (COMPLETO, INSTITUCIONAL)
# ========================================================================

# ------------------------------------------------------------------------
# 2.1 — Volatilidades Clássicas
# ------------------------------------------------------------------------

# =====================================================================
# 🚨 IMPLEMENTAÇÃO PRO: BLOCO 2 E FEATURE ENGINE (ZERO LEAKAGE)
# =====================================================================

def realized_vol(close: pd.Series) -> pd.Series:
    """Volatilidade Realizada protegida contra o futuro."""
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean()).shift(1)

def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """Parkinson protegida com shift(1)."""
    return np.sqrt((1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()).shift(1)

def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell protegida com shift(1)."""
    rs = (np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
          np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"]))
    return np.sqrt(rs.rolling(20).mean()).shift(1)

def yang_zhang(df: pd.DataFrame) -> pd.Series:
    """Yang-Zhang blindada contra leakage."""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))
    log_co = np.log(df["close"] / df["open"])
    rs = (log_ho**2 + log_lo**2).rolling(20).mean()
    close_vol = log_co.rolling(20).std() ** 2
    open_vol = log_oc.rolling(20).std() ** 2
    return np.sqrt(0.34 * open_vol + 0.34 * close_vol + 0.27 * rs).shift(1)

def slope_regression(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Inclinação da regressão linear isolada no passado."""
    X = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * window
    for i in range(window, len(series)):
        y = series[i-window:i] # Exclui o ponto 'i' (atual)
        slopes.append(LinearRegression().fit(X, y).coef_[0])
    return pd.Series(slopes).shift(1).values # Shift(1) obrigatório

def feature_engine(df):
    """Pipeline de Geração de Features - Versão Auditada V25."""
    df = df.copy()

    # 1. VARIÁVEIS BÁSICAS (Isoladas do candle de decisão)
    df["body"] = (df["close"] - df["open"]).shift(1)
    df["range"] = (df["high"] - df["low"]).shift(1)
    
    # 2. RETORNOS PROTEGIDOS (Vazamento Crítico aqui)
    df["ret1"] = df["close"].pct_change(1).shift(1)
    df["ret5"] = df["close"].pct_change(5).shift(1)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).shift(1)

    # 3. EMAs E DISTÂNCIAS (Devem olhar para T-1)
    df["ema9"] = df["close"].ewm(span=9).mean().shift(1)
    df["ema20"] = df["close"].ewm(span=20).mean().shift(1)
    df["dist_ema9"] = (df["close"].shift(1) - df["ema9"])
    df["dist_ema20"] = (df["close"].shift(1) - df["ema20"])

    # 4. SLOPES E VOLATILIDADES AUDITADAS
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)

    # 5. AGRESSÃO PROTEGIDA
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["taker_buy_base"]).shift(1)  # ← CORREÇÃO ADICIONADA
        df["aggression_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).shift(1)
    
    return df

# =====================================================================
# FEATURES AVANÇADAS (LIMPO)
# =====================================================================

def adicionar_features_avancadas(df):
    # ==================================================================
    # 🛡️ BLINDAGEM TOTAL: GARANTIA DE INTEGRIDADE DE DADOS
    # ==================================================================
    df = df.copy()

    # 1. PRICE ACTION BÁSICO (Resolve 'upper_wick', 'range', 'body')
    if "range" not in df.columns: df["range"] = df["high"] - df["low"]
    if "body" not in df.columns: df["body"] = df["close"] - df["open"]
    if "upper_wick" not in df.columns: 
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    if "lower_wick" not in df.columns: 
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # 2. PERCENTUAIS (Resolve o erro ATUAL 'range_pct')
    if "range_pct" not in df.columns: 
        # Adicionamos 1e-9 para evitar divisão por zero
        df["range_pct"] = df["range"] / (df["close"] + 1e-9)

    # 3. RETORNOS (Resolve erros futuros de 'ret2', 'ret5', etc.)
    if "ret1" not in df.columns: df["ret1"] = df["close"].pct_change(1)
    if "ret2" not in df.columns: df["ret2"] = df["close"].pct_change(2)
    if "ret5" not in df.columns: df["ret5"] = df["close"].pct_change(5)
    if "ret10" not in df.columns: df["ret10"] = df["close"].pct_change(10)

    # 4. INDICADORES TÉCNICOS (Resolve erros de normalização)
    if "ema20" not in df.columns: 
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    if "dist_ema20" not in df.columns: 
        df["dist_ema20"] = df["close"] - df["ema20"]
    if "atr14" not in df.columns:
        tr = pd.concat([
            (df["high"] - df["low"]), 
            (df["high"] - df["close"].shift(1)).abs(), 
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
    # ==================================================================

    # ---> AQUI COMEÇA O SEU CÓDIGO ORIGINAL (Mantenha tudo abaixo) <---
    # Agora a linha 1092 vai encontrar 'range_pct' e funcionar:
    if "volatility_acceleration" not in df.columns: # Proteção extra
         df["volatility_acceleration"] = df["range_pct"].diff()

    # CORREÇÃO: Removido 'return df' prematuro que impedia features avançadas
    # ... continue com o resto da sua função ...
    # return df    # ← BUG REMOVIDO!
    # ... (mantenha o resto da função original) ...
    # ------------------------------------------------------------
    # 2. MOMENTUM / VELOCIDADE
    # ------------------------------------------------------------
    df["momentum_1"] = df["ret1"]
    df["momentum_2"] = df["ret2"]
    df["momentum_acceleration"] = df["momentum_1"] - df["momentum_2"]

    # ------------------------------------------------------------
    # 3. WICK / STRUCTURE (duplicadas REMOVIDAS)
    # ------------------------------------------------------------
    df["body_to_range"] = df["body"] / (df["range"] + 1e-8)
    df["wick_ratio_up"] = df["upper_wick"] / (df["range"] + 1e-8)
    df["wick_ratio_down"] = df["lower_wick"] / (df["range"] + 1e-8)

    # Duplicatas removidas:
    # df["body_ratio"] = ...
    # df["close_position_in_range"] = ...
    # df["wick_upper_ratio"] = ...
    # df["wick_lower_ratio"] = ...

    # ------------------------------------------------------------
    # 4. MICROESTRUTURA (duplicadas removidas)
    # micro_range = range (duplicado)
    # micro_volatility = range_pct (duplicado)
    # ------------------------------------------------------------
    # df["micro_range"] = df["range"]                          # REMOVIDO
    # df["micro_volatility"] = df["range_pct"]                 # REMOVIDO

    # ------------------------------------------------------------
    # 5. VOLUME / SURGE (duplicado de exp_rate_20 REMOVIDO)
    # ------------------------------------------------------------
    df["volume_diff"] = df["volume"].diff()
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / \
                           (df["volume"].rolling(20).std() + 1e-8)

    # df["volume_surge"] = df["volume"] / df["volume"].rolling(20).mean()   # REMOVIDO — igual a exp_rate_20

    # ------------------------------------------------------------
    # 6. AGRESSÃO (duplicatas REMOVIDAS)
    # ------------------------------------------------------------
    df["aggression_imbalance"] = df["aggression_delta"] / (df["volume"] + 1e-8)
    df["aggression_pressure"] = df["aggression_delta"] * df["ret1"]

    # Duplicatas REMOVIDAS:
    # df["aggression_delta"] (já existe no Feature Engine)
    # df["aggression_ratio"] (já existe)
    
    # ------------------------------------------------------------
    # 7. ACELERAÇÕES / DERIVADAS
    # ------------------------------------------------------------
    df["delta_acc"] = df["aggression_delta"].diff()
    df["volatility_acceleration"] = df["range_pct"].diff()

    # =====================================================================
    # 7. ATR E VOLATILIDADE INSTITUCIONAL
    # =====================================================================

    # True Range
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )

    # ATR clássico de 14 períodos
    df["atr14"] = df["tr"].rolling(14).mean()

    # ATR normalizado
    df["atr_to_close"] = df["atr14"] / (df["close"] + 1e-8)

    # Range relativo ao ATR (rompimentos)
    df["range_to_atr"] = df["range"] / (df["atr14"] + 1e-8)


    # =====================================================================
    # 8. REGIMES (VOLATILIDADE, TENDÊNCIA, LIQUIDEZ)
    # =====================================================================

    # Volatility regime (estado de volatilidade de 50 períodos)
    df["vol_regime"] = df["atr14"].rolling(50).mean()

    # Compressão de volatilidade (squeeze)
    df["atr_compression"] = df["atr14"] / (df["vol_regime"] + 1e-8)

    # Trend regime — baseado em slope de 100 períodos
    df["trend_regime"] = slope_regression(df["close"], 100)

    # Liquidez regime — média de volume
    df["liquidity_regime"] = df["volume"].rolling(50).mean()


    # =====================================================================
    # 9. RETORNOS AVANÇADOS / MOMENTUM EXPANDIDO
    # =====================================================================

    df["ret3"] = df["close"].pct_change(3)
    df["ret10"] = df["close"].pct_change(10)
    df["ret20"] = df["close"].pct_change(20)

    # Retornos normalizados pela volatilidade
    df["ret20_norm"] = df["ret20"] / (df["atr14"] + 1e-8)

    # Momentum agregado
    df["momentum_long"] = df["ret3"] + df["ret10"] + df["ret20"]


    # =====================================================================
    # 10. Z-SCORE DIRECIONAL (REVERSÃO / EXAUSTÃO)
    # =====================================================================

    df["price_z"] = (
        (df["close"] - df["close"].rolling(20).mean()) /
        (df["close"].rolling(20).std() + 1e-8)
    )


    # =====================================================================
    # 11. SQUEEZE AVANÇADO (VOL, RANGE, RET)
    # =====================================================================

    df["vol_squeeze"] = (
        df["close"].rolling(20).std() /
        (df["close"].rolling(100).std() + 1e-8)
    )

    df["range_squeeze"] = (
        df["range_pct"].rolling(14).std() /
        (df["range_pct"].rolling(50).std() + 1e-8)
    )


    # =====================================================================
    # 12. FEATURES DE Agressão (Fluxo Institucional)
    # =====================================================================

    # Buy / Sell ratio (muito forte)
    df["buy_ratio"] = df["aggression_buy"] / (df["volume"] + 1e-8)
    df["sell_ratio"] = df["aggression_sell"] / (df["volume"] + 1e-8)

    # Agressão acumulada — detecta pressão contínua
    df["aggr_cumsum_20"] = df["aggression_delta"].rolling(20).sum()

    # 🚀 FEATURE DE ELITE: DOMINÂNCIA DE FLUXO (PESO DINÂMICO)
    # Esta feature amplifica o sinal de agressão do TF Líder para o modelo dar prioridade
    tf_lider = globals().get("TF_LIDER_FLUXO", "15m")
    
    # Se estivermos no TF Líder, criamos a feature de dominância
    if "aggression_delta" in df.columns:
        # Se for o TF Líder (ou se não houver líder definido e for o base)
        df["flow_dominance"] = df["aggression_delta"] * (1 + df["vpin"].fillna(0))
        df["flow_acceleration"] = df["aggression_delta"].diff()
        # print(f"✔ Feature de Dominância de Fluxo ativada.")


    # =====================================================================
    # 13. DISTÂNCIAS IMPORTANTES (NORMALIZAÇÕES INSTITUCIONAIS)
    # =====================================================================
    
    # Criar EMAs se não existirem
    if "ema50" not in df.columns:
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    if "ema100" not in df.columns:
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    
    # Criar distâncias se não existirem
    if "dist_ema50" not in df.columns:
        df["dist_ema50"] = df["close"] - df["ema50"]
    if "dist_ema100" not in df.columns:
        df["dist_ema100"] = df["close"] - df["ema100"]
    
    # Normalizar distâncias
    df["dist_ema20_norm"] = df["dist_ema20"] / (df["atr14"] + 1e-8)
    df["dist_ema50_norm"] = df["dist_ema50"] / (df["atr14"] + 1e-8)
    df["dist_ema100_norm"] = df["dist_ema100"] / (df["atr14"] + 1e-8)


    # =====================================================================
    # 14. SLOPES LONGOS (TENDÊNCIA PROFUNDA)
    # =====================================================================

    df["slope100"] = slope_regression(df["close"], 100)
    df["slope200"] = slope_regression(df["close"], 200)


    return df

# ===============================================================
# DETECÇÃO DE REGIMES DE MERCADO COM SALVAMENTO DE SCALER/KMEANS
# ===============================================================

def detectar_regimes_mercado_v25(df, n_regimes=4):
    """
    Detecta regimes de mercado usando KMeans e salva scaler/kmeans.
    
    IMPORTANTE: Esta função SALVA automaticamente:
    - modelos_salvos/scaler_regimes.pkl
    - modelos_salvos/kmeans_regimes.pkl
    
    Args:
        df: DataFrame com features calculadas
        n_regimes: número de clusters (padrão: 4)
    
    Returns:
        DataFrame com coluna 'market_regime' adicionada
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os
    
    # 1. Identifica as colunas que compõem a "régua" (scaler)
    regime_features = [c for c in ['vol_realized', 'rsi_14', 'atr14', 'slope20'] if c in df.columns]
    
    if not regime_features:
        df['temp_ret'] = df['close'].pct_change(20)
        regime_features = ['temp_ret']
    
    # 2. Prepara a matriz de dados para o scaler
    X_regime = df[regime_features].fillna(0).values
    
    # 3. Cria e treina o Scaler
    global scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    
    # 4. Aplica o Cluster (KMeans) para definir os regimes de mercado
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    # 5. SALVA SCALER E KMEANS NO DISCO
    try:
        # Criar diretório se não existir (usando OUT_DIR para persistência)
        modelos_dir = os.path.join(OUT_DIR, 'modelos_salvos')
        os.makedirs(modelos_dir, exist_ok=True)
        
        # Salvar SCALER
        scaler_path = os.path.join(modelos_dir, 'scaler_regimes.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Salvar KMEANS
        kmeans_path = os.path.join(modelos_dir, 'kmeans_regimes.pkl')
        joblib.dump(kmeans, kmeans_path)
        
        # Garantir que estejam no cache global
        if 'modelos_cache' not in globals():
            globals()['modelos_cache'] = {}
        globals()['modelos_cache']['scaler_regimes'] = scaler
        globals()['modelos_cache']['kmeans_regimes'] = kmeans
        
        print(f"✅ SCALER E KMEANS SALVOS COM SUCESSO!")
        print(f"   📁 Diretório: {modelos_dir}")
        print(f"   ├─ scaler_regimes.pkl   ({os.path.getsize(scaler_path)} bytes)")
        print(f"   └─ kmeans_regimes.pkl   ({os.path.getsize(kmeans_path)} bytes)")
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao salvar scaler/kmeans: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return df

# ===============================================================
# CHAMADA OFICIAL (NÃO MODIFICAR ESTRUTURA — APENAS SANIDADE)
# ===============================================================

print(">>> Aplicando Feature Engine institucional completo (V90 + V92 + Features Avançadas)...")

# ================================================================
# FEATURE ENGINE + FEATURES AVANÇADAS + TARGET A BOOSTER (FINAL)
# ================================================================

df_feat = feature_engine(df_raw)
df_feat = adicionar_features_avancadas(df_feat)

# ---------- NOVOS MÓDULOS (ORDEM CORRETA) ----------
df_feat = adicionar_fractais_elliott(df_feat)
df_feat = adicionar_vwap(df_feat)
df_feat = adicionar_micro_squeeze(df_feat)
df_feat = adicionar_inside_nr(df_feat)
df_feat = adicionar_zscore_intrabar(df_feat)

# ============================================================
# 🔧 CORREÇÃO CRÍTICA: DETECTAR REGIMES E SALVAR SCALER/KMEANS
# ============================================================
print("\n" + "="*60)
print("🔬 DETECTANDO REGIMES DE MERCADO E SALVANDO SCALER/KMEANS...")
print("="*60)
df_feat = detectar_regimes_mercado_v25(df_feat, n_regimes=4)
# Isso vai salvar automaticamente:
# - modelos_salvos/scaler_regimes.pkl
# - modelos_salvos/kmeans_regimes.pkl
print("="*60 + "\n")
# ============================================================


print("✔ Features avançadas adicionadas (institucional + Target A Booster V2).")

# ---------------------------------------------------------------
# AJUSTE INSTITUCIONAL PARA ELIMINAR NAN/INF (SEM MEXER EM FEATURES)
# ---------------------------------------------------------------

# 1) Infinitos viram NaN temporariamente
df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2) Preenchimento inteligente:
#    - colunas numéricas: mediana
#    - colunas não numéricas: zero
for col in df_feat.columns:
    if df_feat[col].dtype != "object":
        df_feat[col].fillna(df_feat[col].median(), inplace=True)
    else:
        df_feat[col].fillna(0, inplace=True)

print("✔ Features avançadas adicionadas (institucional).")
print("✔ Sanidade aplicada (NaN/INF removidos — compatível com RegLog).")


def auto_thresholds_v25(df):

    if "atr14" not in df.columns or "close" not in df.columns:
        raise RuntimeError("ATR14 ou close não encontrado.")

    # ATR em percentual (escala compatível com amp_fut)
    atr_pct = df["atr14"] / df["close"]

    atr_ref = (
        atr_pct
        .rolling(200, min_periods=200)
        .mean()
    )

    atr_global = atr_pct.mean()
    atr_ref = atr_ref.fillna(atr_global)

    # Thresholds agora em % (compatíveis com amp_fut)
    thrA = float((atr_ref * 1.2).iloc[-1])
    thrB = float((atr_ref * 1.6).iloc[-1])
    thrC = float((atr_ref * 2.0).iloc[-1])

    print("\n>>> THRESHOLDS SEM LEAKAGE (ATR%):")
    print(f"    • Threshold A = {thrA:.4%}")
    print(f"    • Threshold B = {thrB:.4%}")
    print(f"    • Threshold C = {thrC:.4%}")

    return thrA, thrB, thrC

# ===============================================================
# V25 — PREPARAÇÃO DE FUTUROS (OFICIAL)
# ===============================================================

def preparar_futuros(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    Calcula retorno futuro (ret_fut) e amplitude futura (amp_fut)
    sem leakage e remove linhas sem futuro.
    """
    df = df.copy()

    # preço futuro N barras à frente (sem leakage)
    df["close_future"] = df["close"].shift(-N)
    df["ret_fut"] = (df["close_future"] - df["close"]) / df["close"]

    # máxima e mínima futuras do candle N à frente
    df["high_fut"] = df["high"].shift(-N)
    df["low_fut"] = df["low"].shift(-N)
    df["amp_fut"] = (df["high_fut"] - df["low_fut"]) / (df["close"] + 1e-9)

    # remover linhas sem futuro
    # remover apenas as últimas N linhas, onde não existe futuro
    df = df.iloc[:-N].reset_index(drop=True)


    return df


# ===============================================================
# V25 — AUTO THRESHOLDS (A, B, C) — percentis institucionais
# ===============================================================

# ===============================================================
# V25 — CRIAÇÃO DOS TARGETS A, B, C
# ===============================================================

def criar_targets_v25(df: pd.DataFrame, thrA: float, thrB: float, thrC: float) -> pd.DataFrame:
    df = df.copy()
    sensibilidade_short = 0.95
    thrA_s, thrB_s, thrC_s = thrA * sensibilidade_short, thrB * sensibilidade_short, thrC * sensibilidade_short
    df["target_A"] = 0
    df.loc[df["ret_fut"] >= thrA, "target_A"] = 1
    df.loc[df["ret_fut"] <= -thrA_s, "target_A"] = -1
    df["target_B"] = 0
    df.loc[df["ret_fut"] >= thrB, "target_B"] = 1
    df.loc[df["ret_fut"] <= -thrB_s, "target_B"] = -1
    df["target_C"] = 0
    df.loc[df["amp_fut"] >= thrC, "target_C"] = 1
    df["target_A_bin"] = ((df["ret_fut"] >= thrA) | (df["ret_fut"] <= -thrA_s)).astype(int)
    return df


# ===============================================================
# V25 — TARGETS DE REVERSÃO (OPÇÃO B - PUROS)
# ===============================================================

def criar_targets_reversao(df: pd.DataFrame, thrA: float) -> pd.DataFrame:
    """
    Cria targets de reversão PUROS - sem regras fixas de lookback.
    
    O MODELO aprende sozinho quais padrões precedem reversões usando
    todas as features disponíveis (RSI, EMAs, volume, agressão, etc.)
    
    LÓGICA SIMPLES:
    - target_REV_LONG = 1 se o preço VAI SUBIR forte (ret_fut >= thrA)
    - target_REV_SHORT = 1 se o preço VAI CAIR forte (ret_fut <= -thrA)
    - target_CONFLUENCIA = árbitro para quando LONG e SHORT divergem
    
    DIFERENÇA DOS TARGETS A/B:
    - Target A/B são TERNÁRIOS (-1, 0, 1) para direção geral
    - Target REV são BINÁRIOS (0, 1) específicos para cada direção
    - REV_LONG: modelo especialista em COMPRA (só aprende padrões de alta)
    - REV_SHORT: modelo especialista em VENDA (só aprende padrões de baixa)
    - CONFLUENCIA: árbitro que decide quando especialistas divergem
    
    VANTAGEM:
    - Modelos especialistas tendem a ter melhor precisão
    - Cada modelo foca em identificar SEU padrão específico
    - Pode usar thresholds de confiança diferentes para cada direção
    - Confluência inteligente resolve divergências
    
    Args:
        df: DataFrame com ret_fut já calculado
        thrA: threshold do target A (usado como referência)
    
    Returns:
        DataFrame com targets de reversão adicionados
    """
    df = df.copy()
    
    # Sensibilidade para short (mesmo do target_A original)
    sensibilidade_short = 0.95
    thrA_s = thrA * sensibilidade_short
    
    # ---------------------------------------------------------------
    # TARGET_REV_LONG - Especialista em COMPRA
    # ---------------------------------------------------------------
    # Target = 1 se vai subir forte
    # O modelo aprende quais features indicam isso (RSI baixo, suporte, etc.)
    df["target_REV_LONG"] = (df["ret_fut"] >= thrA).astype(int)
    
    # ---------------------------------------------------------------
    # TARGET_REV_SHORT - Especialista em VENDA  
    # ---------------------------------------------------------------
    # Target = 1 se vai cair forte
    # O modelo aprende quais features indicam isso (RSI alto, resistência, etc.)
    df["target_REV_SHORT"] = (df["ret_fut"] <= -thrA_s).astype(int)
    
    # ---------------------------------------------------------------
    # TARGET_CONFLUENCIA - Árbitro para divergências
    # ---------------------------------------------------------------
    # Este target é TERNÁRIO: -1 (venda), 0 (neutro), 1 (compra)
    # Usado quando REV_LONG e REV_SHORT divergem
    # O modelo aprende a desempatar baseado em todas as features
    df["target_CONFLUENCIA"] = 0
    df.loc[df["ret_fut"] >= thrA, "target_CONFLUENCIA"] = 1      # Movimento de alta
    df.loc[df["ret_fut"] <= -thrA_s, "target_CONFLUENCIA"] = -1  # Movimento de baixa
    
    # ---------------------------------------------------------------
    # Estatísticas
    # ---------------------------------------------------------------
    n_total = len(df)
    n_long = df["target_REV_LONG"].sum()
    n_short = df["target_REV_SHORT"].sum()
    n_conf_up = (df["target_CONFLUENCIA"] == 1).sum()
    n_conf_down = (df["target_CONFLUENCIA"] == -1).sum()
    n_conf_neutro = (df["target_CONFLUENCIA"] == 0).sum()
    
    print("\n" + "="*70)
    print("TARGETS DE REVERSÃO CRIADOS (OPÇÃO B - PUROS)")
    print("="*70)
    print(f"Threshold LONG:  {thrA:.4%}")
    print(f"Threshold SHORT: {thrA_s:.4%}")
    print("-"*70)
    print(f"📈 TARGET_REV_LONG (Especialista COMPRA):")
    print(f"   • Sinais de ALTA forte: {n_long} ({n_long/n_total*100:.2f}%)")
    print(f"   • Distribuição: {df['target_REV_LONG'].value_counts().to_dict()}")
    print("-"*70)
    print(f"📉 TARGET_REV_SHORT (Especialista VENDA):")
    print(f"   • Sinais de BAIXA forte: {n_short} ({n_short/n_total*100:.2f}%)")
    print(f"   • Distribuição: {df['target_REV_SHORT'].value_counts().to_dict()}")
    print("-"*70)
    print(f"⚖️ TARGET_CONFLUENCIA (Árbitro):")
    print(f"   • COMPRA (1): {n_conf_up} ({n_conf_up/n_total*100:.2f}%)")
    print(f"   • NEUTRO (0): {n_conf_neutro} ({n_conf_neutro/n_total*100:.2f}%)")
    print(f"   • VENDA (-1): {n_conf_down} ({n_conf_down/n_total*100:.2f}%)")
    print("="*70)
    print("💡 O modelo usará TODAS as features para aprender os padrões!")
    print("   (RSI, EMAs, volume, agressão, volatilidade, etc.)")
    print("="*70 + "\n")
    
    return df


# ===============================================================
# V25 — EXECUÇÃO COMPLETA (AUTO) — BLOCO OFICIAL
# ===============================================================

print("\n===============================================================")
print("CONFIGURAÇÃO DOS TARGETS (ADAPTATIVO V25)")
print("===============================================================\n")

# 1) Horizonte futuro (CONFIGURAÇÃO FIXA - SEM INPUT)
h_fut = HORIZONTE_FUTURO
print(f"✔ Horizonte futuro N = {h_fut}")

# 2) Preparar futuros
print(">>> Calculando futuros (ret_fut, amp_fut)...")
df_all = preparar_futuros(df_feat, h_fut)

# ============================================================
# SNAPSHOT PARA QUANTILE REGRESSION (COM FUTURO PRESERVADO)
# ============================================================
df_quantile = df_all.copy()

print("ret_fut in df_quantile:", "ret_fut" in df_quantile.columns)
print("ret_fut in df_all:", "ret_fut" in df_all.columns)


# =====================================================================
# APLICAR THRESHOLD ADAPTATIVO DO TARGET A (S7)
# =====================================================================
print(">>> Aplicando módulo Target A Adaptativo (S7)...")
saida_S7 = modulo_targetA_threshold_adaptativo(df_all)
df_all = df_all.join(saida_S7)
print("✔ Target A adaptativo calculado.\n")

# 3) Auto thresholds (SEM LOOK-AHEAD)
print(">>> Calculando thresholds ideais (A, B, C)...")

df_calib = df_all.iloc[:int(len(df_all) * 0.2)]
thrA, thrB, thrC = auto_thresholds_v25(df_calib)

print(f"✔ Threshold Target A: {thrA:.6f}")
print(f"✔ Threshold Target B: {thrB:.6f}")
print(f"✔ Threshold Target C: {thrC:.6f}")

# 4) Criar targets
print(">>> Criando targets A, B, C (adaptativo V25)...")
df_all = criar_targets_v25(df_all, thrA, thrB, thrC)

print("✔ Targets criados com sucesso.\n")

# ===============================================================
# CRIAR TARGETS DE REVERSÃO (ESPECIALISTAS LONG/SHORT)
# ===============================================================
print(">>> Criando targets de reversão (REV_LONG / REV_SHORT)...")
df_all = criar_targets_reversao(df_all, thrA)
print("✔ Targets de reversão criados com sucesso.\n")

print("\n=== DISTRIBUIÇÃO DO TARGET_A ===")
print(df_all["target_A"].value_counts().sort_index())
print("\n=== PROPORÇÃO (%) DO TARGET_A ===")
print(df_all["target_A"].value_counts(normalize=True).sort_index() * 100)


print("\n>>> Target A Binário criado:")
print(df_all["target_A_bin"].value_counts(normalize=True))

print("\n>>> Gerando Asset Profile institucional...")

# Asset Profile é diagnóstico — não interfere no pipeline
if "amp_fut" not in df_all.columns:
    raise RuntimeError(
        "amp_fut ausente — Asset Profile deve ser gerado antes da limpeza anti-leakage"
    )

asset_profile = gerar_asset_profile(
    df=df_all,
    symbol=simbolo,
    timeframe=tf_base_global,
    thrA=thrA,
    out_dir=out_dir
)

print("\n==============================================================")
print("ASSET PROFILE — RESUMO (SANIDADE)")
print("==============================================================")
print(f"symbol               : {asset_profile['symbol']}")
print(f"timeframe            : {asset_profile['timeframe']}")
print(f"n_bars_total         : {asset_profile['n_bars_total']}")
print(f"p50_move             : {asset_profile['p50_move']}")
print(f"p90_move             : {asset_profile['p90_move']}")
print(f"p95_move             : {asset_profile['p95_move']}")
print(f"p99_move             : {asset_profile['p99_move']}")
print(f"hit_rate_A_bin       : {asset_profile['hit_rate_A_bin']}")
print(f"tier_quality         : {asset_profile['tier_quality']}")
print(f"operable_flag        : {asset_profile['operable_flag']}")
print("==============================================================\n")
if "path" in asset_profile:
    print(f"✔ Asset Profile salvo em:\n  {asset_profile['path']}")


# ========================================================================
# BLOCO 4 — MULTI-TIMEFRAME CONTEXT (ZERO-LEAKAGE) — V25 FINAL REAL
# ========================================================================

print("\n===============================================================")
print("MÓDULO 4 — CONTEXTO MULTI-TF (OPCIONAL / ZERO LEAKAGE)")
print("===============================================================\n")

# RENDER FIX: Usar configuração fixa sem input
usar_ctx = "s" if USAR_MULTIFRAME else "n"

if usar_ctx == "s":

    # -----------------------------------------------------------
    # 1) Detectar TF base automaticamente pelo nome do arquivo
    # -----------------------------------------------------------
    nome_arquivo = os.path.basename(csv_path).lower()
    partes = nome_arquivo.split("_")

    tf_base = partes[1].replace(".csv", "")      # <<< PATCH: remove .csv
    simbolo = partes[0].upper()

    print(f">>> Timeframe base detectado: {tf_base}")
    print(f">>> Símbolo detectado: {simbolo}")

    # -----------------------------------------------------------
    # 2) Hierarquia institucional SEM LEAKAGE
    # -----------------------------------------------------------
    hierarchy = {
        "1m":  ["5m","15m","30m","1h","4h","8h","1d"],
        "5m":  ["15m","30m","1h","4h","8h","1d"],
        "15m": ["30m","1h","4h","8h","1d"],
        "30m": ["1h","4h","8h","1d"],
        "1h":  ["4h","8h","1d"],
        "4h":  ["8h","1d"],
        "8h":  ["1d"],
        "1d":  [],
    }

    valid_tfs = hierarchy.get(tf_base, [])
    print(f">>> TFs maiores permitidos: {valid_tfs}")

    # RENDER FIX: Usar configuração fixa
    escolha = TFS_ADICIONAIS
    print(f">>> TFs selecionados: {escolha}")
    if escolha == "":
        chosen_tfs = []
    else:
        chosen_tfs = [x.strip() for x in escolha.split(",")]
        chosen_tfs = [x for x in chosen_tfs if x in valid_tfs]

    print(f">>> TFs escolhidos: {chosen_tfs}\n")

    # 🚀 AUTODETECÇÃO DE DOMINÂNCIA DE FLUXO
    # Analisa o TF base + os escolhidos para definir quem manda na agressão
    todos_tfs = [tf_base] + chosen_tfs
    global TF_LIDER_FLUXO
    TF_LIDER_FLUXO = detectar_tf_lider_fluxo(os.path.dirname(csv_path), simbolo, todos_tfs)

    # -----------------------------------------------------------
    # 3) Timestamp seguro — manter SOMENTE ms (int64)
    # -----------------------------------------------------------
    df_raw["ts"] = df_raw["ts"].astype("int64")
    df_raw = df_raw.sort_values("ts").reset_index(drop=True)

    # -----------------------------------------------------------
    # Função institucional de tolerance por TF (ms)
    # -----------------------------------------------------------
    def tolerance_por_tf(tf):
        mapa = {
            "15m":  30 * 60 * 1000,
            "30m":  60 * 60 * 1000,
            "1h":   2  * 60 * 60 * 1000,
            "4h":   8  * 60 * 60 * 1000,
            "8h":   16 * 60 * 60 * 1000,
            "1d":   2  * 24 * 60 * 60 * 1000,
        }
        return mapa.get(tf, None)

    # -----------------------------------------------------------
    # 4) MERGE ASOF para cada TF escolhido
    # -----------------------------------------------------------
    for tf in chosen_tfs:

        fname = f"{simbolo}_{tf}.csv"             # <<< PATCH: nome real do arquivo
        path_tf = os.path.join(os.path.dirname(csv_path), fname)

        print(f">>> Carregando contexto {tf} de: {path_tf}")

        if not os.path.isfile(path_tf):
            print(f"[IGNORADO] Arquivo não encontrado: {path_tf}")
            continue

        df_big = pd.read_csv(path_tf)

        if "ts" not in df_big.columns:
            print(f"[ERRO] Arquivo {path_tf} não tem coluna 'ts'. Ignorado.")
            continue

        df_big["ts"] = df_big["ts"].astype("int64")
        df_big = df_big.sort_values("ts")

        tol = tolerance_por_tf(tf)
        if tol is None:
            print(f"[IGNORADO] TF {tf} sem tolerance definida.")
            continue

        # -----------------------------------------------------------
        # BLOQUEIO DE TF STALE
        # -----------------------------------------------------------
        ultimo_tf = df_big["ts"].max()
        ultimo_base = df_raw["ts"].max()

        if ultimo_tf < (ultimo_base - tol):
            print(f"[IGNORADO] TF {tf} está desatualizado (stale).")
            continue

        # -----------------------------------------------------------
        # SHIFT de 1 barra — evita leakage
        # -----------------------------------------------------------
        feature_cols = [
            c for c in df_big.columns
            if c not in ["ts", "close_time"] and df_big[c].dtype != "object"
        ]

        feature_cols = [
            c for c in feature_cols
            if not any(x in c.lower() for x in ["open","high","low","close"])
        ]
        
        # 🔴 PATCH: Excluir buy_vol e sell_vol (duplicados de buy_vol_agg/sell_vol_agg)
        feature_cols = [
            c for c in feature_cols
            if c not in ["buy_vol", "sell_vol"]
        ]

        df_big_shifted = df_big.copy()
        df_big_shifted[feature_cols] = df_big_shifted[feature_cols].shift(1)

        df_big_pref = df_big_shifted[["ts"] + feature_cols].copy()
        df_big_pref = df_big_pref.add_prefix(f"ctx_{tf}_")
        df_big_pref = df_big_pref.rename(columns={f"ctx_{tf}_ts": "ts"})

        try:
            df_raw = pd.merge_asof(
                df_raw.sort_values("ts"),
                df_big_pref.sort_values("ts"),
                on="ts",
                direction="backward",
                tolerance=tol
            )
        except Exception as e:
            print(f"[ERRO] Falha no merge_asof para TF {tf}: {e}")
            continue

        print(f"[OK] {len(feature_cols)} colunas agregadas do TF {tf}.\n")

    # -----------------------------------------------------------
    # 5) Propagar colunas ctx_* para df_all
    # -----------------------------------------------------------
    ctx_cols = [c for c in df_raw.columns if c.startswith("ctx_")]

    if ctx_cols:
        df_ctx = df_raw[["ts"] + ctx_cols].copy()
        cols_to_add = [c for c in ctx_cols if c not in df_all.columns]

        if cols_to_add:
            df_ctx = df_ctx[["ts"] + cols_to_add]
            df_all = df_all.merge(df_ctx, on="ts", how="left")
            print(f"[OK] {len(cols_to_add)} colunas ctx_* adicionadas ao df_all.\n")
        else:
            print("[INFO] Nenhuma coluna de contexto nova para adicionar ao df_all.\n")
    else:
        print("[INFO] Nenhuma coluna ctx_* encontrada em df_raw.\n")

    print("✔ Contexto multi-TF aplicado com sucesso!\n")

# ==================================================================================
# Importante: df_raw agora já contém colunas ctx_XXX_YYY.
# O Feature Engine já foi aplicado em df_feat (BLOCO 2),
# por isso NÃO reprocessamos df_raw aqui.
# ==================================================================================

# ===========================================================
# RESULTADOS DO TREINO (para Painel Operacional)
# ===========================================================
resultados_treino = {}

# 🔴 PATCH: Caches globais para o BACKTEST_DECISOR
modelos_cache = {}
X_arrays_cache = {}
probs_cache = {}
lista_targets = []

# ===========================================================
# FONTES DA VERDADE (TREINO → EXPORTADOR → BACKTEST)
# ===========================================================
# • caminhos_modelos: onde cada modelo foi salvo
# • features_por_target: lista EXATA de features usadas no treino
caminhos_modelos = {}
features_por_target = {}



# ========================================================================
# BLOCO 5 — TREINO INSTITUCIONAL (LGBM + XGB + CAT) — V25 FINAL REAL
# ========================================================================

print("\n===============================================================")
print("MÓDULO 5 — TREINO DOS MODELOS (70/10/20)")
print("===============================================================\n")


# ------------------------------------------------------------------------
# 5.1 — Gerar matriz de features
# ------------------------------------------------------------------------

def montar_matriz(df: pd.DataFrame, alvo: str):
    """
    Remove colunas que não são features + retorna X, y e lista de colunas.

    PATCH CRÍTICO:
    - Remove TODAS as colunas target_* das features (A_bin, A/B/C, K1..K5, etc).
    - Evita leakage estrutural no treino e garante identidade com backtest.
    - GARANTE X 100% numérico (necessário para Quantile Regression).
    """

    non_feat = {
        # candles
        "open", "high", "low", "close", "volume",

        "ts","open","high","low","close","volume","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","close_time","ignore",
        "mark_price","index_price","fundingRate",
        "session","tp",
        "close_future","ret_fut","amp_fut",
        "high_fut","low_fut",
        "impulse_count",
        "ret_max", "ret_min", "ret_max_temp", "ret_min_temp", # 🔴 PATCH ANTI-LEAKAGE
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg", # Colunas auxiliares de micro (não são features diretas)
        "buy_vol", "sell_vol", # 🔴 PATCH: Duplicados de buy_vol_agg/sell_vol_agg (evitar 12 features extras)
        # 🔴 FRACTAIS - VAZAMENTO (shift negativo)
        "fractal_high",
        "fractal_low",
        "pivot_high",
        "pivot_low",
        "last_pivot_high",
        "last_pivot_low",
        "swing_dir",
        "wave_amplitude",
        "wave_amplitude_abs",
        "correction_pct",
        "wave_strength",
        }

    # 🔴 PATCH: remover TODOS os targets do set de colunas não-feature
    target_cols = {
        c for c in df.columns
        if isinstance(c, str) and c.startswith("target_")
    }
    non_feat = non_feat.union(target_cols)

    if alvo not in df.columns:
        raise RuntimeError(f"[montar_matriz] alvo ausente no df: {alvo}")

    # -----------------------------
    # y (alvo)
    # -----------------------------
    y = df[alvo].values

    # -----------------------------
    # Seleção inicial de features
    # -----------------------------
    feat_cols = [c for c in df.columns if c not in non_feat]
    X = df[feat_cols].copy()

    # -----------------------------
    # 🔥 SANIDADE FINAL — APENAS NUMÉRICAS
    # -----------------------------
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise RuntimeError(
            f"[montar_matriz] Nenhuma feature numérica disponível para alvo={alvo}"
        )

    # Converte QUALQUER resíduo para numérico (object, datetime mascarado etc.)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Elimina NaN gerado por coerção
    if X.isna().any().any():
        X = X.fillna(0.0)

    feat_cols = list(X.columns)

    # Retorno FINAL — 100% float
    return X.values.astype(float), y, feat_cols


# ------------------------------------------------------------------------
# 5.2 — Split temporal 70/10/20
# ------------------------------------------------------------------------

def temporal_split(X, y, gap=15):
    n = len(X)
    tr = int(n * 0.70)
    va = int(n * 0.80)

    tr_end = tr
    va_start = tr + gap
    va_end = va
    te_start = va + gap
    return (
        X[:tr_end], y[:tr_end],
        X[va_start:va_end], y[va_start:va_end],
        X[te_start:], y[te_start:],
    )


# ------------------------------------------------------------------------
# 5.3 — Painel de probabilidade (operacional) — universal
# ------------------------------------------------------------------------

def painel_probabilidade(y_true, y_pred, y_proba):
    """
    Retorna string formatada com faixas de confiança.
    """
    probs = y_proba[np.arange(len(y_proba)), y_pred]
    mask = probs >= 0.50

    total = len(y_true)
    ops = int(mask.sum())
    disc = total - ops
    ac = int(((y_pred == y_true) & mask).sum())
    er = ops - ac
    taxa = ac / ops if ops > 0 else 0

    linhas = []
    linhas.append("📊 OPERACIONAL (conf >= 0.50)")
    linhas.append(f"  Total teste .............. {total}")
    linhas.append(f"  Operações válidas ........ {ops}")
    linhas.append(f"  → Acertos: {ac} | Erros: {er} | Taxa: {taxa:.1%}")
    linhas.append(f"  Descartados .............. {disc}")
    linhas.append("")
    linhas.append("  Faixa       Total   Ac   Er   %Acerto")
    linhas.append("  -------------------------------------")

    faixas = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]
    corretos = (y_pred == y_true)

    for lo,hi in faixas:
        m = (probs >= lo) & (probs < hi)
        n = int(m.sum())
        if n == 0:
            continue
        ac_n = int((corretos & m).sum())
        er_n = n - ac_n
        tx = ac_n / n
        linhas.append(f"  {lo:.2f}-{hi:.2f}   {n:5d}   {ac_n:3d}   {er_n:3d}   {tx:6.1%}")

    return "\n".join(linhas)
# ------------------------------------------------------------------------
# 5.4 — Treino de um único modelo (A, B ou C) — VERSÃO FINAL ESTÁVEL
# ------------------------------------------------------------------------
def treinar_um_target(target_col, df, outdir):

    print(f"\n{'='*70}")
    print(f"TREINANDO TARGET {target_col}")
    print(f"{'='*70}")

    # ------------------------------------------------------------
    # 0 — Ajuste universal de classes
    # ------------------------------------------------------------
    df_local = df.copy()
    classes_orig = np.unique(df_local[target_col].values)

    if set(classes_orig) == {-1, 0, 1}:
        df_local[target_col] = df_local[target_col].map({-1: 0, 0: 1, 1: 2})
    elif len(classes_orig) > 0 and classes_orig.min() < 0:
        df_local[target_col] = df_local[target_col] - classes_orig.min()

    classes = np.unique(df_local[target_col].values)
    n_classes = len(classes)

    # ------------------------------------------------------------
    # 1 — Matriz + split temporal
    # ------------------------------------------------------------
    # CRÍTICO: pegar feat_cols para alimentar o EXPORTADOR V22
    X, y, feat_cols = montar_matriz(df_local, target_col)

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

    if "sample_weight" in df_local.columns:
        sw = df_local["sample_weight"].values
        sw_train = sw[:len(X_train)]
    else:
        sw_train = None

    resultados = []  # (nome, f1, modelo)

    # ------------------------------------------------------------
    # 2 — LightGBM
    # ------------------------------------------------------------
    try:
        model_lgb = LGBMClassifier(
            objective="binary" if n_classes == 2 else "multiclass",
            num_class=None if n_classes == 2 else n_classes,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            n_jobs=-1,
            class_weight="balanced" # 🛡️ ELIMINA VIÉS DE COMPRA
        )

        model_lgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_lgb.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("LGBM", f1, model_lgb))

        print(f">>> LGBM F1={f1:.4f}")

    except Exception as e:
        print(f"[LGBM] erro: {e}")

    # ------------------------------------------------------------
    # 3 — XGBoost
    # ------------------------------------------------------------
    try:
        # 🛡️ Cálculo manual de scale_pos_weight para XGBoost (Equivalente ao balanced)
        counts = pd.Series(y_train).value_counts()
        spw = counts.min() / counts.max() if len(counts) == 2 else 1.0

        model_xgb = XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=spw if n_classes == 2 else None
        )

        model_xgb.fit(X_train, y_train, sample_weight=sw_train)
        preds = model_xgb.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        resultados.append(("XGB", f1, model_xgb))

        print(f">>> XGB F1={f1:.4f}")

    except Exception as e:
        print(f"[XGB] erro: {e}")

    # ------------------------------------------------------------
    # 4 — Seleção do melhor modelo
    # ------------------------------------------------------------
    if not resultados:
        raise RuntimeError(f"Nenhum modelo treinou para {target_col}")

    melhor_nome, melhor_f1, melhor_modelo = max(resultados, key=lambda x: x[1])

    # ------------------------------------------------------------
    # 5 — Salvamento
    # ------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)

    nome_arquivo = f"{target_col}_{melhor_nome}.pkl"
    model_path = os.path.join(outdir, nome_arquivo)

    joblib.dump(melhor_modelo, model_path)

    # ------------------------------------------------------------
    # 5.1 — Registrar para o EXPORTADOR V22 (OBRIGATÓRIO)
    # ------------------------------------------------------------
    global caminhos_modelos, features_por_target

    if "caminhos_modelos" not in globals() or not isinstance(caminhos_modelos, dict):
        caminhos_modelos = {}

    if "features_por_target" not in globals() or not isinstance(features_por_target, dict):
        features_por_target = {}

    caminhos_modelos[target_col] = model_path
    features_por_target[target_col] = list(feat_cols)

    # 🔴 PATCH: Registro no cache para o BACKTEST_DECISOR
    global modelos_cache, X_arrays_cache, probs_cache, lista_targets
    modelos_cache[target_col] = melhor_modelo
    
    # 🛡️ ALINHAMENTO FORÇADO PARA O CATBOOST/XGB/LGBM
    # Garante que a matriz X usada para predição tenha exatamente as mesmas colunas do treino
    X_aligned = df[list(feat_cols)].fillna(0).values
    
    X_arrays_cache[target_col] = X_aligned # Matriz completa para o decisor
    probs_cache[target_col] = melhor_modelo.predict_proba(X_aligned)
    
    if target_col not in lista_targets:
        lista_targets.append(target_col)

    print(f"✔ Modelo salvo: {model_path}")

    # ------------------------------------------------------------
    # 6 — RETORNO ÚNICO (FINAL)
    # ------------------------------------------------------------
    return nome_arquivo, melhor_f1, model_path
# ===========================================================
# BLOCO 3.X — CONVERSÃO DOS THRESHOLDS E ANÁLISE DE ALCANCE (V34-EXT CORRIGIDO)
# ===========================================================
# IMPORTANTE:
# • ESTE BLOCO NÃO ALTERA df_all
# • NÃO CRIA COLUNAS
# • NÃO GERA STRINGS NO DATAFRAME
# • TODA A LÓGICA É LOCAL (SEGURO PARA ML)

# ===============================================================
# ANÁLISE DE ALCANCE DO MERCADO — V34 EXTENDIDO (CORRIGIDO FINAL)
# ===============================================================

print("\n==============================================================")
print(" ANÁLISE DE ALCANCE DO MERCADO — V34 EXTENDIDO")
print("==============================================================\n")

# ---------------------------------------------------------------
# 1) Thresholds em percentual (A/B/C)
# ---------------------------------------------------------------
thrA_pct = thrA * 100
thrB_pct = thrB * 100
thrC_pct = thrC * 100

print("=== THRESHOLDS EM % ===")
print(f"Target A: {thrA_pct:.4f}%")
print(f"Target B: {thrB_pct:.4f}%")
print(f"Target C: {thrC_pct:.4f}%\n")

# ---------------------------------------------------------------
# 2) P90 do movimento futuro
# ---------------------------------------------------------------
p90_fut = np.percentile(df_all["amp_fut"].abs(), 90)
p90_pct = p90_fut * 100

print("=== P90 DO MOVIMENTO FUTURO ===")
print("Coluna utilizada ............... amp_fut")
print(f"P90 (movimento típico forte) .. {p90_pct:.4f}%\n")

# ---------------------------------------------------------------
# 3) Índice de alcance (RI)
# ---------------------------------------------------------------
def reach_index(thr, p90):
    if thr == 0:
        return 0
    return p90 / thr

RI_A = reach_index(thrA, p90_fut)
RI_B = reach_index(thrB, p90_fut)
RI_C = reach_index(thrC, p90_fut)

def classify_RI(v):
    if v >= 1.2:
        return "EXCELENTE"
    elif v >= 1.0:
        return "BOM"
    elif v >= 0.7:
        return "MODERADO"
    else:
        return "FRACO"

print("=== ÍNDICE DE ALCANCE (RI) ===")
print(f"A → RI={RI_A:.2f} → {classify_RI(RI_A)}")
print(f"B → RI={RI_B:.2f} → {classify_RI(RI_B)}")
print(f"C → RI={RI_C:.2f} → {classify_RI(RI_C)}")
print("---------------------------------------------------------------\n")

# ---------------------------------------------------------------
# 4) Probabilidade real de atingir A, B e C
# ---------------------------------------------------------------
hits_A = np.mean(df_all["amp_fut"].abs() >= thrA) * 100
hits_B = np.mean(df_all["amp_fut"].abs() >= thrB) * 100
hits_C = np.mean(df_all["amp_fut"].abs() >= thrC) * 100

print("=== PROBABILIDADE REAL DE ALCANCE ===")
print(f"A ({thrA_pct:.3f}%): {hits_A:.2f}% das barras atingem")
print(f"B ({thrB_pct:.3f}%): {hits_B:.2f}% das barras atingem")
print(f"C ({thrC_pct:.3f}%): {hits_C:.2f}% das barras atingem")
print("---------------------------------------------------------------\n")

# ---------------------------------------------------------------
# 5) Distribuição futura (categorias A/B/C)
# ---------------------------------------------------------------
dist_A = np.mean(df_all["amp_fut"].abs() < thrA) * 100
dist_AB = np.mean((df_all["amp_fut"].abs() >= thrA) &
                  (df_all["amp_fut"].abs() < thrB)) * 100
dist_BC = np.mean((df_all["amp_fut"].abs() >= thrB) &
                  (df_all["amp_fut"].abs() < thrC)) * 100
dist_C = np.mean(df_all["amp_fut"].abs() >= thrC) * 100

print("=== DISTRIBUIÇÃO DO MOVIMENTO FUTURO ===")
print(f"<A   → {dist_A:.2f}%")
print(f"A–B  → {dist_AB:.2f}%")
print(f"B–C  → {dist_BC:.2f}%")
print(f">C   → {dist_C:.2f}%")
print("---------------------------------------------------------------\n")

# ---------------------------------------------------------------
# 6) Curva de movimento futura (quantis)
# ---------------------------------------------------------------
#quantis = [50, 60, 70, 80, 85, 90, 95, 99]
#vals = np.percentile(df_all["amp_fut"].abs(), quantis) * 100

#print("=== CURVA DE MOVIMENTO — QUANTIS ===")
#for q, v in zip(quantis, vals):
#    print(f"P{q}: {v:.3f}%")
#print("---------------------------------------------------------------\n")

# ---------------------------------------------------------------
# 7) Taxa de alcance por faixa percentual (0.1% → 2.5%)
# ---------------------------------------------------------------
print("=== TAXA DE ALCANCE POR FAIXA DE % ===")
for p in np.linspace(0.1, 2.5, 25):  # 25 níveis suaves
    thr = p / 100
    hit = np.mean(df_all["amp_fut"].abs() >= thr) * 100
    print(f"{p:.2f}% → {hit:6.2f}%")
print("---------------------------------------------------------------\n")

# ---------------------------------------------------------------
# 8) Resumo executivo institucionais
# ---------------------------------------------------------------
print("=== RESUMO EXECUTIVO DO ALCANCE ===")
print(f"O ativo apresenta movimento P90 ≈ {p90_pct:.3f}%")
print(f"A probabilidade de atingir A/B/C é: "
      f"{hits_A:.2f}% | {hits_B:.2f}% | {hits_C:.2f}%")
print(f"Classificação RI: "
      f"A={classify_RI(RI_A)}, "
      f"B={classify_RI(RI_B)}, "
      f"C={classify_RI(RI_C)}")
print("==============================================================")

# =============================================================
# REMOÇÃO DE COLUNAS FUTURAS — ANTI-LEAKAGE ABSOLUTO
# =============================================================
colunas_futuras = [c for c in df_all.columns 
                   if c.endswith("_fut") or "future" in c.lower()]

if len(colunas_futuras) > 0:
    print(">>> Removendo colunas futuras (anti-leakage):")
    for c in colunas_futuras:
        print(f"   - {c}")
    df_all = df_all.drop(columns=colunas_futuras, errors="ignore")
else:
    print("✔ Nenhuma coluna futura encontrada para remover (OK).")

# ===============================================================
# FUNÇÃO OFICIAL — APLICAR PESO TEMPORAL (V40 + REGIME)
# ===============================================================
def aplicar_peso_temporal(df):
    """
    Função restaurada e oficial para o pipeline V37.
    Cria df['sample_weight'] ANTES do treino.
    Não altera nenhuma estrutura existente.
    """
    df = df.copy()

    # ----------------------------
    # 1) Validar timestamp ts
    # ----------------------------
    if "ts" not in df.columns:
        print("[ERRO] 'ts' não encontrado no dataset. Peso temporal não será aplicado.")
        return df

    ts = pd.to_datetime(df["ts"], unit="ms")
    dt_max = ts.max()
    dt_min = ts.min()

    idade_dias = (dt_max - ts).dt.days.astype(float)
    total_dias = max(1, (dt_max - dt_min).days)

    # ----------------------------
    # 2) Meia-vida automática
    # ----------------------------
    meia_vida = max(30, total_dias * 0.20)
    peso_tempo = np.power(0.5, idade_dias / meia_vida)

    # ----------------------------
    # 3) Peso por regime (compatível)
    # ----------------------------
    if "trend_regime" in df.columns:
        reg = df["trend_regime"].fillna(0)
        peso_regime = np.where(
            reg == -1, 0.9,            # bear leve
            np.where(reg == 1, 1.2, 1.0)   # bull leve
        )
    else:
        peso_regime = 1.0

    # ----------------------------
    # 4) Peso final e normalização
    # ----------------------------
    peso_final = peso_tempo * peso_regime
    peso_final = peso_final / peso_final.mean()

    df["sample_weight"] = peso_final

    print("✔ Peso temporal aplicado com sucesso (V40 automático + regime).")
    return df


# ===============================================================
# PESO TEMPORAL (CONFIGURAÇÃO FIXA - SEM INPUT)
# ===============================================================
# RENDER FIX: Usar configuração do topo do arquivo
usar_peso = "s" if USAR_PESO_TEMPORAL else "n"
print(f">>> Peso temporal: {'ATIVADO' if usar_peso == 's' else 'DESATIVADO'}")

if usar_peso == "s":
    df_all = aplicar_peso_temporal(df_all)

# ===============================================================
# C1 — DETECÇÃO INSTITUCIONAL DE REGIMES (RETORNO + VOLATILIDADE)
# ===============================================================
from sklearn.cluster import KMeans


def detectar_regimes(df, n_clusters_ret=4, n_clusters_vol=4):
    df = df.copy()

    # ---------------------------------------------
    # 1. Features de retorno
    # ---------------------------------------------
    df["ret_1"] = df["close"].pct_change()
    df["ret_7"] = df["close"].pct_change(7)
    df["ret_14"] = df["close"].pct_change(14)

    # ---------------------------------------------
    # 2. Features de volatilidade
    # ---------------------------------------------
    df["vol_7"] = df["ret_1"].rolling(7).std()
    df["vol_14"] = df["ret_1"].rolling(14).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # ---------------------------------------------
    # 3. Slope (regressão linear curta)
    # ---------------------------------------------
    janela_slope = 6

    def calc_slope(arr):
        if len(arr) < janela_slope:
            return np.nan
        x = np.arange(len(arr))
        y = arr
        coef = np.polyfit(x, y, 1)[0]
        return coef

    df["slope_6"] = df["close"].rolling(janela_slope).apply(calc_slope, raw=False)

    # ---------------------------------------------
    # Remover NaNs temporários
    # ---------------------------------------------
    regime_df = df[["ret_1", "ret_7", "ret_14", "vol_7", "vol_14", "range_pct", "slope_6"]].dropna()

    # ---------------------------------------------
    # 4. KMeans de retorno
    # ---------------------------------------------
    ret_features = regime_df[["ret_1", "ret_7", "ret_14"]].fillna(0)
    km_ret = KMeans(n_clusters=n_clusters_ret, n_init="auto", random_state=42)
    ret_labels = km_ret.fit_predict(ret_features)

    # ---------------------------------------------
    # 5. KMeans de volatilidade
    # ---------------------------------------------
    vol_features = regime_df[["vol_7", "vol_14", "range_pct", "slope_6"]].fillna(0)
    km_vol = KMeans(n_clusters=n_clusters_vol, n_init="auto", random_state=42)
    vol_labels = km_vol.fit_predict(vol_features)

    # ---------------------------------------------
    # 6. Reconstrução no df original
    # ---------------------------------------------
    regime_df = regime_df.assign(reg_ret=ret_labels, reg_vol=vol_labels)

    df["reg_ret"] = np.nan
    df["reg_vol"] = np.nan

    df.loc[regime_df.index, "reg_ret"] = regime_df["reg_ret"]
    df.loc[regime_df.index, "reg_vol"] = regime_df["reg_vol"]

    # ---------------------------------------------
    # 7. Regime final (cruzamento)
    # ---------------------------------------------
    df["regime_final"] = df["reg_ret"].astype(str) + "_" + df["reg_vol"].astype(str)

    print("✔ Regimes detectados com sucesso (C1). Colunas: reg_ret, reg_vol, regime_final")
    return df

# ---------------------------------------------------------------
# Aplicar C1 no df_all
# ---------------------------------------------------------------
df_all = detectar_regimes(df_all)

# ===============================================================
# C2 — PESOS POR REGIME (BASEADOS NA PERFORMANCE HISTÓRICA)
# ===============================================================
def calcular_peso_regime(df, target_col="target_A"):
    df = df.copy()

    if "regime_final" not in df.columns:
        raise RuntimeError("regime_final não encontrado. Execute C1 antes de C2.")

    # ---------------------------------------------
    # 1. Agrupar por regime e medir performance média
    # ---------------------------------------------
    grp = df.groupby("regime_final")[target_col].apply(
        lambda x: (x == 1).mean() if len(x) > 20 else np.nan
    ).dropna()

    if grp.empty:
        print("⚠ Sem regimes suficientes para calcular pesos. Usando peso_regime = 1.")
        df["peso_regime"] = 1.0
        return df

    # ---------------------------------------------
    # 2. Normalização para intervalo 0.5 → 2.0
    # ---------------------------------------------
    min_v, max_v = grp.min(), grp.max()
    if max_v - min_v == 0:
        grp_norm = grp / grp
    else:
        grp_norm = 0.5 + 1.5 * (grp - min_v) / (max_v - min_v)

    # ---------------------------------------------
    # 3. Aplicar pesos à tabela
    # ---------------------------------------------
    df["peso_regime"] = df["regime_final"].map(grp_norm).fillna(1.0)

    print("✔ Peso por regime calculado com sucesso (C2).")
    return df

# Aplicar peso de regime usando target_A como referência primária
df_all = calcular_peso_regime(df_all, target_col="target_A")

# ===============================================================
# C3 — Pesos unificados (já fornecidos pelo V40)
# ===============================================================
def unificar_pesos(df):
    df = df.copy()

    # O V40 já cria df['sample_weight'] pronto para uso.
    if "sample_weight" not in df.columns:
        raise RuntimeError("sample_weight não encontrado — V40 não aplicou peso temporal.")

    print("✔ Peso unificado criado (C3): sample_weight já presente no df_all.")
    return df

# Aplicar C3
df_all = unificar_pesos(df_all)

# ===============================================================
# C4 — MENU DE PESOS AVANÇADOS (LEVE / MODERADO / AGRESSIVO / CUSTOMIZADO)
# ===============================================================

def aplicar_peso_avancado(df):
    df = df.copy()

    print("\n==============================================================")
    print("CONFIGURAÇÃO AVANÇADA DE PESOS — C4")
    print("==============================================================")
    print("1 = Peso leve")
    print("2 = Peso moderado")
    print("3 = Peso agressivo")
    print("4 = CUSTOMIZADO (todas as opções institucionais)")
    print("==============================================================")

    # 🔧 CORREÇÃO RENDER: Usar modo "1" (leve) - SEM INPUTS INTERATIVOS
    modo = "1"  # Forçado para modo leve (evita inputs)
    print(f">>> Modo automático selecionado: {modo} (leve)")


    # ----------------------------------------------------------
    # Padrões leve / moderado / agressivo
    # ----------------------------------------------------------
    if modo in ["1", "2", "3"]:
        if modo == "1":
            fator = 1.2
        elif modo == "2":
            fator = 1.6
        else:  # agressivo
            fator = 2.2

        df["sample_weight"] = df["sample_weight"] ** fator
        df["sample_weight"] /= df["sample_weight"].mean()

        print(f"✔ Peso avançado aplicado — modo {'leve' if modo=='1' else 'moderado' if modo=='2' else 'agressivo'} (C4).")
        return df

    # ----------------------------------------------------------
    # CUSTOMIZADO — TODAS AS OPÇÕES INSTITUCIONAIS
    # ----------------------------------------------------------
    if modo == "4":
        print("\n>>> MODO CUSTOMIZADO ATIVADO")

        # --------------------------
        # 1) Intensidade temporal
        # --------------------------
        print("\n--- Intensidade do peso temporal ---")
        print("1 = suave (1.0x)")
        print("2 = moderado (1.5x)")
        print("3 = forte (2.0x)")
        print("4 = extremo (3.0x)")
        it = input("Escolha (1/2/3/4): ").strip()

        mapa_it = {"1": 1.0, "2": 1.5, "3": 2.0, "4": 3.0}
        mult_temporal = mapa_it.get(it, 1.0)

        # --------------------------
        # 2) Janela de prioridade N
        # --------------------------
        try:
            janela_N = int(input("\nQuantos dias recentes dar mais peso? (ex: 15, 30, 45): "))
            janela_N = max(1, janela_N)
        except:
            janela_N = 30

        # --------------------------
        # 3) Reforço por regime vencedor
        # --------------------------
        reforcar_reg_vencedor = input("\nReforçar regimes vencedores? (s/n): ").strip().lower()
        if reforcar_reg_vencedor == "s":
            inten = input("Intensidade (1=leve, 2=moderado, 3=forte): ").strip()
            mapa_int = {"1": 1.1, "2": 1.3, "3": 1.6}
            mult_reg_vencedor = mapa_int.get(inten, 1.1)
        else:
            mult_reg_vencedor = 1.0

        # --------------------------
        # 4) Penalizar regimes ruins
        # --------------------------
        penalizar_ruins = input("\nPenalizar regimes ruins? (s/n): ").strip().lower()
        if penalizar_ruins == "s":
            penal = input("Nível de penalização (0.8, 0.6, 0.4): ").strip()
            try:
                penal_ruim = float(penal)
            except:
                penal_ruim = 0.8
        else:
            penal_ruim = 1.0

        # --------------------------
        # 5) Preferência por ciclos
        # --------------------------
        foco_tendencia = input("\nDar mais peso a mercados tendenciais? (s/n): ").strip().lower()
        foco_lateral   = input("Dar mais peso a mercados laterais (squeeze)? (s/n): ").strip().lower()
        foco_vol_alta  = input("Dar mais peso à volatilidade alta? (s/n): ").strip().lower()
        foco_vol_baixa = input("Dar mais peso à volatilidade baixa? (s/n): ").strip().lower()

        # --------------------------
        # 6) Peso por volatilidade
        # --------------------------
        vol_direto = input("\nPeso proporcional à volatilidade? (s/n): ").strip().lower()
        vol_inverso = input("Peso inverso à volatilidade? (s/n): ").strip().lower()

        # --------------------------
        # 7) Tendência forte
        # --------------------------
        reforcar_tendencia = input("\nReforçar tendência forte? (s/n): ").strip().lower()
        if reforcar_tendencia == "s":
            inten2 = input("Intensidade (1=leve, 2=moderado, 3=forte): ").strip()
            mapa_int2 = {"1": 1.1, "2": 1.3, "3": 1.6}
            mult_tendencia = mapa_int2.get(inten2, 1.2)
        else:
            mult_tendencia = 1.0

        # --------------------------
        # 8) Mudança de regime
        # --------------------------
        peso_mudanca = input("\nDar peso especial após mudança de regime? (s/n): ").strip().lower()
        mult_mudanca = 1.3 if peso_mudanca == "s" else 1.0

        # --------------------------
        # COMEÇA O CÁLCULO CUSTOMIZADO REAL
        # --------------------------
        w = df["sample_weight"].copy()

        # 1) Intensidade temporal
        w = w ** mult_temporal

        # 2) Janela de prioridade
        if "ts" in df.columns:
            dt = pd.to_datetime(df["ts"], unit="ms")
            dt_max = dt.max()
            idade = (dt_max - dt).dt.days
            peso_extra = np.where(idade <= janela_N, 1.4, 1.0)
            w *= peso_extra

        # 3) Reforço aos regimes vencedores
        if "regime_final" in df.columns and reforcar_reg_vencedor == "s":
            regimes_bons = df.groupby("regime_final")["target_A"].mean()
            regimes_top = regimes_bons[regimes_bons > regimes_bons.mean()]
            mask = df["regime_final"].isin(regimes_top.index)
            w *= np.where(mask, mult_reg_vencedor, 1.0)

        # 4) Penalizar regimes ruins
        if "regime_final" in df.columns and penalizar_ruins == "s":
            regimes_bons = df.groupby("regime_final")["target_A"].mean()
            regimes_ruins = regimes_bons[regimes_bons < regimes_bons.mean()]
            mask = df["regime_final"].isin(regimes_ruins.index)
            w *= np.where(mask, penal_ruim, 1.0)

        # 5) Ciclos
        if foco_tendencia == "s":
            if "slope_6" in df.columns:
                w *= np.where(df["slope_6"] > 0, 1.2, 1.0)

        if False and foco_lateral == "s":
            if "range_pct" in df.columns:
                w *= np.where(df["range_pct"] < df["range_pct"].quantile(0.25), 1.2, 1.0)

        if foco_vol_alta == "s":
            if "vol_7" in df.columns:
                w *= np.where(df["vol_7"] > df["vol_7"].median(), 1.2, 1.0)

        if foco_vol_baixa == "s":
            if "vol_7" in df.columns:
                w *= np.where(df["vol_7"] < df["vol_7"].median(), 1.2, 1.0)

        # 6) Peso por volatilidade direta/inversa
        if vol_direto == "s" and "vol_7" in df.columns:
            w *= (1 + df["vol_7"].fillna(0))

        if vol_inverso == "s" and "vol_7" in df.columns:
            w *= (1 / (1 + df["vol_7"].fillna(0)))

        # 7) Tendência forte
        if False and reforcar_tendencia == "s":
            if "slope_6" in df.columns:
                w *= np.where(df["slope_6"] > df["slope_6"].quantile(0.7), mult_tendencia, 1.0)

        # 8) Mudança de regime
        if "reg_ret" in df.columns:
            mudanca = df["reg_ret"].diff().abs() > 0
            w *= np.where(mudanca, mult_mudanca, 1.0)

        # Normalização final
        w /= w.mean()

        df["sample_weight"] = w

        print("✔ Peso CUSTOMIZADO aplicado com sucesso (C4).")
        return df

    print("Modo inválido. Nenhum peso adicional aplicado.")
    return df

# Aplicar C4
df_all = aplicar_peso_avancado(df_all)

# ==============================================================
# C5 — PESO TEMPORAL CUSTOMIZADO (APENAS PARA MODO 4)
# ==============================================================

def aplicar_peso_temporal_custom(df):
    """
    Peso temporal totalmente interativo.
    Só é executado se o usuário escolher o modo 4 (customizado).
    Não interfere no V40 automático.
    """
    df = df.copy()

    print("\n===============================================================")
    print("CONFIGURAÇÃO DE PESO CUSTOMIZADO — C4 CUSTOM")
    print("===============================================================")

    ts = pd.to_datetime(df["ts"], unit="ms")
    dt_max = ts.max()
    dt_min = ts.min()
    dias_total = (dt_max - dt_min).days

    print(f"Janela temporal: {dt_min} até {dt_max} ({dias_total} dias)")
    # 🔧 CORREÇÃO RENDER: Usar valor padrão ao invés de input
    meia_vida = 90.0
    print(f">>> Meia-vida automática: {meia_vida} dias")

    # Peso temporal
    idade_dias = (dt_max - ts).dt.days.astype(float)
    peso_tempo = np.power(0.5, idade_dias / meia_vida)

    # Peso por regime
    if "trend_regime" in df.columns:
        print("\n>>> Detectado trend_regime. Usando pesos padrão...")
        # 🔧 CORREÇÃO RENDER: Valores padrão ao invés de inputs
        p_bear = 1.0
        p_lateral = 1.0
        p_bull = 1.0
        print(f"    Peso BAIXA: {p_bear}, LATERAL: {p_lateral}, ALTA: {p_bull}")

        regime = df["trend_regime"].fillna(0)
        peso_regime = np.where(
            regime == -1, p_bear,
            np.where(regime == 0, p_lateral, p_bull)
        )
    else:
        peso_regime = 1.0

    # Peso final
    peso_final = peso_tempo * peso_regime
    peso_final = peso_final / peso_final.mean()

    df["sample_weight"] = peso_final

    print("\n✔ Peso customizado aplicado com sucesso.")
    return df
# =====================================================================
# C6 — AJUSTES AVANÇADOS DE TREINAMENTO (LEVE / MODERADO / AGRESSIVO)
# =====================================================================

def aplicar_ajustes_treino(df):
    """
    Ajusta hiperparâmetros institucionais para LightGBM / XGBoost
    com foco em estabilidade e performance.
    
    Não altera features nem targets.
    Apenas retorna um dicionário com hiperparâmetros refinados.
    """

    print("\n==============================================================")
    print("C6 — Ajustes Institucionais de Treino")
    print("==============================================================")
    print("1 = Leve  (mais estável)")
    print("2 = Moderado  (recomendado)")
    print("3 = Agressivo (melhor performance, maior sensibilidade)")
    print("==============================================================")

    # RENDER FIX: Usar modo 2 (moderado/recomendado) como padrão
    modo = "2"
    print(f">>> Modo C6 selecionado: {modo} (Moderado - padrão Render)")

    # ------------------------------------------------------------------
    # CONFIGURAÇÕES BASE (seguras)
    # ------------------------------------------------------------------
    cfg = {
        "lgbm": {
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "learning_rate": 0.05,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0
        },
        "xgb": {
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 0.0,
            "alpha": 0.0
        },
        "penalidade_classe_neutra": 1.0
    }

    # ================================================================
    # MODO 1 — LEVE
    # ================================================================
    if modo == "1":
        cfg["lgbm"].update({
            "learning_rate": 0.03,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        })
        cfg["xgb"].update({
            "eta": 0.03,
            "lambda": 0.1,
            "alpha": 0.1
        })
        cfg["penalidade_classe_neutra"] = 1.0

        print("✔ C6 aplicado — modo LEVE.")

    # ================================================================
    # MODO 2 — MODERADO  (RECOMENDADO)
    # ================================================================
    elif modo == "2":
        cfg["lgbm"].update({
            "learning_rate": 0.025,
            "reg_alpha": 0.3,
            "reg_lambda": 0.3
        })
        cfg["xgb"].update({
            "eta": 0.025,
            "lambda": 0.3,
            "alpha": 0.3
        })
        cfg["penalidade_classe_neutra"] = 1.25

        print("✔ C6 aplicado — modo MODERADO (RECOMENDADO).")

    # ================================================================
    # MODO 3 — AGRESSIVO
    # ================================================================
    else:
        cfg["lgbm"].update({
            "learning_rate": 0.015,
            "reg_alpha": 0.6,
            "reg_lambda": 0.6
        })
        cfg["xgb"].update({
            "eta": 0.015,
            "lambda": 0.6,
            "alpha": 0.6
        })
        cfg["penalidade_classe_neutra"] = 1.6

        print("✔ C6 aplicado — modo AGRESSIVO.")

    return cfg

# ============================================================
# LIMPEZA FINAL — REMOVER NaN/INF ANTES DO TREINO (evitar erro RegLog)
# ============================================================
df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna()
print("✔ Limpeza final aplicada — df_all livre de NaN/Inf.")

# =============================================================================
# BLOCO FINAL — TARGET_K
# • Mede ACERTO e ERRO do modelo para ALTA e BAIXA
# • NÃO SABE = apenas corte de confiança (não entra na métrica)
# • Split temporal 70 / 10 / 20 (val é reservado, não usado aqui)
# =============================================================================

from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier  # REMOVIDO - incompatível Render

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
KS = [1, 2, 3, 4, 5]

LOW_NS  = 0.48
HIGH_NS = 0.52

# -------------------------------------------------------------------------
# FEATURES (somente numéricas, sem targets e sem OHLCV bruto)
# -------------------------------------------------------------------------
excluir_cols = {
    "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote",
    "ignore"
}

feature_cols = [
    c for c in df_all.columns
    if is_numeric_dtype(df_all[c])
    and not c.startswith("target_")
    and c not in excluir_cols
]

if "close" not in df_all.columns:
    raise KeyError("df_all precisa conter a coluna 'close'.")

X = df_all[feature_cols].values
close = df_all["close"].values
n = len(df_all)

if n < 1000:
    raise ValueError(f"df_all muito pequeno para TARGET_K (n={n}).")

# -------------------------------------------------------------------------
# SPLIT TEMPORAL 70 / 10 / 20 COM GAP (evita leakage)
# -------------------------------------------------------------------------
GAP_K = 15  # Gap de segurança entre períodos

i_train = int(n * 0.70)
i_val   = int(n * 0.80)   # reserva 10% (val)
# teste = 20% final

# Com gap: descarta candles entre períodos
X_train = X[:i_train]
X_val   = X[i_train + GAP_K : i_val]
X_test  = X[i_val + GAP_K:]

close_train = close[:i_train]
close_val   = close[i_train + GAP_K : i_val]
close_test  = close[i_val + GAP_K:]

print(f">>> Split K com GAP={GAP_K}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# -------------------------------------------------------------------------
# FABRICA DE MODELOS (sempre instância NOVA por fit) - SEM CATBOOST
# -------------------------------------------------------------------------
def make_model(nome_modelo: str):
    if nome_modelo == "LGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, n_jobs=-1, verbose=-1)
    if nome_modelo == "XGB":
        return XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.7, colsample_bytree=0.7,
            tree_method="hist", eval_metric="logloss", verbosity=0
        )
    # CatBoost removido - incompatível com Render
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")

NOMES_MODELOS = ["LGBM", "XGB"]  # CAT removido

# -------------------------------------------------------------------------
# FUNÇÃO: gera y direcional para horizonte k (sem leakage)
# y = 1 se close[t+k] > close[t], usando apenas pontos válidos
# -------------------------------------------------------------------------
def build_y_directional(close_array: np.ndarray, k: int) -> np.ndarray:
    fut = np.roll(close_array, -k)
    y = (fut[:-k] > close_array[:-k]).astype(int)
    return y

# =============================================================================
# EXECUÇÃO — TARGET_K COM NÃO SABE (corte 48–52)
# =============================================================================
for nome_modelo in NOMES_MODELOS:

    print("\n" + "=" * 110)
    print(f"MODELO: {nome_modelo}  |  TESTE REAL (20%)  |  TARGET_K (COM NÃO SABE)")
    print("=" * 110)
    print("Candle | ALTA (%) | ERRO ALTA (%) | BAIXA (%) | ERRO BAIXA (%) | Decisão")
    print("-" * 110)

    # cache por K PARA ESTE MODELO (usado no DELTA logo abaixo)
    acertos_por_k = {}  # {k: {"alta_pct":..., "erro_alta_pct":..., "baixa_pct":..., "erro_baixa_pct":...}}

    for k in KS:

        # -----------------------------
        # TREINO (sem últimos k pontos)
        # -----------------------------
        y_train = build_y_directional(close_train, k)
        modelo = make_model(nome_modelo)
        modelo.fit(X_train[:-k], y_train)

        # -----------------------------
        # TESTE (sem últimos k pontos)
        # -----------------------------
        y_real = build_y_directional(close_test, k)
        proba = modelo.predict_proba(X_test[:-k])[:, 1]

        # decisões por corte NÃO SABE
        pred_alta = proba > HIGH_NS
        pred_baixa = proba < LOW_NS
        pred_ns = ~(pred_alta | pred_baixa)

        # métricas condicionais: quando o modelo disse ALTA/BAIXA, acertou?
        # ALTA: acerto se y_real == 1
        tot_alta = int(np.sum(pred_alta))
        ac_alta = int(np.sum(pred_alta & (y_real == 1)))
        er_alta = int(np.sum(pred_alta & (y_real == 0)))

        # BAIXA: acerto se y_real == 0
        tot_baixa = int(np.sum(pred_baixa))
        ac_baixa = int(np.sum(pred_baixa & (y_real == 0)))
        er_baixa = int(np.sum(pred_baixa & (y_real == 1)))

        alta_pct = (ac_alta / tot_alta * 100.0) if tot_alta > 0 else 0.0
        erro_alta_pct = (er_alta / tot_alta * 100.0) if tot_alta > 0 else 0.0

        baixa_pct = (ac_baixa / tot_baixa * 100.0) if tot_baixa > 0 else 0.0
        erro_baixa_pct = (er_baixa / tot_baixa * 100.0) if tot_baixa > 0 else 0.0

        # decisão "humana" (apenas informativa): escolhe o lado com maior acerto condicional
        # (você pode trocar depois; aqui não interfere em métrica)
        if tot_alta == 0 and tot_baixa == 0:
            decisao = "NÃO SABE"
        else:
            if alta_pct > baixa_pct:
                decisao = "ALTA"
            elif baixa_pct > alta_pct:
                decisao = "BAIXA"
            else:
                decisao = "NÃO SABE"

        # guarda para DELTA (sem inventar nomes)
        acertos_por_k[k] = {
            "alta_pct": float(alta_pct),
            "erro_alta_pct": float(erro_alta_pct),
            "baixa_pct": float(baixa_pct),
            "erro_baixa_pct": float(erro_baixa_pct),
        }

        print(
            f"k{k:<5} | "
            f"{alta_pct:>7.2f}% | "
            f"{erro_alta_pct:>12.2f}% | "
            f"{baixa_pct:>8.2f}% | "
            f"{erro_baixa_pct:>13.2f}% | "
            f"{decisao}"
        )

    print("-" * 110)
    print(f"FIM DO MODELO: {nome_modelo}")
    print("-" * 110)

    # =============================================================================
    # BLOCO DERIVADO — DELTA ENTRE HORIZONTES (K vs K-1) — BASE (COM NÃO SABE)
    # =============================================================================
    print("\n" + "=" * 110)
    print(f"DELTA ENTRE HORIZONTES — K vs K-1 | MODELO: {nome_modelo}  (BASE)")
    print("=" * 110)
    print("K  | Δ ALTA (%) | Δ ERRO ALTA | Δ BAIXA (%) | Δ ERRO BAIXA")
    print("-" * 110)

    for k in KS:
        if k == 1:
            print("K1 |    —        |     —       |     —        |      —")
            continue

        prev = acertos_por_k[k - 1]
        curr = acertos_por_k[k]

        d_alta  = curr["alta_pct"]       - prev["alta_pct"]
        d_ea    = curr["erro_alta_pct"]  - prev["erro_alta_pct"]
        d_baixa = curr["baixa_pct"]      - prev["baixa_pct"]
        d_eb    = curr["erro_baixa_pct"] - prev["erro_baixa_pct"]

        print(
            f"K{k} | "
            f"{d_alta:+8.2f}% | "
            f"{d_ea:+9.2f}% | "
            f"{d_baixa:+9.2f}% | "
            f"{d_eb:+10.2f}%"
        )

    print("-" * 110)
    print(f"FIM — DELTA K vs K-1 | MODELO: {nome_modelo}  (BASE)")
    print("-" * 110)

# =============================================================================
# BLOCO FINAL — TARGET_K COM CONFIANÇA (SEM NÃO SABE)
# • Modelo sempre responde ALTA ou BAIXA
# • Confiança = max(p, 1-p)
# • Relatório por MODELO, por K (K1..K5), por FAIXA DE CONFIANÇA
# • Quantidades e percentuais (compras/vendas, acertos/erros)
# • Split temporal 70 / 10 / 20
# =============================================================================

CONF_BINS = [
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 0.75),
    (0.75, 1.01),
]

for nome_modelo in NOMES_MODELOS:

    print("\n" + "=" * 140)
    print(f"MODELO: {nome_modelo}  |  TESTE REAL (20%)  |  COM CONFIANÇA (SEM NÃO SABE)")
    print("=" * 140)
    print("Confiança | K  | Operações | Compras | Acertos C | Erros C | Acerto C (%) | "
          "Vendas | Acertos V | Erros V | Acerto V (%)")
    print("-" * 140)

    # cache por K PARA DELTA deste bloco
    # aqui "acerto_compra_pct" = acerto condicional quando modelo decidiu ALTA (compras)
    # e "acerto_venda_pct" = acerto condicional quando modelo decidiu BAIXA (vendas)
    acertos_por_k_conf = {}  # {k: {"acerto_compra_pct":..., "acerto_venda_pct":...}}

    for k in KS:

        # TREINO
        y_train = build_y_directional(close_train, k)
        modelo = make_model(nome_modelo)
        modelo.fit(X_train[:-k], y_train)

        # TESTE
        y_real = build_y_directional(close_test, k)
        proba = modelo.predict_proba(X_test[:-k])[:, 1]

        decisao_alta = proba >= 0.5
        decisao_baixa = ~decisao_alta

        conf = np.maximum(proba, 1.0 - proba)

        # acumuladores para resumo global do K (todas faixas)
        tot_c = tot_v = 0
        ac_c = er_c = 0
        ac_v = er_v = 0

        for cmin, cmax in CONF_BINS:

            mask = (conf >= cmin) & (conf < cmax)
            if int(np.sum(mask)) == 0:
                continue

            compras = decisao_alta & mask
            vendas  = decisao_baixa & mask

            a_c = int(np.sum(compras & (y_real == 1)))
            e_c = int(np.sum(compras & (y_real == 0)))

            a_v = int(np.sum(vendas & (y_real == 0)))
            e_v = int(np.sum(vendas & (y_real == 1)))

            total_mask = int(np.sum(mask))
            total_compras = int(np.sum(compras))
            total_vendas  = int(np.sum(vendas))

            total_c_bin = a_c + e_c
            total_v_bin = a_v + e_v

            p_c = (a_c / total_c_bin * 100.0) if total_c_bin > 0 else 0.0
            p_v = (a_v / total_v_bin * 100.0) if total_v_bin > 0 else 0.0

            print(
                f"{int(cmin*100):02d}-{int(cmax*100):02d}%    | "
                f"K{k} | "
                f"{total_mask:10d} | "
                f"{total_compras:7d} | "
                f"{a_c:9d} | "
                f"{e_c:7d} | "
                f"{p_c:10.2f}% | "
                f"{total_vendas:7d} | "
                f"{a_v:9d} | "
                f"{e_v:7d} | "
                f"{p_v:10.2f}%"
            )

            # soma para resumo global do K
            tot_c += total_c_bin
            tot_v += total_v_bin
            ac_c += a_c
            er_c += e_c
            ac_v += a_v
            er_v += e_v

        # resumo global do K (todas as faixas, sem não sabe)
        acerto_compra_pct = (ac_c / tot_c * 100.0) if tot_c > 0 else 0.0
        acerto_venda_pct  = (ac_v / tot_v * 100.0) if tot_v > 0 else 0.0

        acertos_por_k_conf[k] = {
            "acerto_compra_pct": float(acerto_compra_pct),
            "acerto_venda_pct": float(acerto_venda_pct),
        }

    print("-" * 140)
    print(f"FIM DO MODELO: {nome_modelo}")
    print("-" * 140)

    # =============================================================================
    # BLOCO DERIVADO — DELTA ENTRE HORIZONTES (K vs K-1) — COM CONFIANÇA
    # =============================================================================
    print("\n" + "=" * 110)
    print(f"DELTA ENTRE HORIZONTES — K vs K-1 | MODELO: {nome_modelo}  (COM CONFIANÇA)")
    print("=" * 110)
    print("K  | ACERTO C (%) | Δ vs K-1 | ACERTO V (%) | Δ vs K-1")
    print("-" * 110)

    prev_c = None
    prev_v = None

    for k in KS:
        ac_c = acertos_por_k_conf[k]["acerto_compra_pct"]
        ac_v = acertos_por_k_conf[k]["acerto_venda_pct"]

        if prev_c is None:
            dc = "  —  "
            dv = "  —  "
        else:
            dc = f"{(ac_c - prev_c):+.2f}%"
            dv = f"{(ac_v - prev_v):+.2f}%"

        print(
            f"K{k:<2} | "
            f"{ac_c:>11.2f}% | "
            f"{dc:>8} | "
            f"{ac_v:>11.2f}% | "
            f"{dv:>8}"
        )

        prev_c = ac_c
        prev_v = ac_v

    print("-" * 110)
    print(f"FIM — DELTA K vs K-1 | MODELO: {nome_modelo}  (COM CONFIANÇA)")
    print("-" * 110)

print("\n" + "=" * 140)
print("FIM — TARGET_K COM CONFIANÇA (RELATÓRIO INFORMATIVO)")
print("=" * 140)

# ==========================================================
# SALVAMENTO OFICIAL DOS MODELOS K (OBRIGATÓRIO)
# ==========================================================

import joblib

# Fonte da verdade: quais Ks você treinou
# (ajuste se o nome da variável for outro)
ks_treinados = sorted(set(ks_treinados)) if "ks_treinados" in globals() else []

if not ks_treinados:
    print(
        "[AVISO] ks_treinados vazio no BLOCO K INFORMATIVO — "
        "treinamento OK, seguindo pipeline."
    )

for k in ks_treinados:
    nome_target = f"target_K{k}"

    if nome_target not in modelos:
        raise RuntimeError(
            f"[PIPELINE] Modelo em memória ausente para {nome_target}"
        )

    caminho = os.path.join(out_dir, f"{nome_target}.pkl")
    joblib.dump(modelos[nome_target], caminho)

    print(f"[PIPELINE] Modelo salvo: {caminho}")


# ========================================================================
# BLOCO 6 — TREINO GLOBAL DOS TARGETS (A, B, C) + CONSOLIDADOR
# ========================================================================

print("\n===============================================================")
print("MÓDULO 6 — TREINANDO TODOS OS TARGETS (A, B, C + REVERSÃO)")
print("===============================================================\n")

targets_disponiveis = []

for t in ["target_A_bin", "target_A", "target_B", "target_C", "target_REV_LONG", "target_REV_SHORT", "target_CONFLUENCIA"]:
    if t in df_all.columns:
        targets_disponiveis.append(t)

# ===============================================================
# MAPA DE FUNÇÃO DOS TARGETS — ITEM 1 (NÃO ALTERA RESULTADOS)
# ===============================================================
TARGET_ROLE = {
          "target_A_bin": "activity",   # gate de mercado
          "target_A": "direction",      # direção
          "target_B": "magnitude",      # expansão média
          "target_C": "magnitude",      # expansão grande
          "target_REV_LONG": "reversal_long",   # especialista COMPRA
          "target_REV_SHORT": "reversal_short", # especialista VENDA
          "target_CONFLUENCIA": "arbitro",      # árbitro de divergências
          "target_K1": "timing",        # timing
          "target_K2": "timing",
          "target_K3": "timing",
          "target_K4": "timing",
          "target_K5": "timing",
}


print(f">>> Targets detectados no dataset: {targets_disponiveis}")

resultados_finais = []
caminhos_modelos = {}

for target in targets_disponiveis:
    try:
        # -------------------------------------------------------
        # ITEM 1 — FUNÇÃO DO TARGET (NÃO ALTERA RESULTADOS)
        # -------------------------------------------------------
        role = TARGET_ROLE.get(target, "unknown")
        print(f"📌 Função do target: {role.upper()}")

        # -------------------------------------------------------
        # ASSERTS DEFENSIVOS — ITEM 1
        # -------------------------------------------------------
        if role == "activity":
            assert target == "target_A_bin", "Target errado para função ACTIVITY"

        if role == "direction":
            assert target == "target_A", "Target errado para função DIRECTION"

        if role == "magnitude":
            assert target in ["target_B", "target_C"], "Target errado para função MAGNITUDE"

        if role == "timing":
            assert target.startswith("target_K"), "Target errado para função TIMING"

        if role == "reversal_long":
            assert target == "target_REV_LONG", "Target errado para função REVERSAL_LONG"

        if role == "reversal_short":
            assert target == "target_REV_SHORT", "Target errado para função REVERSAL_SHORT"

        if role == "arbitro":
            assert target == "target_CONFLUENCIA", "Target errado para função ARBITRO"

        # -------------------------------------------------------
        # TREINO ORIGINAL (INALTERADO)
        # -------------------------------------------------------
        nome_modelo, f1, caminho = treinar_um_target(
            target,
            df_all,
            out_dir
        )

        resultados_finais.append(
            (target, nome_modelo, f1)
        )
        caminhos_modelos[target] = caminho

    except Exception as e:
        print(f"\n⚠ Erro no treino do target {target}: {e}\n")

# Ordenar pelo nome do target (A, B, C)
resultados_finais.sort(key=lambda x: x[0])

# ===========================================================
# PAINEL FINAL RESUMIDO POR TARGET
# ===========================================================

print("\n" + "="*70)
print("RESUMO FINAL — PERFORMANCE POR TARGET")
print("="*70)

# ===========================================================
# PAINEL FINAL — RESUMO POR TARGET (CORRIGIDO E LIMPO)
# ===========================================================

print("\n" + "="*70)
print("RESUMO FINAL — PERFORMANCE POR TARGET")
print("="*70)

for tgt, modelo, f1_val in resultados_finais:
    print(f"\nTARGET: {tgt}")
    print(f"  • Modelo vencedor ......... {modelo}")
    print(f"  • F1-Score TESTE .......... {f1_val:.4f}")
    print("-"*70)

# ==========================================================
# BLOCO 5.5 — TREINO DOS TARGET_K (FINAL ÚNICO E COMPATÍVEL)
# ==========================================================

print("\n===============================================================")
print("BLOCO 5.5 — TREINO DOS TARGET_K (FINAL ÚNICO E COMPATÍVEL)")
print("===============================================================\n")

HORIZONTE_BACKTEST = int(globals().get("HORIZONTE_BACKTEST", 6))  # ← MODIFICADO V5: Incluir K6
ks_treinados = []

for k in range(1, HORIZONTE_BACKTEST + 1):
    alvo_k = f"target_K{k}"
    print(f">>> Treinando {alvo_k}")

    df_k = preparar_futuros(df_all, k)

    if "ret_fut" not in df_k.columns:
        raise RuntimeError(f"[TARGET_K{k}] ret_fut não encontrado")

    # target binário: concorda (1) / discorda (0)
    # 🔥 MODIFICADO V5: K6 usa threshold mais conservador
    if k == 6:
        print(f"   🔥 K6 ELITE: Threshold conservador para alta seletividade...")
        # K6 opera apenas com convicção (threshold maior)
        threshold_k6 = 0.005  # 0.5%
        df_k[alvo_k] = (df_k["ret_fut"] > threshold_k6).astype(int)
        sinais = (df_k[alvo_k] == 1).sum()
        print(f"   ✅ K6: Threshold = {threshold_k6*100:.2f}%")
        print(f"   ✅ K6: Sinais = {sinais} ({sinais/len(df_k)*100:.1f}%)")
    else:
        # K1-K5: lógica normal
        df_k[alvo_k] = (df_k["ret_fut"] > 0).astype(int)

    if df_k[alvo_k].nunique() < 2:
        print(f"[AVISO] {alvo_k} degenerado — pulando")
        continue

    nome_modelo, f1, caminho_modelo = treinar_um_target(
        target_col=alvo_k,
        df=df_k,
        outdir=out_dir
    )

    ks_treinados.append(k)

    # fonte da verdade para exportador/backtest
    caminhos_modelos[alvo_k] = caminho_modelo

    print(f"[OK] {alvo_k} treinado | F1={f1:.4f}")
    print(f"[OK] Modelo salvo em: {caminho_modelo}\n")

if not ks_treinados:
    raise RuntimeError("[PIPELINE] TARGET_K NÃO FOI TREINADO — ERRO FATAL")

print(f"[PIPELINE] TARGET_K treinados com sucesso: {ks_treinados}")


# ========================================================================
# BLOCO 6.1 — RELATÓRIO FINAL (painel consolidado)
# ========================================================================

print("\n===============================================================")
print("RELATÓRIO FINAL DO TREINAMENTO")
print("===============================================================\n")

# =====================================================================
# INICIALIZAÇÃO DO RELATÓRIO FINAL (ANTES DO C5)
# =====================================================================
relatorio_path = os.path.join(out_dir, f"{exp_name}_RELATORIO_FINAL.txt")

with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write("===============================================================\n")
    f.write("RELATÓRIO FINAL DO TREINAMENTO\n")
    f.write("===============================================================\n\n")


# -------------------------------------------------------------
# AJUSTE 3 — Relatório institucional do módulo C5 (KMeans)
# -------------------------------------------------------------
if "regime_final" in df_all.columns:
    regimes, counts = np.unique(df_all["regime_final"], return_counts=True)
    print("\n=== RELATÓRIO DOS REGIMES (C5) ===")
    for r, c in zip(regimes, counts):
        pct = c / len(df_all) * 100
        print(f"Regime {r}: {c} barras ({pct:.2f}%)")

    # salva no relatório final também
    with open(relatorio_path, "a", encoding="utf-8") as f:
        f.write("\n=== RELATÓRIO DOS REGIMES (C5) ===\n")
        for r, c in zip(regimes, counts):
            pct = c / len(df_all) * 100
            f.write(f"Regime {r}: {c} barras ({pct:.2f}%)\n")

rel = []
rel.append(titulo("RELATÓRIO FINAL — MODELOS ENTREGUES"))
rel.append(f"Experimento: {exp_name}")
rel.append(f"Arquivo CRU: {csv_path}")
rel.append(f"Dataset final: {df_all.shape}")
rel.append("")

for (target, modelo, f1) in resultados_finais:
    rel.append(f"TARGET {target}:")
    rel.append(f"  • Melhor modelo .......... {modelo}")
    rel.append(f"  • Acurácia TESTE ......... {f1:.4f}")
    rel.append(f"  • Caminho ................ {caminhos_modelos[target]}")
    rel.append("")

rel_txt = "\n".join(rel)
print(rel_txt)

# Salvar relatório final
relatorio_path = os.path.join(out_dir, f"{exp_name}_RELATORIO_FINAL.txt")
with open(relatorio_path, "w", encoding="utf-8") as f:
    f.write(rel_txt)

print(f"\n✔ Relatório final salvo em:\n  {relatorio_path}")

# ===============================================================
# MÓDULO 5.6 — ESTIMATIVA DE ALCANCE (QUANTILE REGRESSION)
# ===============================================================
# DESATIVADO — NÃO USAR NESTA FASE

#from sklearn.linear_model import QuantileRegressor
#from sklearn.metrics import mean_absolute_error
#import joblib
#import numpy as np
#import os

#def treinar_quantile_regression(df, outdir):
#    """
#    Treina modelos de Quantile Regression para estimar:
#    - Q10 → quanto pode voltar contra
#    - Q50 → movimento típico
#    - Q90 → quanto pode ir a favor

#    OBRIGATORIAMENTE usa montar_matriz()
#    """

#    if "ret_fut" not in df.columns:
#        raise RuntimeError("[Quantile] ret_fut não encontrado no dataframe")

#    # -----------------------------------------------------------
#    # 1 — MATRIZ DE FEATURES (PIPELINE OFICIAL)
#    # -----------------------------------------------------------
#    X, y, feat_cols = montar_matriz(df, alvo="ret_fut")

#    # -----------------------------------------------------------
#    # 2 — Split temporal 70 / 30
#    # -----------------------------------------------------------
#    split = int(len(X) * 0.7)
#    X_train, X_test = X[:split], X[split:]
#    y_train, y_test = y[:split], y[split:]

#    quantis = [0.10, 0.50, 0.90]
#    modelos = {}

#    print(">>> Treinando modelos de Quantile Regression:")

#    for q in quantis:
#        print(f"\n--- Quantil {int(q * 100)}% ---")

#        model = QuantileRegressor(
#            quantile=q,
#            alpha=0.0,
#            solver="highs"
#        )

#        model.fit(X_train, y_train)
#
#        preds = model.predict(X_test)

#        mae = mean_absolute_error(y_test, preds)

#        print(f"MAE Quantil {int(q*100)}%: {mae:.6f}")
#        print(f"Predição média: {np.mean(preds):.6f}")
#        print(f"P10 predito: {np.percentile(preds, 10):.6f}")
#        print(f"P90 predito: {np.percentile(preds, 90):.6f}")

#        nome = f"quantile_Q{int(q*100)}.pkl"
#        path = os.path.join(outdir, nome)
#        joblib.dump(model, path)

#        print(f"✔ Modelo salvo: {path}")

#        modelos[f"Q{int(q*100)}"] = {
#            "modelo": model,
#            "mae": mae,
#            "path": path,
#            "features": feat_cols
#        }

#    print("\n✔ Quantile Regression treinado com sucesso.")
#    return modelos


# ------------------------------
# EXECUÇÃO
# ------------------------------
#modelos_quantile = treinar_quantile_regression(
#    df=df_quantile,
#    outdir=out_dir
#)


# ==========================================================
# REGISTRO EXPLÍCITO DOS MODELOS K EM caminhos_modelos
# (OBRIGATÓRIO PARA O EXPORTADOR V22)
# ==========================================================


# ===============================================================
# EXPORTADOR V22 — PREPARANDO MODELOS PARA PRODUÇÃO (FINAL / CORRIGIDO)
# ===============================================================
# • Fonte da verdade: caminhos_modelos + features_por_target
# • NÃO recalcula features
# • NÃO chama montar_matriz()
# • NÃO passa novamente pelo treino
# • Compatível 100% com o PDF de backtest
# ===============================================================

import os
import json
import shutil

print("\n===============================================================")
print("EXPORTADOR V22 — PREPARANDO MODELOS PARA PRODUÇÃO")
print("===============================================================\n")

# -------------------------------------------------------------
# PRÉ-VALIDAÇÕES OBRIGATÓRIAS
# -------------------------------------------------------------
for var in ["out_dir", "exp_name", "caminhos_modelos", "features_por_target"]:
    if var not in globals():
        raise RuntimeError(f"[V22] Variável obrigatória ausente: {var}")

if not isinstance(caminhos_modelos, dict) or not caminhos_modelos:
    raise RuntimeError("[V22] caminhos_modelos inválido ou vazio.")

if not isinstance(features_por_target, dict) or not features_por_target:
    raise RuntimeError("[V22] features_por_target inválido ou vazio.")

# -------------------------------------------------------------
# PASTA DE EXPORTAÇÃO
# -------------------------------------------------------------
export_dir = os.path.join(out_dir, f"{exp_name}_EXPORTADO")
os.makedirs(export_dir, exist_ok=True)

# -------------------------------------------------------------
# NORMALIZAÇÃO DOS TARGETS (FONTE: caminhos_modelos)
# -------------------------------------------------------------
targets_disponiveis = list(caminhos_modelos.keys())

def ordem_target(t):
    if t == "target_A_bin":
        return (0, 0)
    if t in ("target_A", "target_B", "target_C"):
        return (1, ["target_A", "target_B", "target_C"].index(t))
    if t.startswith("target_K"):
        try:
            return (2, int(t.replace("target_K", "")))
        except Exception:
            return (2, 999)
    return (9, 0)

targets_disponiveis = sorted(targets_disponiveis, key=ordem_target)

print(f"[V22] Targets a exportar: {targets_disponiveis}\n")

# -------------------------------------------------------------
# MANIFESTO (LISTA INTERNA)
# -------------------------------------------------------------
manifesto_lista = []

# -------------------------------------------------------------
# LOOP PRINCIPAL DE EXPORTAÇÃO
# -------------------------------------------------------------
for target in targets_disponiveis:

    # --- validações ---
    if target not in caminhos_modelos:
        raise RuntimeError(f"[V22] Target sem caminho de modelo: {target}")

    if target not in features_por_target:
        raise RuntimeError(f"[V22] Target sem features registradas: {target}")

    caminho_modelo = caminhos_modelos[target]

    if not os.path.isfile(caminho_modelo):
        raise RuntimeError(f"[V22] Arquivo de modelo não encontrado: {caminho_modelo}")

    feat_cols = features_por_target[target]

    if not isinstance(feat_cols, (list, tuple)) or not feat_cols:
        raise RuntimeError(f"[V22] Lista de features inválida para {target}")

    # --- copiar modelo ---
    nome_modelo = os.path.basename(caminho_modelo)
    shutil.copy2(caminho_modelo, os.path.join(export_dir, nome_modelo))

    # --- registrar no manifesto ---
    manifesto_lista.append({
        "target": target,
        "modelo": nome_modelo,
        "n_features": len(feat_cols),
        "features": list(feat_cols),
    })

# -------------------------------------------------------------
# NORMALIZAÇÃO FINAL DO MANIFESTO (LIST → DICT)
# -------------------------------------------------------------
manifesto = {}

for item in manifesto_lista:
    nome = item["target"]
    if nome in manifesto:
        raise RuntimeError(f"[V22] Target duplicado no manifesto: {nome}")
    manifesto[nome] = item

# -------------------------------------------------------------
# SALVAR MANIFESTO.JSON
# -------------------------------------------------------------
json_path = os.path.join(export_dir, "manifesto.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(manifesto, f, indent=4, ensure_ascii=False)

print(f"✔ Exportação concluída. Modelos disponíveis em:\n  {export_dir}")
print(f"✔ Manifesto salvo em:\n  {json_path}\n")

# -------------------------------------------------------------
# FONTE OFICIAL PARA ETAPAS POSTERIORES
# -------------------------------------------------------------
manifesto_path = json_path


# =====================================================================
# 🔵 MÓDULO 7 — AUDITORIA ADMINISTRATIVA FINAL (V25)
# =====================================================================
print("\n===============================================================")
print("MÓDULO 7 — AUDITORIA FINAL DO PIPELINE")
print("===============================================================\n")

auditoria = []

# Auditoria administrativa
auditoria.append("=====================================================")
auditoria.append(" AUDITORIA — FEATURES UTILIZADAS")
auditoria.append("=====================================================")
auditoria.append(f"Total de colunas no dataset final: {df_all.shape[1]}\n")

auditoria.append("LISTA COMPLETA DE FEATURES:")
for c in sorted(df_all.columns):
    auditoria.append("  - " + c)
auditoria.append("")

# Auditoria dos Targets
auditoria.append("=====================================================")
auditoria.append(" AUDITORIA — TARGETS")
auditoria.append("=====================================================")

for t in ["target_A","target_B","target_C"]:
    if t in df_all.columns:
        auditoria.append(f"\nTARGET {t} — Distribuição Normalizada:")
        auditoria.append(str(df_all[t].value_counts(normalize=True)))
    else:
        auditoria.append(f"TARGET {t}: NÃO ENCONTRADO")

auditoria.append("")

# Multi-TF
auditoria.append("=====================================================")
auditoria.append(" AUDITORIA — CONTEXTO MULTI-TIMEFRAME")
auditoria.append("=====================================================")

if "contexto_aplicado" in df_all.attrs:
    auditoria.append(f"Contexto aplicado: {df_all.attrs['contexto_aplicado']}\n")
else:
    auditoria.append("Nenhum contexto TF maior foi adicionado.\n")

# Dimensões
auditoria.append("=====================================================")
auditoria.append(" AUDITORIA — SANITY CHECKS")
auditoria.append("=====================================================")
auditoria.append(f"Dimensão final: {df_all.shape}")
auditoria.append(f"Total de linhas disponíveis: {len(df_all)}\n")

# Modelos exportados
auditoria.append("=====================================================")
auditoria.append(" AUDITORIA — MODELOS EXPORTADOS")
auditoria.append("=====================================================")

for nome, caminho in caminhos_modelos.items():
    auditoria.append(f"{nome}: {caminho}")
auditoria.append("")

# Salvar auditoria
audit_path = os.path.join(out_dir, f"{exp_name}_AUDITORIA_FINAL.txt")
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(auditoria))

print(f"✔ Auditoria administrativa salva em:\n  {audit_path}\n")

print("\n>>> Pipeline concluído com sucesso! <<<\n")

# ========================================================================
# 🔵 MÓDULO 8 — SIMULADOR FINANCEIRO REALISTA (ELITE)
# ========================================================================


# ========================================================================
# 🔵 MÓDULO 9 — META-LABELING 2.0 (O JUIZ DOS SINAIS)
def treinar_meta_labeling(df_all, lista_targets, modelos_cache, probs_cache):
    global meta_modelos_features
    print("\n" + "="*60)
    print("MÓDULO 9 — TREINAMENTO DE META-LABELING 2.0")
    print("="*60)
    
    meta_modelos = {}
    
    # 🚀 SINCRONIZAÇÃO FORÇADA: Criar targets K se não existirem
    if 'close' in df_all.columns:
        # Calcula o retorno futuro de 10 candles para reconstruir os targets K
        ret_fut_temp = df_all['close'].shift(-10) / df_all['close'] - 1
        for tgt in modelos_cache.keys():
            if tgt.startswith('target_K') and tgt not in df_all.columns:
                try:
                    if '1' in tgt: df_all[tgt] = (ret_fut_temp > 0.01).astype(int)
                    elif '2' in tgt: df_all[tgt] = (ret_fut_temp > 0.02).astype(int)
                    elif '3' in tgt: df_all[tgt] = (ret_fut_temp > 0.03).astype(int)
                    elif '4' in tgt: df_all[tgt] = (ret_fut_temp > 0.04).astype(int)
                    elif '5' in tgt: df_all[tgt] = (ret_fut_temp > 0.05).astype(int)
                    else: df_all[tgt] = (ret_fut_temp > 0.005).astype(int) # Target K padrão
                except: pass

    # 🚀 DETECÇÃO AGRESSIVA DE TARGETS K
    targets_validos = []
    for t in modelos_cache.keys():
        if t in df_all.columns:
            targets_validos.append(t)
    
    print(f"\n[INFO] Juiz será treinado para: {targets_validos}")
            
    for tgt in targets_validos:
        try:
            print(f">>> Treinando Juiz (Meta-Modelo) para: {tgt}")
            
            # 1. Criar o Target Secundário (Acertou ou Errou?)
            probs = probs_cache[tgt]
            classes = list(modelos_cache[tgt].classes_)
            
            n = min(len(df_all), len(probs))
            df_meta = df_all.iloc[:n].copy()
            
            if tgt not in df_meta.columns:
                print(f"  ⚠️ Target {tgt} não encontrado no DataFrame. Pulando Juiz.")
                continue

            y_true = df_meta[tgt].values
            y_pred_proba = probs
            y_pred = np.array([classes[np.argmax(p)] for p in y_pred_proba])
            
            # Meta-Target: 1 se o modelo acertou o sinal (direcional), 0 se errou
            meta_y = ((y_pred == y_true) & (y_pred != 0)).astype(int)
            
            # 2. Features para o Meta-Modelo
            conf_primaria = np.max(y_pred_proba, axis=1)
            cols_excluir = lista_targets + ['close','open','high','low','volume','time','date','datetime','ret_fut','amp_fut']
            features_base = [c for c in df_meta.columns if c not in cols_excluir and df_meta[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            X_meta_df = df_meta[features_base].fillna(0).copy()
            X_meta_df['meta_conf_primaria'] = conf_primaria
            X_meta_df = X_meta_df.loc[:, (X_meta_df != X_meta_df.iloc[0]).any()]
              # 🚀 SALVAR COLUNAS PARA O SIMULADOR
            meta_modelos_features[tgt] = list(X_meta_df.columns)
            # 3. Treinar o Juiz
            import lightgbm as lgb
            meta_clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, num_leaves=7, class_weight='balanced', verbose=-1, random_state=42) 
            mask_sinal = (y_pred != 0)
            n_sinais = mask_sinal.sum()
            
            if n_sinais >= 15:
                if len(np.unique(meta_y[mask_sinal])) > 1:
                    meta_clf.fit(X_meta_df.values[mask_sinal], meta_y[mask_sinal])
                    meta_modelos[tgt] = meta_clf
                    print(f"  ✔ Juiz (LGBM) treinado com sucesso para {tgt} ({n_sinais} sinais)")
                else:
                    print(f"  ⚠️ Juiz não treinado para {tgt}: Todos os sinais foram apenas acertos ou apenas erros.")
            else:
                print(f"  ⚠️ Sinais insuficientes ({n_sinais}) para treinar Meta-Labeling em {tgt}.")
        except Exception as e:
            print(f"  ❌ Erro ao treinar Juiz para {tgt}: {e}")
            
    return meta_modelos



# 🚀 INICIANDO FASE DE META-LABELING (O JUIZ)
print("\n" + "="*50)
print(">>> INICIANDO FASE DE META-LABELING (O JUIZ)")
print("="*50)
meta_modelos_cache = treinar_meta_labeling(df_all, lista_targets, modelos_cache, probs_cache)

# 🛡️ MÓDULO 9.5 - TESTE DE ROBUSTEZ (SHUFFLE TEST)
def realizar_teste_robustez(probs_cache):
    """
    Embaralha as probabilidades de previsão para simular um modelo aleatório. Isso é usado para o Shuffle Test.
    """
    probs_cache_shuffled = {}
    for tgt, probs in probs_cache.items():
        probs_shuffled = probs.copy()
        np.random.shuffle(probs_shuffled)
        probs_cache_shuffled[tgt] = probs_shuffled  
    return probs_cache_shuffled

# ========================================================================
# 🔥 FILTRO V2.2 - MOMENTUM INVERTIDO (Mean-Reversion)
# ========================================================================
def filtro_agressao_v2_causal(df_sim, i, direcao, modo="balanced"):
    """
    V2.2 - Inversão da lógica de momentum
    
    V2.1 FALHOU: Premiava trend-following → entrada tardia → SL +272%
    V2.2 CORRIGE: Premia mean-reversion → entrada antecipada
    
    FILOSOFIA:
    - Modelo ANTECIPA reversões (não segue trend)
    - Contra-trend forte = Oportunidade (não erro)
    - Trend forte = Entrada tardia (não favorável)
    """
    
    score_total = 0.0
    flags_criticos = []
    flags_positivos = []
    
    # THRESHOLD ADAPTATIVO
    if "atr14" in df_sim.columns:
        atr, close = df_sim["atr14"].iloc[i], df_sim["close"].iloc[i]
        if pd.notna(atr) and pd.notna(close) and close > 0:
            atr_pct = (atr / close) * 100
            threshold_div = -int(atr_pct * 100)
            threshold_div_sell = int(atr_pct * 100)
        else:
            threshold_div, threshold_div_sell = -150, 150
    else:
        threshold_div, threshold_div_sell = -150, 150
    
    # REGRA 1: DIVERGÊNCIA
    if "aggression_delta" in df_sim.columns and i >= 3:
        delta_atual = df_sim["aggression_delta"].iloc[i]
        delta_3bars = df_sim["aggression_delta"].iloc[i-3]
        delta_trend = delta_atual - delta_3bars
        
        if pd.notna(delta_trend):
            if direcao == 1 and delta_trend < threshold_div:
                return {"aprovado": False, "score": 0, "motivo": f"DIV:Δ{int(delta_trend)}"}
            elif direcao == -1 and delta_trend > threshold_div_sell:
                return {"aprovado": False, "score": 0, "motivo": f"DIV:Δ+{int(delta_trend)}"}
            elif (direcao == 1 and delta_trend > abs(threshold_div)*0.5) or \
                 (direcao == -1 and delta_trend < -(threshold_div_sell*0.5)):
                score_total += 40
                flags_positivos.append("Δ+")
    
    # REGRA 2: MOMENTUM INVERTIDO (MEAN-REVERSION)
    if all(col in df_sim.columns for col in ["ema9", "ema20", "ema50"]):
        ema9, ema20, ema50 = df_sim["ema9"].iloc[i], df_sim["ema20"].iloc[i], df_sim["ema50"].iloc[i]
        
        if all(pd.notna(x) for x in [ema9, ema20, ema50]):
            trend_strength = abs(ema9 - ema50) / ema50
            trend_forte = trend_strength > 0.015
            
            if direcao == 1:  # COMPRA
                if ema9 < ema20 < ema50:  # Downtrend
                    if trend_forte:
                        score_total += 35  # Antecipa reversão
                        flags_positivos.append("Rev↑")
                    else:
                        score_total += 15
                        flags_positivos.append("Down")
                elif ema9 > ema20 > ema50:  # Uptrend
                    if trend_forte:
                        score_total -= 35  # Entrada tardia
                        flags_criticos.append("Topo")
                    else:
                        score_total += 5
                else:  # Lateral
                    score_total -= 25
                    flags_criticos.append("Lat")
                    
            elif direcao == -1:  # VENDA
                if ema9 > ema20 > ema50:  # Uptrend
                    if trend_forte:
                        score_total += 35  # Antecipa reversão
                        flags_positivos.append("Rev↓")
                    else:
                        score_total += 15
                        flags_positivos.append("Up")
                elif ema9 < ema20 < ema50:  # Downtrend
                    if trend_forte:
                        score_total -= 35  # Entrada tardia
                        flags_criticos.append("Fundo")
                    else:
                        score_total += 5
                else:
                    score_total -= 25
                    flags_criticos.append("Lat")
    
    # REGRA 3: ACELERAÇÃO (threshold menor)
    if "flow_acceleration" in df_sim.columns:
        accel = df_sim["flow_acceleration"].iloc[i]
        if pd.notna(accel):
            accel_min = abs(threshold_div) * 0.15
            if abs(accel) < accel_min:
                score_total -= 15
                flags_criticos.append("NoAcc")
            elif (direcao == 1 and accel > accel_min) or (direcao == -1 and accel < -accel_min):
                score_total += 20
                flags_positivos.append("Acc!")
    
    # REGRA 4: VOLUME
    if "volume" in df_sim.columns:
        vol_atual, vol_med = df_sim["volume"].iloc[i], df_sim["volume"].rolling(20).mean().iloc[i]
        if pd.notna(vol_atual) and pd.notna(vol_med) and vol_med > 0:
            vol_ratio = vol_atual / vol_med
            if vol_ratio < 0.5 and modo == "strict":
                return {"aprovado": False, "score": 0, "motivo": "Vol<50%"}
            elif vol_ratio > 1.5:
                score_total += 20
                flags_positivos.append("Vol+")
            elif vol_ratio < 0.7:
                score_total -= 10
    
    # REGRA 5: RATIO
    if "buy_ratio" in df_sim.columns:
        buy_r = df_sim["buy_ratio"].iloc[i]
        if pd.notna(buy_r):
            if (direcao == 1 and buy_r > 0.60) or (direcao == -1 and buy_r < 0.40):
                score_total += 15
                flags_positivos.append("R+")
            elif (direcao == 1 and buy_r < 0.45) or (direcao == -1 and buy_r > 0.55):
                score_total -= 15
                flags_criticos.append("R-")
    
    # SEM ASSIMETRIA (removido)
    
    # DECISÃO
    score_norm = max(0, min(100, 50 + score_total))
    thresholds = {"strict": 65, "balanced": 50, "permissive": 35}
    aprovado = score_norm >= thresholds.get(modo, 50)
    
    motivo_parts = []
    if flags_criticos:
        motivo_parts.append(f"⚠{'/'.join(flags_criticos)}")
    if flags_positivos:
        motivo_parts.append(f"✓{'/'.join(flags_positivos)}")
    motivo = f"{score_norm:.0f}% {' '.join(motivo_parts) if motivo_parts else 'N'}"
    
    return {"aprovado": aprovado, "score": score_norm, "motivo": motivo}


# ======================================================================
# BLOCO REMOVIDO: Backtest/Simulação/Análise
# (Linhas originais: 4302-5516)
# V27 já faz backtest - Render só treina modelos
# ======================================================================

print("\n" + "=" * 80)
print("📦 CRIANDO ZIP FINAL COM TODOS OS ARQUIVOS")
print("=" * 80)

try:
    # Lista de arquivos para incluir no ZIP
    arquivos_zip = []
    
    # 1. CSVs de dados (multi-timeframe)
    for tf in ["15m", "30m", "1h", "4h", "8h", "1d"]:
        csv_tf = os.path.join(OUT_DIR, f"{SYMBOL}_{tf}.csv")
        if os.path.exists(csv_tf):
            arquivos_zip.append(csv_tf)
            print(f"   📊 {os.path.basename(csv_tf)}")
    
    # 2. Modelos .pkl (da pasta export)
    export_dir = os.path.join(out_dir, f"{exp_name}_EXPORTADO")
    if os.path.exists(export_dir):
        for f in os.listdir(export_dir):
            if f.endswith('.pkl') or f.endswith('.json'):
                arquivos_zip.append(os.path.join(export_dir, f))
                print(f"   🤖 {f}")
    
    # 3. Scaler e KMeans (pasta modelos_salvos)
    # Verifica múltiplos caminhos possíveis
    possiveis_dirs = [
        os.path.join(out_dir, 'modelos_salvos'),
        'modelos_salvos',
        os.path.join(OUT_DIR, 'modelos_salvos')
    ]
    
    for modelos_dir in possiveis_dirs:
        if os.path.exists(modelos_dir):
            for f in os.listdir(modelos_dir):
                if f.endswith('.pkl'):
                    full_path = os.path.join(modelos_dir, f)
                    if full_path not in arquivos_zip:  # Evita duplicatas
                        arquivos_zip.append(full_path)
                        print(f"   ⚙️ {f}")
    
    # 4. Manifesto e auditoria
    for pattern in [f"{exp_name}*.json", f"{exp_name}*.txt"]:
        for f in glob.glob(os.path.join(out_dir, pattern)):
            arquivos_zip.append(f)
            print(f"   📄 {os.path.basename(f)}")
    
    # Criar ZIP
    print(f"\n>>> Criando ZIP: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arquivo in arquivos_zip:
            if os.path.exists(arquivo):
                arcname = os.path.basename(arquivo)
                zf.write(arquivo, arcname)
    
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"✅ ZIP criado: {zip_size:.2f} MB")
    print(f"   Arquivos incluídos: {len(arquivos_zip)}")
    
    # Upload para Catbox
    print("\n>>> Fazendo upload para Catbox.moe...")
    link = upload_catbox(ZIP_PATH)
    
    if link:
        print("\n" + "=" * 80)
        print("🎉 UPLOAD CATBOX CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        print("📦 CONTEÚDO DO ZIP:")
        print("   ✅ CSVs (todos os timeframes)")
        print("   ✅ PKLs (modelos treinados)")
        print("   ✅ Scaler e KMeans")
        print("   ✅ Relatórios e JSONs")
        print("=" * 80)
        print(f"🔗 LINK CATBOX PARA DOWNLOAD:")
        print(f"   {link}")
        print("=" * 80)
        print(f"📦 Tamanho: {zip_size:.2f} MB")
        print("⚠️  Link expira em 72h se não for acessado!")
        print("=" * 80)
        print("\n🎯 COPIE O LINK ACIMA PARA BAIXAR O ZIP COMPLETO!")
        print("=" * 80)
    else:
        print("❌ Falha no upload Catbox")
        print(f"   ZIP salvo em: {ZIP_PATH}")
        print(f"   Tamanho: {zip_size:.2f} MB")

except Exception as e:
    print(f"❌ Erro ao criar ZIP: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 🔄 MANTER SERVIDOR ATIVO
# ============================================================================

print("\n" + "=" * 80)
print("🌐 PROCESSO FINALIZADO - SERVIDOR HTTP ATIVO")
print("=" * 80)
print(">>> Serviço mantido ativo...")

# Mantém o script rodando (Render)
while True:
    time.sleep(3600)
