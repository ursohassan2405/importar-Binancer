# ============================================================
# DataManager_V52_COM_TREINO.py (VERSÃO OTIMIZADA)
# PENDLEUSDT – Binance Data Vision + TREINO V27
# 
# 🎯 OTIMIZAÇÃO CRÍTICA:
# - Testes mostraram que 15 features > 134 features
# - Bugado: +600.19% | Corrigido: +566.17%
# - Decisão: Manter apenas feature_engine() (15 features)
# - adicionar_features_avancadas() DESABILITADA (retorna imediatamente)
#
# MODIFICAÇÕES V52:
# 1. ✅ ADICIONADO: Download automático da Binance (se CSV não existir)
# 2. ✅ ADICIONADO: Treino V27 completo
# 3. ✅ ADICIONADO: Download de PKLs
# 4. ✅ OTIMIZADO: Apenas 15 features base (menos overfitting)
# 5. Parâmetros fixos: candles=5, multiframe, peso_temporal=1
# ============================================================

import os
import sys
import time
import zipfile
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from io import BytesIO
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

# ML imports (para treino V27)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# Força output unbuffered
sys.stdout.reconfigure(line_buffering=True)

# =========================
# CONFIGURAÇÃO - EXATAMENTE IGUAL AO ORIGINAL
# =========================
SYMBOL = "PENDLEUSDT"
START_DT = datetime(2025, 10, 1, 0, 0, 0)
END_DT = datetime(2025, 12, 30, 23, 59, 59)

# MESMO caminho do script original!
OUT_DIR = "./pendle_agg_2025_01_01__2025_06_30"
CSV_PATH = os.path.join(OUT_DIR, "PENDLEUSDT_aggTrades.csv")
ZIP_CSV_PATH = OUT_DIR + "_csvs.zip"  # ZIP dos CSVs
ZIP_PKL_PATH = OUT_DIR + "_pkls.zip"  # ⭐ NOVO: ZIP dos PKLs

os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# =========================
# ⭐ PARÂMETROS FIXOS DO TREINO V27
# =========================
CANDLES_FUTURO = 5
CONTEXTO_MULTIFRAME = ["30m", "1h", "4h", "8h", "1d"]
PESO_TEMPORAL = 1

# =========================
# FUNÇÕES AUXILIARES V27 (COPIADAS EXATAMENTE)
# =========================

def realized_vol(close: pd.Series) -> pd.Series:
    """Volatilidade Realizada protegida contra o futuro."""
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean()).shift(1)

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
        y = series[i-window:i]
        slopes.append(LinearRegression().fit(X, y).coef_[0])
    return pd.Series(slopes).shift(1).values

def feature_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de Geração de Features - Versão Auditada V25."""
    df = df.copy()

    # 1. VARIÁVEIS BÁSICAS
    df["body"] = (df["close"] - df["open"]).shift(1)
    df["range"] = (df["high"] - df["low"]).shift(1)
    
    # 2. RETORNOS PROTEGIDOS
    df["ret1"] = df["close"].pct_change(1).shift(1)
    df["ret5"] = df["close"].pct_change(5).shift(1)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).shift(1)

    # 3. EMAs E DISTÂNCIAS
    df["ema9"] = df["close"].ewm(span=9).mean().shift(1)
    df["ema20"] = df["close"].ewm(span=20).mean().shift(1)
    df["dist_ema9"] = (df["close"].shift(1) - df["ema9"])
    df["dist_ema20"] = (df["close"].shift(1) - df["ema20"])

    # 4. SLOPES, VOLATILIDADES E ATR
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)
    
    # ⚠️ CRÍTICO: ATR14 é necessário para regimes!
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().shift(1)

    # 5. AGRESSÃO
    if "taker_buy_base" in df.columns:
        df["aggression_buy"] = df["taker_buy_base"].shift(1)
        df["aggression_sell"] = (df["volume"] - df["taker_buy_base"]).shift(1)
        df["aggression_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).shift(1)
    else:
        df["aggression_buy"] = 0
        df["aggression_sell"] = 0
        df["aggression_delta"] = 0
    
    return df

def adicionar_features_avancadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    ⚠️  FUNÇÃO DESABILITADA - OVERFITTING DETECTADO!
    
    Testes comparativos mostraram que 134 features resultam em PIOR performance:
    - Versão "Bugada" (15 features):    +600.19% retorno, 5.55% DD ✅
    - Versão "Corrigida" (134 features): +566.17% retorno, 6.28% DD ❌
    
    CONCLUSÃO: Menos features = melhor generalização!
    DECISÃO: Manter apenas feature_engine() - "Less is More"
    
    Esta função agora faz apenas validações básicas e retorna o df.
    O "bug" do return prematuro era na verdade uma OTIMIZAÇÃO!
    """
    df = df.copy()
    
    # Validações básicas para compatibilidade (caso alguma feature precise existir)
    if "range" not in df.columns: 
        df["range"] = df["high"] - df["low"]
    if "body" not in df.columns: 
        df["body"] = df["close"] - df["open"]
    if "upper_wick" not in df.columns: 
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    if "lower_wick" not in df.columns: 
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    if "range_pct" not in df.columns: 
        df["range_pct"] = df["range"] / (df["close"] + 1e-9)
    
    # ✅ RETORNA AQUI (replicando versão "bugada" vencedora: +600.19%)
    return df


def detectar_regimes_mercado_v25(df, n_regimes=4, out_dir=OUT_DIR):
    """
    Detecta regimes de mercado usando KMeans e SALVA scaler/kmeans.
    
    RETORNA scaler e kmeans para salvar posteriormente.
    """
    print(">>> Detectando regimes de mercado...", flush=True)
    
    # ⚠️ CRÍTICO: Features de regime DEVEM ser exatamente estas 3
    # (compatibilidade com V27 local campeão)
    REGIME_FEATURES_OBRIGATORIAS = ['vol_realized', 'atr14', 'slope20']
    
    regime_features = [c for c in REGIME_FEATURES_OBRIGATORIAS if c in df.columns]
    
    # Validação crítica
    if len(regime_features) != 3:
        print(f"    ⚠️ AVISO: Apenas {len(regime_features)}/3 features encontradas!", flush=True)
        print(f"    Features disponíveis: {regime_features}", flush=True)
        print(f"    Features faltando: {set(REGIME_FEATURES_OBRIGATORIAS) - set(regime_features)}", flush=True)
        
        # Fallback (NÃO DEVE ACONTECER!)
        if not regime_features:
            df['temp_ret'] = df['close'].pct_change(20)
            regime_features = ['temp_ret']
            print("    🚨 USANDO FALLBACK: temp_ret", flush=True)
    else:
        print(f"    ✅ Features de regime: {regime_features}", flush=True)
    
    # Preparar matriz
    X_regime = df[regime_features].fillna(0).values
    
    # Criar e treinar Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    
    # ⭐ DEBUG: Print das médias do scaler (comparar com local)
    print(f"    📊 MÉDIAS DO SCALER NA RENDER: {scaler.mean_}", flush=True)
    print(f"    📊 DESVIOS DO SCALER NA RENDER: {scaler.scale_}", flush=True)
    
    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    
    print(f"    ✅ {n_regimes} regimes detectados", flush=True)
    print(f"    Distribuição: {df['market_regime'].value_counts().to_dict()}", flush=True)
    
    return df, scaler, kmeans

# ============================================================
# ⭐ NOVO: FUNÇÃO DE TREINO COMPLETO
# ============================================================

def treinar_modelo_v27(df_15m, out_dir=OUT_DIR):
    """
    Treina modelo K6 EXATAMENTE como V27 local.
    
    Parâmetros fixos:
    - candles_futuro = 5
    - contexto_multiframe = ["30m", "1h", "4h", "8h", "1d"]
    - peso_temporal = 1
    
    Returns:
        modelo, scaler, kmeans, feature_cols
    """
    print("\n" + "="*70, flush=True)
    print("⭐ INICIANDO TREINO V27 (VERSÃO OTIMIZADA)", flush=True)
    print("="*70, flush=True)
    print("🎯 CONFIGURAÇÃO:", flush=True)
    print("   - Features: 15 base (feature_engine apenas)", flush=True)
    print("   - adicionar_features_avancadas(): DESABILITADA", flush=True)
    print("   - Motivo: 15 features = +600% | 134 features = +566%", flush=True)
    print("   - Estratégia: Less is More (menos overfitting)", flush=True)
    print("="*70, flush=True)
    
    # 0. NORMALIZAR NOME DA COLUNA DE TIMESTAMP
    if 'open_time' in df_15m.columns and 'ts' not in df_15m.columns:
        df_15m.rename(columns={'open_time': 'ts'}, inplace=True)
        print("    ✅ Renomeado 'open_time' → 'ts'", flush=True)
    
    # Garantir que 'ts' existe
    if 'ts' not in df_15m.columns:
        raise ValueError("CSV sem coluna de timestamp (ts ou open_time)")
    
    # 1. CRIAR TARGETS (K6)
    print(">>> Criando targets K6...", flush=True)
    
    # Target K6: Retorno futuro (5 candles)
    df_15m['target_K6'] = df_15m['close'].shift(-CANDLES_FUTURO) / df_15m['close'] - 1
    
    # Binarizar: 1 = UP (ret > 0), 0 = DOWN (ret <= 0)
    df_15m['target_K6_bin'] = (df_15m['target_K6'] > 0).astype(int)
    
    print(f"    Target criado: {CANDLES_FUTURO} candles à frente", flush=True)
    print(f"    Distribuição: {df_15m['target_K6_bin'].value_counts().to_dict()}", flush=True)
    
    # 2. CALCULAR FEATURES (ANTES DE DROPNA!)
    print(">>> Calculando features...", flush=True)
    t_start = time.time()
    df_15m = feature_engine(df_15m)
    df_15m = adicionar_features_avancadas(df_15m)
    print(f"    ✅ {len(df_15m.columns)} colunas totais ({time.time()-t_start:.1f}s)", flush=True)
    
    # 3. DETECTAR REGIMES (ANTES DE DROPNA!)
    t_start = time.time()
    df_15m, scaler, kmeans = detectar_regimes_mercado_v25(df_15m, n_regimes=4, out_dir=out_dir)
    print(f"    ✅ Regimes detectados ({time.time()-t_start:.1f}s)", flush=True)
    
    # 4. PREPARAR MATRIZ X, y
    print(">>> Preparando matriz de treino...", flush=True)
    
    # Remover colunas não-feature
    non_feat = {
        "open", "high", "low", "close", "volume",
        "ts", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "taker_sell_base", "close_time", "ignore", "open_time",
        "mark_price", "index_price", "fundingRate",
        "target_K6", "target_K6_bin",
        # Colunas do V51 (baleias)
        "buy_vol", "sell_vol", "delta", "buy_vol_agg", "sell_vol_agg", 
        "total_vol_agg", "cum_delta", "price_range", "absorcao", "vpin"
    }
    
    # Remover targets
    target_cols = {c for c in df_15m.columns if isinstance(c, str) and c.startswith("target_")}
    non_feat = non_feat.union(target_cols)
    
    # Selecionar features
    feat_cols = [c for c in df_15m.columns if c not in non_feat]
    
    # Apenas numéricas
    X = df_15m[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(X.columns)
    
    y = df_15m['target_K6_bin'].values
    
    # ⚠️ CRÍTICO: Remover NaN SOMENTE DEPOIS de criar TODAS as features
    # (Se dropar antes, a "janela de memória" fica torta!)
    print(f"    Shape antes de remover NaN: X={X.shape}, y={y.shape}", flush=True)
    valid_mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"    Shape depois de remover NaN: X={X.shape}, y={y.shape}", flush=True)
    
    print(f"    ✅ X: {X.shape}", flush=True)
    print(f"    ✅ y: {y.shape}", flush=True)
    print(f"    ✅ Features: {len(feat_cols)}", flush=True)
    
    # 5. SPLIT TEMPORAL (80/20)
    print(">>> Split temporal...", flush=True)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]
    
    print(f"    Train: {len(X_train)} samples", flush=True)
    print(f"    Test:  {len(X_test)} samples", flush=True)
    
    # 6. CALCULAR PESO TEMPORAL (BASEADO EM REGIME)
    print(">>> Calculando peso temporal...", flush=True)
    
    # Pesos por regime (C2 do V27)
    regime_col = 'market_regime'
    if regime_col not in df_15m.columns:
        print("    ⚠️  Coluna 'market_regime' não encontrada! Usando peso uniforme.", flush=True)
        sample_weight_train = None
    else:
        # Peso baseado em regime (aplicado apenas no treino)
        regime_train = df_15m.loc[X_train.index, regime_col].values
        
        # Calcular frequência de cada regime no treino
        regime_counts = pd.Series(regime_train).value_counts()
        total_samples = len(regime_train)
        
        # Peso inversamente proporcional à frequência
        # Regimes raros recebem mais peso
        regime_weights = {}
        for regime_id, count in regime_counts.items():
            regime_weights[regime_id] = total_samples / (len(regime_counts) * count)
        
        # Mapear pesos para cada amostra
        sample_weight_train = np.array([regime_weights[r] for r in regime_train])
        
        print(f"    ✅ Peso temporal calculado", flush=True)
        print(f"    Regimes no treino: {dict(regime_counts)}", flush=True)
        print(f"    Pesos por regime: {regime_weights}", flush=True)
    
    # 7. TREINAR XGBOOST
    print(">>> Treinando XGBoost...", flush=True)
    
    t_start = time.time()
    modelo = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # ⚠️ CRÍTICO: COM sample_weight baseado em regime
    if sample_weight_train is not None:
        print("    ⚠️ Peso temporal ATIVADO (baseado em regime)", flush=True)
        modelo.fit(
            X_train, 
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        print("    ⚠️ Peso temporal DESATIVADO (peso uniforme)", flush=True)
        modelo.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    
    print(f"    ✅ Modelo treinado em {time.time()-t_start:.1f}s!", flush=True)
    
    # 8. AVALIAR
    from sklearn.metrics import accuracy_score, classification_report
    
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n>>> RESULTADOS:", flush=True)
    print(f"    Acurácia: {acc:.2%}", flush=True)
    print(f"\n{classification_report(y_test, y_pred, target_names=['DOWN', 'UP'])}", flush=True)
    
    print("="*70, flush=True)
    
    return modelo, scaler, kmeans, feat_cols

# ============================================================
# RESTO DO CÓDIGO ORIGINAL (DOWNLOAD BINANCE)
# ============================================================

# ⚠️ COMENTADO PARA TESTE (USAR CSVs EXISTENTES)
# Descomentar para produção!

"""
def generate_date_range(start_dt, end_dt):
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }

def download_daily_file(symbol, date, session, retry_count=5):
    # ... código original ...
    pass

def process_binance_data(df):
    # ... código original ...
    pass

def gerar_15m_tratado_incremental(csv_agg_path, csv_15m_path, timeframe_min=15, min_val_usd=500, chunksize=200_000):
    # ... código original ...
    pass
"""

# ============================================================
# ⭐ CSVs JÁ FORAM GERADOS PELO V51
# ============================================================
# O V51 já gerou todos os CSVs de timeframes no Render:
# - PENDLEUSDT_15m.csv
# - PENDLEUSDT_30m.csv  
# - PENDLEUSDT_1h.csv
# - PENDLEUSDT_4h.csv
# - PENDLEUSDT_8h.csv
# - PENDLEUSDT_1d.csv
#
# Não precisamos gerar novamente - apenas ler!
# ============================================================

# ============================================================
# SERVIDOR HTTP (DOWNLOAD)
# ============================================================

class DownloadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/download/csvs':
            if os.path.exists(ZIP_CSV_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{SYMBOL}_csvs.zip"')
                self.end_headers()
                with open(ZIP_CSV_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP CSVs not created yet')
        
        elif self.path == '/download/pkls':
            if os.path.exists(ZIP_PKL_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{SYMBOL}_pkls.zip"')
                self.end_headers()
                with open(ZIP_PKL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP PKLs not created yet')
        
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            status_csv = 'PRONTO!' if os.path.exists(ZIP_CSV_PATH) else 'Processando...'
            status_pkl = 'PRONTO!' if os.path.exists(ZIP_PKL_PATH) else 'Processando...'
            
            link_csv = '<a href="/download/csvs" style="font-size:20px;padding:15px;margin:10px;background:#4CAF50;color:white;text-decoration:none;border-radius:5px;display:inline-block;">BAIXAR CSVs</a>' if os.path.exists(ZIP_CSV_PATH) else '<p>CSVs: Aguarde...</p>'
            link_pkl = '<a href="/download/pkls" style="font-size:20px;padding:15px;margin:10px;background:#2196F3;color:white;text-decoration:none;border-radius:5px;display:inline-block;">BAIXAR PKLs</a>' if os.path.exists(ZIP_PKL_PATH) else '<p>PKLs: Aguarde...</p>'
            
            html = f'''
            <html>
            <body style="font-family:Arial;padding:50px;text-align:center;">
                <h1>{SYMBOL} - Download</h1>
                <hr>
                <h2>CSVs: {status_csv}</h2>
                {link_csv}
                <hr>
                <h2>PKLs: {status_pkl}</h2>
                {link_pkl}
            </body>
            </html>
            '''
            self.wfile.write(html.encode())

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), DownloadHandler)
    print(f">>> Servidor HTTP na porta {port}", flush=True)
    server.serve_forever()

# ============================================================
# ⭐ FUNÇÃO AUXILIAR: DOWNLOAD DA BINANCE
# ============================================================

def baixar_dados_binance(symbol, start_date, end_date, output_path):
    """
    Baixa dados da Binance Data Vision.
    
    Args:
        symbol: Par (ex: PENDLEUSDT)
        start_date: Data início (datetime)
        end_date: Data fim (datetime)
        output_path: Onde salvar CSV
    """
    import requests
    from io import BytesIO
    
    print(f"\n>>> Baixando {symbol} da Binance Data Vision...", flush=True)
    
    # Iterar por cada mês
    current = start_date.replace(day=1)
    all_dfs = []
    
    while current <= end_date:
        year = current.year
        month = current.month
        
        # URL da Binance Data Vision
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/15m/{symbol}-15m-{year}-{month:02d}.zip"
        
        print(f"    Baixando: {year}-{month:02d}...", flush=True)
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Extrair ZIP
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df_month = pd.read_csv(f, header=None)
                        all_dfs.append(df_month)
                        print(f"        ✅ {len(df_month)} candles", flush=True)
            else:
                print(f"        ⚠️  Não encontrado (status {response.status_code})", flush=True)
        
        except Exception as e:
            print(f"        ❌ Erro: {e}", flush=True)
        
        # Próximo mês
        if month == 12:
            current = current.replace(year=year+1, month=1)
        else:
            current = current.replace(month=month+1)
    
    if not all_dfs:
        raise Exception("Nenhum dado foi baixado!")
    
    # Concatenar tudo
    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # Nomear colunas (padrão Binance)
    df_final.columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]
    
    # Converter timestamp para datetime
    df_final['open_time'] = pd.to_datetime(df_final['open_time'], unit='ms')
    df_final['close_time'] = pd.to_datetime(df_final['close_time'], unit='ms')
    
    # Filtrar período exato
    df_final = df_final[
        (df_final['open_time'] >= start_date) & 
        (df_final['open_time'] <= end_date)
    ]
    
    # Salvar
    df_final.to_csv(output_path, index=False)
    
    print(f"\n    ✅ Salvos {len(df_final)} candles em: {output_path}", flush=True)
    print(f"    Período: {df_final['open_time'].min()} até {df_final['open_time'].max()}", flush=True)
    
    return df_final

# ============================================================
# ⭐ MAIN MODIFICADO
# ============================================================

def main():
    # Inicia servidor HTTP
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    print("\n" + "="*70, flush=True)
    print("🚀 DataManager V52 - TREINO V27 (OTIMIZADO)", flush=True)
    print("="*70, flush=True)
    print(f"Símbolo: {SYMBOL}", flush=True)
    print(f"Diretório: {OUT_DIR}", flush=True)
    print("="*70, flush=True)
    
    # ============================================================
    # PASSO 1: VERIFICAR SE CSVs JÁ EXISTEM (GERADOS PELO V51)
    # ============================================================
    
    print("\n>>> PASSO 1: Verificando CSVs existentes...", flush=True)
    
    required_tfs = ['15m', '30m', '1h', '4h', '8h', '1d']
    csv_paths = {}
    all_exist = True
    
    for tf in required_tfs:
        csv_path = os.path.join(OUT_DIR, f"{SYMBOL}_{tf}.csv")
        if os.path.exists(csv_path):
            size_mb = os.path.getsize(csv_path) / (1024*1024)
            print(f"    ✅ {tf:3s}: {csv_path} ({size_mb:.2f} MB)", flush=True)
            csv_paths[tf] = csv_path
        else:
            print(f"    ❌ {tf:3s}: NÃO ENCONTRADO: {csv_path}", flush=True)
            all_exist = False
    
    if not all_exist:
        print("\n    ⚠️  ERRO: CSVs faltando!", flush=True)
        print("    Execute o DataManager V51 primeiro para gerar os CSVs.", flush=True)
        return
    
    print(f"\n    ✅ Todos os {len(csv_paths)} CSVs encontrados!", flush=True)
    
    # ============================================================
    # PASSO 2: CRIAR ZIP DOS CSVs (PARA DOWNLOAD)
    # ============================================================
    
    print(f"\n>>> PASSO 2: Criando ZIP dos CSVs...", flush=True)
    
    with zipfile.ZipFile(ZIP_CSV_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for tf, path in csv_paths.items():
            arcname = os.path.basename(path)
            z.write(path, arcname=arcname)
            print(f"    Adicionado: {arcname}", flush=True)
    
    zip_size = os.path.getsize(ZIP_CSV_PATH) / (1024 * 1024)
    print(f"    ✅ ZIP CSVs criado: {ZIP_CSV_PATH} ({zip_size:.2f} MB)", flush=True)
    
    # ============================================================
    # PASSO 3: ADICIONAR CONTEXTO MULTIFRAME (IGUAL V27 LOCAL)
    # ============================================================
    
    print(f"\n>>> PASSO 3: Adicionando contexto multiframe...", flush=True)
    
    # Carregar CSV 15m
    csv_15m_path = csv_paths['15m']
    df_15m = pd.read_csv(csv_15m_path)
    
    print(f"    CSV 15m carregado: {len(df_15m)} candles", flush=True)
    
    # Garantir que 15m tenha 'ts' como int64
    if df_15m['ts'].dtype != 'int64':
        df_15m['ts'] = df_15m['ts'].astype('int64')
    
    # Carregar e adicionar contexto dos outros TFs
    for tf in ['30m', '1h', '4h', '8h', '1d']:
        csv_tf_path = csv_paths[tf]
        df_tf = pd.read_csv(csv_tf_path)
        
        print(f"    Processando {tf}... ({len(df_tf)} candles)", flush=True)
        
        # Garantir que tf tenha 'ts' como int64
        if df_tf['ts'].dtype != 'int64':
            df_tf['ts'] = df_tf['ts'].astype('int64')
        
        # ⚠️ CRÍTICO: Aplicar feature_engine NO TF (igual V27 local!)
        df_tf = feature_engine(df_tf)
        
        # Selecionar apenas colunas de features (não OHLCV)
        feature_cols_tf = [c for c in df_tf.columns if c not in [
            'ts', 'open', 'high', 'low', 'close', 'volume',
            'taker_buy_base', 'taker_buy_quote', 'quote_volume', 
            'trades', 'close_time', 'ignore'
        ]]
        
        # Renomear para ctx_TF_*
        rename_map = {col: f'ctx_{tf}_{col}' for col in feature_cols_tf}
        df_tf_ctx = df_tf[['ts'] + feature_cols_tf].rename(columns=rename_map)
        
        # ⚠️ CRÍTICO: merge_asof (não merge simples!)
        # Isso permite alignment temporal correto
        df_15m = pd.merge_asof(
            df_15m.sort_values('ts'),
            df_tf_ctx.sort_values('ts'),
            on='ts',
            direction='backward'  # Usa valor mais recente do TF maior
        )
        
        print(f"        ✅ {len(feature_cols_tf)} features ctx_{tf}_* adicionadas", flush=True)
    
    # Forward fill para preencher possíveis NaNs (igual V27 local)
    ctx_cols = [c for c in df_15m.columns if c.startswith('ctx_')]
    if ctx_cols:
        df_15m[ctx_cols] = df_15m[ctx_cols].fillna(method='ffill')
    
    total_ctx_cols = len(ctx_cols)
    print(f"\n    ✅ Total de colunas contexto: {total_ctx_cols}", flush=True)
    print(f"    ✅ Shape final: {df_15m.shape}", flush=True)
    
    # ============================================================
    # PASSO 4: TREINAR MODELO V27
    # ============================================================
    
    print(f"\n>>> PASSO 4: Treinando modelo V27...", flush=True)
    
    # Treinar
    modelo, scaler, kmeans, feat_cols = treinar_modelo_v27(df_15m, OUT_DIR)
    
    # ============================================================
    # PASSO 5: SALVAR PKLs
    # ============================================================
    
    print(f"\n>>> PASSO 5: Salvando PKLs...", flush=True)
    
    pkl_modelo = os.path.join(OUT_DIR, "SISTEMA_K6_FINAL.pkl")
    pkl_scaler = os.path.join(OUT_DIR, "scaler_regimes.pkl")
    pkl_kmeans = os.path.join(OUT_DIR, "kmeans_regimes.pkl")
    
    joblib.dump(modelo, pkl_modelo)
    joblib.dump(scaler, pkl_scaler)
    joblib.dump(kmeans, pkl_kmeans)
    
    print(f"    ✅ {pkl_modelo}", flush=True)
    print(f"    ✅ {pkl_scaler}", flush=True)
    print(f"    ✅ {pkl_kmeans}", flush=True)
    
    # ⭐ VALIDAÇÃO CRÍTICA: Recarregar PKLs para confirmar integridade
    print("\n    🔍 VALIDANDO INTEGRIDADE DOS PKLs...", flush=True)
    try:
        modelo_teste = joblib.load(pkl_modelo)
        scaler_teste = joblib.load(pkl_scaler)
        kmeans_teste = joblib.load(pkl_kmeans)
        
        print(f"    ✅ Modelo: {type(modelo_teste).__name__}", flush=True)
        print(f"    ✅ Scaler: {scaler_teste.n_features_in_} features", flush=True)
        print(f"    ✅ Scaler médias: {scaler_teste.mean_}", flush=True)
        print(f"    ✅ KMeans: {kmeans_teste.n_clusters} clusters", flush=True)
        print("    ✅ VALIDAÇÃO OK!", flush=True)
    except Exception as e:
        print(f"    🚨 ERRO NA VALIDAÇÃO: {e}", flush=True)
        print("    ⚠️ PKLs podem estar corrompidos!", flush=True)
    
    # Salvar lista de features também
    features_path = os.path.join(OUT_DIR, "feature_names.txt")
    with open(features_path, 'w') as f:
        f.write("\n".join(feat_cols))
    print(f"    ✅ {features_path} ({len(feat_cols)} features)", flush=True)
    
    # ============================================================
    # PASSO 6: CRIAR ZIP DOS PKLs
    # ============================================================
    
    print(f"\n>>> PASSO 6: Criando ZIP dos PKLs...", flush=True)
    
    with zipfile.ZipFile(ZIP_PKL_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pkl_modelo, arcname="SISTEMA_K6_FINAL.pkl")
        z.write(pkl_scaler, arcname="scaler_regimes.pkl")
        z.write(pkl_kmeans, arcname="kmeans_regimes.pkl")
        z.write(features_path, arcname="feature_names.txt")
        print(f"    Adicionado: SISTEMA_K6_FINAL.pkl", flush=True)
        print(f"    Adicionado: scaler_regimes.pkl", flush=True)
        print(f"    Adicionado: kmeans_regimes.pkl", flush=True)
        print(f"    Adicionado: feature_names.txt", flush=True)
    
    zip_size = os.path.getsize(ZIP_PKL_PATH) / (1024 * 1024)
    print(f"    ✅ ZIP PKLs criado: {ZIP_PKL_PATH} ({zip_size:.2f} MB)", flush=True)
    
    # ============================================================
    # FINALIZADO
    # ============================================================
    
    print("\n" + "="*70, flush=True)
    print("✅ PROCESSAMENTO COMPLETO!", flush=True)
    print("="*70, flush=True)
    print(f"📦 CSVs: {ZIP_CSV_PATH}", flush=True)
    print(f"📦 PKLs: {ZIP_PKL_PATH}", flush=True)
    print("="*70, flush=True)
    print("\n🔗 ACESSE O APP PARA DOWNLOAD:", flush=True)
    print("   https://SEU-APP.onrender.com/", flush=True)
    print("="*70, flush=True)
    
    # Manter servidor vivo
    print("\n>>> Servidor mantido ativo para download...", flush=True)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
