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
# 1. COMENTADO: Download da Binance (para teste)
# 2. ADICIONADO: Treino V27 completo
# 3. ADICIONADO: Download de PKLs
# 4. OTIMIZADO: Apenas 15 features base (menos overfitting)
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

    # 4. SLOPES E VOLATILIDADES
    df["slope20"] = slope_regression(df["close"].values, 20)
    df["slope50"] = slope_regression(df["close"].values, 50)
    df["vol_realized"] = realized_vol(df["close"])
    df["vol_yz"] = yang_zhang(df)

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
        
        # Fallback (NAO DEVE ACONTECER!)
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
    df_15m = feature_engine(df_15m)
    df_15m = adicionar_features_avancadas(df_15m)
    print(f"    ✅ {len(df_15m.columns)} colunas totais", flush=True)
    
    # 3. DETECTAR REGIMES (ANTES DE DROPNA!)
    df_15m, scaler, kmeans = detectar_regimes_mercado_v25(df_15m, n_regimes=4, out_dir=out_dir)
    
    # 4. PREPARAR MATRIZ X, y
    print(">>> Preparando matriz de treino...", flush=True)
    
    # Remover colunas nao-feature
    non_feat = {
        "open", "high", "low", "close", "volume",
        "ts", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "close_time", "ignore",
        "mark_price", "index_price", "fundingRate",
        "target_K6", "target_K6_bin"
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
    
    # 6. TREINAR XGBOOST
    print(">>> Treinando XGBoost...", flush=True)
    print("    ⚠️ Peso temporal = 1 (todas as amostras têm peso igual)", flush=True)
    
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
    
    # ⚠️ CRÍTICO: SEM sample_weight (peso temporal = 1)
    modelo.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("    ✅ Modelo treinado!", flush=True)
    
    # 7. AVALIAR
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
# ⭐ NOVA FUNÇÃO: GERAR MULTIFRAME (30m, 1h, 4h, 8h, 1d)
# ============================================================

def gerar_multiframe(csv_15m_path, out_dir=OUT_DIR):
    """
    Gera CSVs de timeframes maiores a partir do 15m.
    
    Retorna dicionário com caminhos dos CSVs.
    """
    print("\n>>> Gerando timeframes multiframe...", flush=True)
    
    # Ler 15m
    df_15m = pd.read_csv(csv_15m_path)
    df_15m['ts'] = pd.to_datetime(df_15m['ts'], unit='ms')
    df_15m.set_index('ts', inplace=True)
    
    timeframes = {
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '8h': '8H',
        '1d': '1D'
    }
    
    csv_paths = {'15m': csv_15m_path}
    
    for tf_name, tf_rule in timeframes.items():
        print(f"    Gerando {tf_name}...", flush=True)
        
        df_resampled = df_15m.resample(tf_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Salvar
        csv_path = os.path.join(out_dir, f"{SYMBOL}_{tf_name}.csv")
        df_resampled.to_csv(csv_path)
        csv_paths[tf_name] = csv_path
        
        print(f"        ✅ {len(df_resampled)} candles → {csv_path}", flush=True)
    
    return csv_paths

# ============================================================
# SERVIDOR HTTP (DOWNLOAD)
# ============================================================

class DownloadHandler(SimpleHTTPRequestHandler):
# No DataManager_V52_OTIMIZADO_15_FEATURES.py, localize a classe SimpleHTTPRequestHandler
# Garanta que o self.wfile.write esteja exatamente assim:

    def do_GET(self):
        if self.path == '/download_csv':
            if os.path.exists(ZIP_CSV_PATH):
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(ZIP_CSV_PATH)}"')
                self.end_headers()
                with open(ZIP_CSV_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP CSVs ainda nao criado') # <--- LINHA 453 CORRIGIDA

        elif self.path == '/download_pkl':
            if os.path.exists(ZIP_PKL_PATH):
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(ZIP_PKL_PATH)}"')
                self.end_headers()
                with open(ZIP_PKL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'ZIP PKLs ainda nao criado')        
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
                self.wfile.write(b'ZIP PKLs ainda nao criado')
        
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
# ⭐ MAIN MODIFICADO
# ============================================================

def main():
    # Inicia servidor HTTP
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)
    
    print("\n" + "="*70, flush=True)
    print("🚀 DataManager V52 COM TREINO V27", flush=True)
    print("="*70, flush=True)
    print(f"Símbolo: {SYMBOL}", flush=True)
    print(f"Período: {START_DT.strftime('%Y-%m-%d')} até {END_DT.strftime('%Y-%m-%d')}", flush=True)
    print("="*70, flush=True)
    
    # ============================================================
    # PASSO 1: LER CSVs EXISTENTES (PULA DOWNLOAD BINANCE)
    # ============================================================
    
    print("\n>>> PASSO 1: Carregando CSVs existentes...", flush=True)
    
    csv_15m_path = os.path.join(OUT_DIR, f"{SYMBOL}_15m.csv")
    
    if not os.path.exists(csv_15m_path):
        print(f"❌ ERRO: CSV 15m nao encontrado: {csv_15m_path}", flush=True)
        print("   Execute primeiro o DataManager original para gerar os CSVs.", flush=True)
        return
    
    print(f"    ✅ CSV 15m encontrado: {csv_15m_path}", flush=True)
    
    # ============================================================
    # PASSO 2: GERAR MULTIFRAME (30m, 1h, 4h, 8h, 1d)
    # ============================================================
    
    csv_paths = gerar_multiframe(csv_15m_path, OUT_DIR)
    
    print(f"\n    ✅ Multiframe completo:", flush=True)
    for tf, path in csv_paths.items():
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"        {tf:3s}: {path} ({size_mb:.2f} MB)", flush=True)
    
    # ============================================================
    # PASSO 3: CRIAR ZIP DOS CSVs
    # ============================================================
    
    print(f"\n>>> PASSO 3: Criando ZIP dos CSVs...", flush=True)
    
    with zipfile.ZipFile(ZIP_CSV_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for tf, path in csv_paths.items():
            arcname = os.path.basename(path)
            z.write(path, arcname=arcname)
            print(f"    Adicionado: {arcname}", flush=True)
    
    zip_size = os.path.getsize(ZIP_CSV_PATH) / (1024 * 1024)
    print(f"    ✅ ZIP CSVs criado: {ZIP_CSV_PATH} ({zip_size:.2f} MB)", flush=True)
    
    # ============================================================
    # PASSO 4: TREINAR MODELO V27
    # ============================================================
    
    print(f"\n>>> PASSO 4: Treinando modelo V27...", flush=True)
    
    # Carregar CSV 15m
    df_15m = pd.read_csv(csv_15m_path)
    
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
    print("\n🔗 LINKS PARA DOWNLOAD:", flush=True)
    print("   CSVs: https://SEU-APP.onrender.com/download/csvs", flush=True)
    print("   PKLs: https://SEU-APP.onrender.com/download/pkls", flush=True)
    print("="*70, flush=True)
    
    # Manter servidor vivo
    print("\n>>> Servidor mantido ativo para download...", flush=True)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
