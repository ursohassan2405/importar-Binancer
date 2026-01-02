import json
import os

import json
import os

def get_config():
    if os.path.exists('config_render.json'):
        with open('config_render.json', 'r') as f:
            return json.load(f)['trading_model']
    return None

render_config = get_config()

def input_render(prompt, default_val):
    if render_config:
        # Mapear prompts para chaves do config
        if "Arquivo CRU" in prompt: return render_config['input_file']
        if "Pasta de saída" in prompt: return render_config['output_dir']
        if "Nome do experimento" in prompt: return render_config['experiment_name']
        if "Horizonte futuro" in prompt: return str(render_config['target_horizon'])
        if "peso temporal" in prompt: return render_config['apply_temporal_weight']
        if "modo de peso" in prompt: return str(render_config['weight_mode'])
        if "contexto multi-TF" in prompt: return render_config['add_multi_tf']
    return input_render(prompt)
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

import os
import sys
import gc
import glob
import shutil
import json
from datetime import datetime

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

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

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin


import joblib

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
    Recebe algo como: "15m, 30m,1h ,4h"
    Retorna uma lista limpa: ["15m","30m","1h","4h"]
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


print("\n================= IA_CRIPTO — V25 FINAL REAL =================\n")

# ========================================================================
# INPUTS CONVERSACIONAIS — CONFIGURAÇÃO GERAL
# ========================================================================

csv_path = input_render("📌 Arquivo CRU (ex: C:\\BTC_MODELO\\IA_CRIPTO\\datasets\\BTCUSDT_15m_full_V14.csv): ").strip()
if not os.path.isfile(csv_path):
    print(f"\n❌ ERRO: Arquivo não encontrado: {csv_path}")
    sys.exit(1)

out_dir = input_render("📁 Pasta de saída para modelos/relatórios: ").strip()
if out_dir == "":
    out_dir = os.path.join(os.path.dirname(csv_path), "V25_MODELS")
os.makedirs(out_dir, exist_ok=True)

exp_name = input_render("🏷 Nome do experimento: ").strip()
if exp_name == "":
    base = os.path.splitext(os.path.basename(csv_path))[0]
    exp_name = base + "_V25"

print(f"\n✔ EXPERIMENTO: {exp_name}")
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

# Detectar timeframe base (V34)
nome_arq = os.path.basename(csv_path)
tf_base_detectado = nome_arq.split("_")[1]
tf_base_global = tf_base_detectado

tf_base_global = tf_base_detectado   # ← variável usada no Módulo 4
print(f"✔ Timeframe detectado: {tf_base_global}")

# Extrair símbolo do arquivo cru
simbolo = os.path.basename(csv_path).split("_")[0].replace(".csv", "")

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

def realized_vol(close: pd.Series) -> pd.Series:
    """Realized volatility — log-return rolling mean"""
    return np.sqrt((np.log(close / close.shift(1)) ** 2).rolling(20).mean())


def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """Parkinson — baseado em high/low"""
    return np.sqrt(
        (1.0 / (4 * np.log(2))) *
        (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()
    )


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """Rogers-Satchell — usa open/high/low/close"""
    rs = (
        np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) +
        np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"])
    )
    return np.sqrt(rs.rolling(20).mean())


def yang_zhang(df: pd.DataFrame) -> pd.Series:
    """Yang-Zhang — mais robusto a gaps"""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_oc = np.log(df["open"] / df["close"].shift(1))
    log_co = np.log(df["close"] / df["open"])

    rs = log_ho**2 + log_lo**2
    close_vol = log_co.rolling(20).std() ** 2
    open_vol = log_oc.rolling(20).std() ** 2

    return np.sqrt(0.34 * open_vol + 0.34 * close_vol + 0.27 * rs.rolling(20).mean())


# ------------------------------------------------------------------------
# 2.2 — ATR (Welles Wilder)
# ------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev).abs()
    tr3 = (df["low"] - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------------
# 2.3 — Slopes via Regressão Linear
# ------------------------------------------------------------------------

def slope_regression(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Inclinação da regressão linear em janelas deslizantes"""
    X = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * window
    for i in range(window, len(series)):
        y = series[i-window:i]
        slopes.append(LinearRegression().fit(X, y).coef_[0])
    return np.array(slopes)


# ------------------------------------------------------------------------
# 2.4 — Microestrutura e agressão
# ------------------------------------------------------------------------

def microstructure(df: pd.DataFrame) -> pd.DataFrame:
    if "taker_buy_base" in df.columns and "volume" in df.columns:
        buy = df["taker_buy_base"]
        sell = df["volume"] - df["taker_buy_base"]
        df["delta"] = buy - sell
        df["aggression"] = df["delta"] / (df["volume"] + 1e-9)
        df["imbalance"] = buy / (df["volume"] + 1e-9)
    else:
        df["delta"] = np.nan
        df["aggression"] = np.nan
        df["imbalance"] = np.nan
    return df


# =====================================================================
# FEATURE ENGINE — V90 + V92 + AVANÇADAS (LIMPO, SEM DUPLICIDADES)
# =====================================================================

def feature_engine(df):
    df = df.copy()

    # ------------------------------------------------------------
    # 1. VARIÁVEIS BÁSICAS DO CANDLE
    # ------------------------------------------------------------
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range_pct"] = df["range"] / df["close"].replace(0, np.nan)

    # ------------------------------------------------------------
    # 2. RETORNOS
    # ------------------------------------------------------------
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # ------------------------------------------------------------
    # 3. EMAs E DISTÂNCIAS
    # ------------------------------------------------------------
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema100"] = df["close"].ewm(span=100).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    df["dist_ema9"] = df["close"] - df["ema9"]
    df["dist_ema20"] = df["close"] - df["ema20"]
    df["dist_ema50"] = df["close"] - df["ema50"]
    df["dist_ema100"] = df["close"] - df["ema100"]
    df["dist_ema200"] = df["close"] - df["ema200"]

    # ------------------------------------------------------------
    # 4. SLOPES (REGRESSÃO) — SLOPE20 / SLOPE50 (MANTIDOS)
    # ------------------------------------------------------------
    df["slope20"] = slope_regression(df["close"], 20)
    df["slope50"] = slope_regression(df["close"], 50)

    # Duplicados removidos:
    # df["wave_slope_20"] = slope_regression(df["close"], 20)     # REMOVIDO — duplicado
    # df["trend_slope_50"] = slope_regression(df["close"], 50)    # REMOVIDO — duplicado

    # ------------------------------------------------------------
    # 5. AGRESSÃO (V92 BASE + ELITE MICROESTRUTURA)
    # ------------------------------------------------------------
    df["aggression_buy"] = df["taker_buy_base"]
    df["aggression_sell"] = df["volume"] - df["taker_buy_base"]
    df["aggression_delta"] = df["aggression_buy"] - df["aggression_sell"]
    df["aggression_ratio"] = df["aggression_buy"] / (df["aggression_sell"] + 1e-8)
    df["imbalance"] = df["aggression_delta"]

    # 🟢 ATIVAÇÃO DE MICROESTRUTURA (Se colunas existirem no CSV)
    if "cum_delta" in df.columns:
        # Z-Score do Delta (Detecta agressão anômala)
        df["delta_z"] = (df["cum_delta"] - df["cum_delta"].rolling(100).mean()) / (df["cum_delta"].rolling(100).std() + 1e-9)
        
        # Divergência de Delta (Preço sobe vs Delta cai)
        df["price_delta_div"] = (df["close"].pct_change(5) > 0) & (df["cum_delta"].rolling(5).mean() < 0)
        df["price_delta_div"] = df["price_delta_div"].astype(int)
        
        # Z-Score de VPIN (Toxicidade do fluxo)
        if "vpin" in df.columns:
            df["vpin_z"] = (df["vpin"] - df["vpin"].rolling(100).mean()) / (df["vpin"].rolling(100).std() + 1e-9)
        
        # Força da Absorção
        if "absorcao" in df.columns:
            df["abs_strength"] = df["absorcao"].rolling(10).mean()

    # ------------------------------------------------------------
    # 6. FIBONACCI
    # ------------------------------------------------------------
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["fib_0_382"] = df["low"] + 0.382 * df["range"]
    df["fib_0_5"] = df["low"] + 0.5 * df["range"]
    df["fib_0_618"] = df["low"] + 0.618 * df["range"]
    df["fib_ext_1_618"] = df["high"] + 1.618 * df["range"]
    
    return df

def adicionar_feature_space_density_zero_leakage_optimized(
    df: pd.DataFrame,
    n_neighbors: int = 20,
    window: int = 2000,
    min_history: int = 100
) -> pd.DataFrame:
    """
    Feature Space Density (KNN) — ZERO LEAKAGE
    • Usa apenas o passado
    • Janela móvel para performance
    • Normalização apenas no final
    """

    df = df.copy()

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    num_cols = [
        c for c in df.columns
        if df[c].dtype != "object"
        and c not in [
            "target_A",
            "target_B",
            "target_C",
            "target_A_bin",
            "close_future",
            "ret_fut",
            "amp_fut",
            "high_fut",
            "low_fut"
        ]
    ]

    X = df[num_cols].fillna(0.0).values
    density = np.zeros(len(df))

    for i in range(len(df)):
        if i < min_history:
            density[i] = 0.0
            continue

        # 🔹 Janela móvel apenas com passado
        start = max(0, i - window)
        X_past = X[start:i]

        if len(X_past) < min_history:
            density[i] = 0.0
            continue

        scaler = StandardScaler()
        X_past_scaled = scaler.fit_transform(X_past)
        x_now = scaler.transform(X[i:i+1])

        k = min(n_neighbors, len(X_past_scaled) - 1)
        if k < 2:
            density[i] = 0.0
            continue

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(X_past_scaled)

        dists, _ = knn.kneighbors(x_now)
        density[i] = 1.0 / (dists.mean() + 1e-9)

    # 🔹 Normalização global APÓS o cálculo completo
    density = (density - density.mean()) / (density.std() + 1e-9)
    df["feature_space_density"] = density

    return df


# =====================================================================
# FEATURES AVANÇADAS (LIMPO)
# =====================================================================

def adicionar_features_avancadas(df):
    df = df.copy()

    # ------------------------------------------------------------
    # 1. RETORNOS ADICIONAIS (ret1/ret5 duplicados REMOVIDOS)
    # ------------------------------------------------------------
    df["ret2"] = df["close"].pct_change(2)
    df["ret_ratio"] = df["ret1"] / (df["ret5"] + 1e-8)

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
# V25 — EXECUÇÃO COMPLETA (AUTO) — BLOCO OFICIAL
# ===============================================================

print("\n===============================================================")
print("CONFIGURAÇÃO DOS TARGETS (ADAPTATIVO V25)")
print("===============================================================\n")

# 1) Usuário informa somente o horizonte futuro
try:
    h_fut = int(input_render("Horizonte futuro N (ex: 3): ").strip())
    if h_fut <= 0:
        raise ValueError
except:
    print("Valor inválido → usando N = 3 (padrão institucional).")
    h_fut = 3

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

# 3) Auto thresholds
print(">>> Calculando thresholds ideais (A, B, C)...")
thrA, thrB, thrC = auto_thresholds_v25(df_all)

print(f"✔ Threshold Target A: {thrA:.6f}")
print(f"✔ Threshold Target B: {thrB:.6f}")
print(f"✔ Threshold Target C: {thrC:.6f}")

# 4) Criar targets
print(">>> Criando targets A, B, C (adaptativo V25)...")
df_all = criar_targets_v25(df_all, thrA, thrB, thrC)

print("✔ Targets criados com sucesso.\n")

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

usar_ctx = input_render("Adicionar contexto multi-TF? (s/n): ").strip().lower()

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

    escolha = input_render("Quais TFs deseja adicionar? (ex: 1h,4h,8h): ").strip().lower()
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
        "ts","open","high","low","close","volume","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","close_time","ignore",
        "mark_price","index_price","fundingRate",
        "session","tp",
        "close_future","ret_fut","amp_fut",
        "high_fut","low_fut",
        "impulse_count",
        "ret_max", "ret_min", "ret_max_temp", "ret_min_temp", # 🔴 PATCH ANTI-LEAKAGE
        "total_vol_agg", "buy_vol_agg", "sell_vol_agg" # Colunas auxiliares de micro (não são features diretas)
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

def backtest_operacional_adaptativo(
    df: pd.DataFrame,
    custo_total_pct: float,
    horizonte: int,
    conf_min: float,
    symbol: str,
    timeframe: str,
):
    """
    BACKTEST OPERACIONAL — RELATÓRIO ECONÔMICO (ADAPTATIVO / NÃO INTRUSIVO)
    """

    # --------------------------------------------------
    # Validações mínimas
    # --------------------------------------------------
    required_cols = {"open", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Backtest operacional abortado — colunas ausentes: {missing}")

    # --------------------------------------------------
    # Detectar colunas de direção e confiança
    # --------------------------------------------------
    direcao_cols = [c for c in df.columns if c.startswith("decisao_k")]
    conf_cols    = [c for c in df.columns if c.startswith("conf_k")]

    if not direcao_cols or not conf_cols:
        print("⚠ Backtest econômico não executado — colunas de direção/confiança ausentes.")
        print("   (Target_K permanece válido)")
        return

    direcao_col = direcao_cols[0]
    conf_col    = conf_cols[0]

    print(f"\n[BACKTEST OPERACIONAL]")
    print(f"Coluna direção usada  : {direcao_col}")
    print(f"Coluna confiança usada: {conf_col}")

    trades = []
    compras = vendas = 0
    barras_em_trade = 0

    # --------------------------------------------------
    # Loop principal
    # --------------------------------------------------
    for i in range(len(df) - horizonte - 1):
        direcao = df.iloc[i][direcao_col]
        conf    = df.iloc[i][conf_col]

        if not isinstance(conf, (int, float)) or np.isnan(conf):
            continue

        if conf < conf_min or direcao not in ("ALTA", "BAIXA"):
            continue

        entrada = df.iloc[i + 1]["open"]
        saida   = df.iloc[i + horizonte]["close"]

        if direcao == "ALTA":
            compras += 1
            pnl = (saida - entrada) / entrada
        else:
            vendas += 1
            pnl = (entrada - saida) / entrada

        pnl -= custo_total_pct
        trades.append(pnl)
        barras_em_trade += horizonte

    if not trades:
        print("⚠ Nenhum trade executado.")
        return

    trades = np.array(trades)

    # --------------------------------------------------
    # Prints operacionais
    # --------------------------------------------------
    print("\n--- CONTEXTO ---")
    print(f"Ativo / TF        : {symbol} | {timeframe}")
    print(f"Barras analisadas : {len(df)}")
    print(f"Horizonte         : {horizonte} candles")

    print("\n--- ATIVIDADE ---")
    print(f"Total trades      : {len(trades)}")
    print(f"Compras           : {compras}")
    print(f"Vendas            : {vendas}")

    print("\n--- RESULTADO ---")
    print(f"Win rate          : {(trades > 0).mean() * 100:.2f}%")
    print(f"Expectância/trade : {trades.mean() * 100:.4f}%")
    print(f"PnL acumulado     : {trades.sum() * 100:.2f}%")

    print("\n--- OCUPAÇÃO ---")
    print(f"% tempo em trade   : {100 * barras_em_trade / len(df):.2f}%")

    print("=" * 90 + "\n")


# ===============================================================
# PERGUNTA AO USUÁRIO — ATIVAR PESO TEMPORAL
# (variável correta, função já definida ACIMA — sem erros)
# ===============================================================
usar_peso = input_render("Aplicar peso temporal no treinamento? (s/n): ").strip().lower()
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

    modo = input_render("Escolha o modo de peso (1/2/3/4): ").strip()

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
        it = input_render("Escolha (1/2/3/4): ").strip()

        mapa_it = {"1": 1.0, "2": 1.5, "3": 2.0, "4": 3.0}
        mult_temporal = mapa_it.get(it, 1.0)

        # --------------------------
        # 2) Janela de prioridade N
        # --------------------------
        try:
            janela_N = int(input_render("\nQuantos dias recentes dar mais peso? (ex: 15, 30, 45): "))
            janela_N = max(1, janela_N)
        except:
            janela_N = 30

        # --------------------------
        # 3) Reforço por regime vencedor
        # --------------------------
        reforcar_reg_vencedor = input_render("\nReforçar regimes vencedores? (s/n): ").strip().lower()
        if reforcar_reg_vencedor == "s":
            inten = input_render("Intensidade (1=leve, 2=moderado, 3=forte): ").strip()
            mapa_int = {"1": 1.1, "2": 1.3, "3": 1.6}
            mult_reg_vencedor = mapa_int.get(inten, 1.1)
        else:
            mult_reg_vencedor = 1.0

        # --------------------------
        # 4) Penalizar regimes ruins
        # --------------------------
        penalizar_ruins = input_render("\nPenalizar regimes ruins? (s/n): ").strip().lower()
        if penalizar_ruins == "s":
            penal = input_render("Nível de penalização (0.8, 0.6, 0.4): ").strip()
            try:
                penal_ruim = float(penal)
            except:
                penal_ruim = 0.8
        else:
            penal_ruim = 1.0

        # --------------------------
        # 5) Preferência por ciclos
        # --------------------------
        foco_tendencia = input_render("\nDar mais peso a mercados tendenciais? (s/n): ").strip().lower()
        foco_lateral   = input_render("Dar mais peso a mercados laterais (squeeze)? (s/n): ").strip().lower()
        foco_vol_alta  = input_render("Dar mais peso à volatilidade alta? (s/n): ").strip().lower()
        foco_vol_baixa = input_render("Dar mais peso à volatilidade baixa? (s/n): ").strip().lower()

        # --------------------------
        # 6) Peso por volatilidade
        # --------------------------
        vol_direto = input_render("\nPeso proporcional à volatilidade? (s/n): ").strip().lower()
        vol_inverso = input_render("Peso inverso à volatilidade? (s/n): ").strip().lower()

        # --------------------------
        # 7) Tendência forte
        # --------------------------
        reforcar_tendencia = input_render("\nReforçar tendência forte? (s/n): ").strip().lower()
        if reforcar_tendencia == "s":
            inten2 = input_render("Intensidade (1=leve, 2=moderado, 3=forte): ").strip()
            mapa_int2 = {"1": 1.1, "2": 1.3, "3": 1.6}
            mult_tendencia = mapa_int2.get(inten2, 1.2)
        else:
            mult_tendencia = 1.0

        # --------------------------
        # 8) Mudança de regime
        # --------------------------
        peso_mudanca = input_render("\nDar peso especial após mudança de regime? (s/n): ").strip().lower()
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
    meia_vida = input_render("Informe a meia-vida (em dias) [enter=90]: ").strip()
    meia_vida = float(meia_vida) if meia_vida else 90.0

    # Peso temporal
    idade_dias = (dt_max - ts).dt.days.astype(float)
    peso_tempo = np.power(0.5, idade_dias / meia_vida)

    # Peso por regime
    if "trend_regime" in df.columns:
        print("\n>>> Detectado trend_regime. Configurando pesos…")
        p_bear  = input_render("Peso regime BAIXA  (-1) [enter=1.0]: ").strip()
        p_lateral = input_render("Peso regime LATERAL (0) [enter=1.0]: ").strip()
        p_bull = input_render("Peso regime ALTA   (1) [enter=1.0]: ").strip()

        p_bear  = float(p_bear) if p_bear else 1.0
        p_lateral = float(p_lateral) if p_lateral else 1.0
        p_bull = float(p_bull) if p_bull else 1.0

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

    modo = input_render("Escolha o modo do C6 (1/2/3): ").strip()

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
from catboost import CatBoostClassifier

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
# SPLIT TEMPORAL 70 / 10 / 20
# -------------------------------------------------------------------------
i_train = int(n * 0.70)
i_val   = int(n * 0.80)   # reserva 10% (val)
# teste = 20% final

X_train = X[:i_train]
X_val   = X[i_train:i_val]
X_test  = X[i_val:]

close_train = close[:i_train]
close_val   = close[i_train:i_val]
close_test  = close[i_val:]

# -------------------------------------------------------------------------
# FABRICA DE MODELOS (sempre instância NOVA por fit)
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
    if nome_modelo == "CAT":
        return CatBoostClassifier(
            iterations=300, learning_rate=0.03,
            depth=6, loss_function="Logloss", verbose=False
        )
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")

NOMES_MODELOS = ["LGBM", "XGB", "CAT"]

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
print("MÓDULO 6 — TREINANDO TODOS OS TARGETS (A, B, C)")
print("===============================================================\n")

targets_disponiveis = []

for t in ["target_A_bin", "target_A", "target_B", "target_C"]:
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

HORIZONTE_BACKTEST = int(globals().get("HORIZONTE_BACKTEST", 5))
ks_treinados = []

for k in range(1, HORIZONTE_BACKTEST + 1):
    alvo_k = f"target_K{k}"
    print(f">>> Treinando {alvo_k}")

    df_k = preparar_futuros(df_all, k)

    if "ret_fut" not in df_k.columns:
        raise RuntimeError(f"[TARGET_K{k}] ret_fut não encontrado")

    # target binário: concorda (1) / discorda (0)
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


# ===========================================================
# ORQUESTRAÇÃO OBRIGATÓRIA — BACKTEST DECISOR
# ===========================================================
if "modelos_cache" not in globals() or not modelos_cache:
    print("\n[INFO] BACKTEST_DECISOR DESATIVADO")
    print("       Motivo: modelos não treinados/carregados.")
    ATIVAR_BACKTEST_DECISOR = False

if "X_arrays_cache" not in globals() or not X_arrays_cache:
    print("\n[INFO] BACKTEST_DECISOR DESATIVADO")
    print("       Motivo: matrizes X não disponíveis.")
    ATIVAR_BACKTEST_DECISOR = False


# ===============================================================
# BACKTEST FINAL — MODELO COMO DECISOR (ISOLADO, SEM CANAL)
# ===============================================================

THRESHOLD_MINIMO = 0.55   # << CORREÇÃO DO ERRO LÓGICO
MARGEM_DIRECAO   = 0.05   # margem mínima entre probabilidades

# ---------------------------------------------------------------
# SANIDADE OBRIGATÓRIA — NÃO INVENTA NADA
# ---------------------------------------------------------------
vars_necessarias = [
    "df_all",
    "modelos_cache",
    "probs_cache",
    "lista_targets",
]

for v in vars_necessarias:
    if v not in globals():
        raise RuntimeError(f"[BACKTEST_DECISOR] Variável obrigatória ausente: {v}")

# ---------------------------------------------------------------
# FUNÇÃO ÚNICA E OFICIAL PARA LER PROBABILIDADES
# ---------------------------------------------------------------
def _get_p_up_down_from_classes(tgt, probs_row):
    """
    Extrai probabilidades de ALTA e BAIXA de forma robusta, 
    independente da ordem das classes no modelo.
    """
    try:
        # Mapeamento real das classes do modelo
        classes = list(modelos_cache[tgt].classes_)
        # Converter para int para evitar problemas de tipo (0.0 vs 0)
        classes_int = [int(c) for c in classes]
        
        p_up = 0.0
        p_down = 0.0

        # 1. Tentar localizar classe de COMPRA (1)
        if 1 in classes_int:
            idx_up = classes_int.index(1)
            p_up = float(probs_row[idx_up])

        # 2. Tentar localizar classe de VENDA (-1 ou 0)
        # Prioridade para -1 (multiclasse), depois 0 (binário)
        if -1 in classes_int:
            idx_down = classes_int.index(-1)
            p_down = float(probs_row[idx_down])
        elif 0 in classes_int:
            # Em modelos binários (0 e 1), 0 é SEMPRE venda
            idx_down = classes_int.index(0)
            p_down = float(probs_row[idx_down])
            
        return p_up, p_down
    except Exception as e:
        return None, None

# ---------------------------------------------------------------
# ESTRUTURA DE ESTATÍSTICAS
# ---------------------------------------------------------------
stats_decisao = {}

for tgt in lista_targets:
    stats_decisao[tgt] = {
        "total": 0,
        "buy": 0,
        "sell": 0,
        "nothing": 0,
        "win_buy": 0,
        "win_sell": 0,
        "conf_buy": [],
        "conf_sell": [],
    }

# ---------------------------------------------------------------
# LOOP PRINCIPAL — DECISÃO POR CANDLE
# ---------------------------------------------------------------
# 🛡️ PATCH: Garantir que o loop respeite o menor tamanho entre df_all e o cache de probabilidades
n_df = len(df_all)
n_probs = min([len(probs_cache[t]) for t in lista_targets]) if lista_targets else 0
n = min(n_df, n_probs)

for i in range(n - 1):

    preco_entrada = df_all["close"].iloc[i]
    preco_saida   = df_all["close"].iloc[i + 1]
    
    # 🛡️ DEBUG DE POLARIDADE (Apenas nos primeiros 5 candles do teste)
    if i < 5:
        print(f"\n[DEBUG i={i}] Preço: {preco_entrada:.4f} -> {preco_saida:.4f} ({((preco_saida/preco_entrada)-1)*100:+.2f}%)")

    for tgt in lista_targets:
        direcao = 0 # 🛡️ RESET CRÍTICO: Garante que um sinal não vaze para o próximo target

        st = stats_decisao[tgt]
        st["total"] += 1

        # 🛡️ Verificação extra de segurança para o índice
        if i >= len(probs_cache[tgt]):
            continue

        probs_row = probs_cache[tgt][i]

        # =========================================================
        # DECISÃO DE DIREÇÃO (TARGETS K E MULTICLASSE A/B/C)
        # =========================================================
        p_up, p_down = _get_p_up_down_from_classes(tgt, probs_row)
        
        if i < 5:
            print(f"  Target: {tgt:12} | P_UP: {p_up:.2f} | P_DOWN: {p_down:.2f}")

        if p_up is None:
            st["nothing"] += 1
            continue

        # -------- LÓGICA DE DECISÃO ADAPTATIVA --------
        # Para targets K (binários), usamos o THRESHOLD_MINIMO (ex: 0.50)
        # Para targets A/B/C (multiclasse), o threshold precisa ser menor (ex: 0.34) pois há 3 classes
        thresh_atual = THRESHOLD_MINIMO
        if not tgt.startswith("target_K") and not tgt.endswith("_bin"):
            thresh_atual = THRESHOLD_MINIMO * 0.65 # Reduz threshold para multiclasse (ex: 0.50 -> 0.325)

        if (p_up >= thresh_atual) and (p_up > p_down + MARGEM_DIRECAO):
            direcao = 1
        elif (p_down >= thresh_atual) and (p_down > p_up + MARGEM_DIRECAO):
            direcao = -1
        else:
            st["nothing"] += 1
            continue

        # ---------------------------------------------------------
        # RESULTADO REAL (1 candle à frente — SEM INVENÇÃO)
        # ---------------------------------------------------------
        if direcao == 1:
            st["buy"] += 1
            st["conf_buy"].append(p_up)
            if preco_saida > preco_entrada:
                st["win_buy"] += 1
        else:
            st["sell"] += 1
            st["conf_sell"].append(p_down)
            if preco_saida < preco_entrada:
                st["win_sell"] += 1

# ---------------------------------------------------------------
# RELATÓRIO FINAL — DECISÃO DO MODELO
# ---------------------------------------------------------------
print("\n================ RESULTADO — DECISÃO DO MODELO =================\n")

for tgt, s in stats_decisao.items():
    total = s["total"]

    wr_buy  = (s["win_buy"]  / s["buy"]  * 100) if s["buy"]  > 0 else 0.0
    wr_sell = (s["win_sell"] / s["sell"] * 100) if s["sell"] > 0 else 0.0
    indec   = (s["nothing"] / total * 100) if total > 0 else 0.0
    
    avg_conf_buy = (sum(s["conf_buy"]) / len(s["conf_buy"]) * 100) if s["conf_buy"] else 0.0
    avg_conf_sell = (sum(s["conf_sell"]) / len(s["conf_sell"]) * 100) if s["conf_sell"] else 0.0

    print(f"TARGET: {tgt}")
    print(f"Total candles avaliados : {total}")
    print(f"BUY      : {s['buy']:4d} | Winrate: {wr_buy:6.2f}% | Confiança Média: {avg_conf_buy:6.2f}%")
    print(f"SELL     : {s['sell']:4d} | Winrate: {wr_sell:6.2f}% | Confiança Média: {avg_conf_sell:6.2f}%")
    print(f"NOTHING  : {s['nothing']:4d} | Indecisão (%)  : {indec:.2f}")
    
    # 🛡️ TESTE DE SANIDADE DE POLARIDADE
    if wr_buy < 40 and s['buy'] > 100:
        print("⚠️ ALERTA: Winrate de COMPRA muito baixo. Possível inversão de polaridade detectada.")
    if wr_sell < 40 and s['sell'] > 100:
        print("⚠️ ALERTA: Winrate de VENDA muito baixo. Possível inversão de polaridade detectada.")
        
    print("-" * 70)
print("\n>>> FIM — BACKTEST MODELO COMO DECISOR (ISOLADO)\n")
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

def modulo_8_simulador_elite(df_backtest, lista_targets, modelos_cache, probs_cache):
    print("\n" + "="*60)
    print("MÓDULO 8 — MOTOR DE EXECUÇÃO INTELIGENTE (ELITE)")
    print("="*60)
    
    rodar = input_render("\nDeseja rodar a simulação financeira realista? (s/n): ").strip().lower()
    if rodar != 's':
        print(">>> Simulação ignorada.")
        return

    try:
        capital_inicial = float(input_render("Capital Inicial (USD) [ex: 1000]: ") or 1000)
        valor_mao = float(input_render("Valor de cada entrada (USD) [ex: 100]: ") or 100)
        alavancagem = float(input_render("Alavancagem (x) [ex: 5]: ") or 5)
        min_conf = float(input_render("Confiança Mínima (0.50 a 0.99) [ex: 0.75]: ") or 0.75)
        taxa_financiamento = float(input_render("Taxa de Financiamento Diária (%) [ex: 0.03]: ") or 0.03) / 100
    except:
        print("Valores inválidos. Usando padrões institucionais.")
        capital_inicial, valor_mao, alavancagem, min_conf, taxa_financiamento = 1000, 100, 5, 0.75, 0.0003

    # Custos Fixos
    comissao = 0.001  # 0.1%
    slippage = 0.0005 # 0.05%
    custo_total_operacao = (comissao + slippage) * 2

    print(f"\n>>> Iniciando Simulação com Mão de ${valor_mao}, {alavancagem}x e Confiança > {min_conf*100:.0f}%")
    
    for tgt in lista_targets:
        if tgt not in modelos_cache: continue
        
        print(f"\n--- Simulando Target: {tgt} ---")
        capital = capital_inicial
        # Estatísticas Detalhadas
        stats = {
            'sinais_total': 0,
            'trades_total': 0,
            'trades_buy': 0,
            'trades_sell': 0,
            'wins_buy': 0,
            'wins_sell': 0,
            'loss_buy': 0,
            'loss_sell': 0,
            'saida_tp': 0,
            'saida_sl': 0,
            'saida_fluxo': 0,
            'saida_tempo': 0,
            'lucro_bruto': 0,
            'prejuizo_bruto': 0,
            'hold_time_total': 0,
            'fluxo_win': 0
        }
        
        capital = capital_inicial
        drawdown_max = 0
        pico_capital = capital_inicial
        posicionado = False
        index_saida = 0
        
        # Sincronizar dados
        n = min(len(df_backtest), len(probs_cache[tgt]))
        df_sim = df_backtest.iloc[:n].copy()
        probs = probs_cache[tgt][:n]
        classes = list(modelos_cache[tgt].classes_)
        
        # Garantir ATR para SL/TP
        if 'atr14' not in df_sim.columns:
            df_sim['atr14'] = (df_sim['high'] - df_sim['low']).rolling(14).mean().bfill()

        for i in range(n - 10):
            # 🛡️ Trava de Ocupação: Só entra se não estiver posicionado
            if posicionado:
                if i >= index_saida:
                    posicionado = False
                else:
                    continue

            # 1. Extrair Probabilidades
            p_up = probs[i][classes.index(1)] if 1 in classes else 0
            p_down = probs[i][classes.index(-1)] if -1 in classes else (probs[i][classes.index(0)] if 0 in classes and len(classes)==2 else 0)
            
            direcao = 0
            if p_up >= min_conf: direcao = 1
            elif p_down >= min_conf: direcao = -1
            
            if direcao != 0:
                stats['sinais_total'] += 1
                
                # 2. Configurar Trade Inteligente
                preco_entrada = df_sim['close'].iloc[i]
                atr = df_sim['atr14'].iloc[i]
                sl = preco_entrada - (direcao * atr * 1.5)
                tp = preco_entrada + (direcao * atr * 3.0)
                
                # 3. Simular Navegação Intra-Trade (até 10 candles)
                pnl_trade = 0
                motivo_saida = "TEMPO"
                for j in range(i + 1, min(i + 11, n)):
                    low_j = df_sim['low'].iloc[j]
                    high_j = df_sim['high'].iloc[j]
                    close_j = df_sim['close'].iloc[j]
                    
                    if 'cum_delta' in df_sim.columns:
                        delta_j = df_sim['cum_delta'].iloc[j]
                        if (direcao == 1 and delta_j < 0) or (direcao == -1 and delta_j > 0):
                            pnl_trade = ((close_j / preco_entrada) - 1) * direcao
                            motivo_saida = "FLUXO"
                            index_saida = j
                            break

                    if (direcao == 1 and low_j <= sl) or (direcao == -1 and high_j >= sl):
                        pnl_trade = ((sl / preco_entrada) - 1) * direcao
                        motivo_saida = "SL"
                        index_saida = j
                        break
                    
                    if (direcao == 1 and high_j >= tp) or (direcao == -1 and low_j <= tp):
                        pnl_trade = ((tp / preco_entrada) - 1) * direcao
                        motivo_saida = "TP"
                        index_saida = j
                        break
                    
                    if j == min(i + 10, n - 1):
                        pnl_trade = ((close_j / preco_entrada) - 1) * direcao
                        motivo_saida = "TEMPO"
                        index_saida = j
                        break

                # 4. Calcular Resultado Financeiro (Lote Fixo)
                pnl_liquido = (pnl_trade * alavancagem) - custo_total_operacao - (taxa_financiamento / 96)
                lucro_usd = valor_mao * pnl_liquido
                
                # Atualizar Estatísticas
                stats['trades_total'] += 1
                if direcao == 1:
                    stats['trades_buy'] += 1
                    if lucro_usd > 0: stats['wins_buy'] += 1
                    else: stats['loss_buy'] += 1
                else:
                    stats['trades_sell'] += 1
                    if lucro_usd > 0: stats['wins_sell'] += 1
                    else: stats['loss_sell'] += 1
                
                if motivo_saida == "TP": stats['saida_tp'] += 1
                elif motivo_saida == "SL": stats['saida_sl'] += 1
                elif motivo_saida == "FLUXO": 
                    stats['saida_fluxo'] += 1
                    if lucro_usd > 0: stats['fluxo_win'] += 1
                else: stats['saida_tempo'] += 1
                
                stats['hold_time_total'] += (index_saida - i)
                
                if lucro_usd > 0: stats['lucro_bruto'] += lucro_usd
                else: stats['prejuizo_bruto'] += abs(lucro_usd)

                capital += lucro_usd
                posicionado = True
                
                if capital > pico_capital: pico_capital = capital
                dd = (pico_capital - capital) / pico_capital
                if dd > drawdown_max: drawdown_max = dd
                
                if capital <= 0:
                    capital = 0
                    break

        retorno_total = ((capital / capital_inicial) - 1) * 100
        winrate_total = ( (stats['wins_buy'] + stats['wins_sell']) / stats['trades_total'] * 100) if stats['trades_total'] > 0 else 0
        profit_factor = (stats['lucro_bruto'] / stats['prejuizo_bruto']) if stats['prejuizo_bruto'] > 0 else float('inf')
        
        avg_win = (stats['lucro_bruto'] / (stats['wins_buy'] + stats['wins_sell'])) if (stats['wins_buy'] + stats['wins_sell']) > 0 else 0
        avg_loss = (stats['prejuizo_bruto'] / (stats['loss_buy'] + stats['loss_sell'])) if (stats['loss_buy'] + stats['loss_sell']) > 0 else 0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else 0
        avg_hold = (stats['hold_time_total'] / stats['trades_total']) if stats['trades_total'] > 0 else 0
        eficiencia_fluxo = (stats['fluxo_win'] / stats['saida_fluxo'] * 100) if stats['saida_fluxo'] > 0 else 0
        aproveitamento = (stats['trades_total'] / stats['sinais_total'] * 100) if stats['sinais_total'] > 0 else 0

        print(f"  CAPITAL FINAL: ${capital:,.2f} | RETORNO: {retorno_total:+.2f}%")
        print(f"  DRAWDOWN MÁXIMO: {drawdown_max*100:.2f}% | PROFIT FACTOR: {profit_factor:.2f}")
        print(f"  PAYOFF: {payoff:.2f} | LUCRO MÉDIO: ${avg_win:.2f} | PREJUÍZO MÉDIO: ${avg_loss:.2f}")
        print(f"  HOLD TIME MÉDIO: {avg_hold:.1f} candles | APROVEITAMENTO SINAIS: {aproveitamento:.1f}%")
        print(f"  FUNIL: Sinais: {stats['sinais_total']} | Trades Efetivados: {stats['trades_total']}")
        print(f"  COMPRAS: {stats['trades_buy']} (Wins: {stats['wins_buy']} | Loss: {stats['loss_buy']})")
        print(f"  VENDAS : {stats['trades_sell']} (Wins: {stats['wins_sell']} | Loss: {stats['loss_sell']})")
        print(f"  SAÍDAS : TP: {stats['saida_tp']} | SL: {stats['saida_sl']} | Fluxo: {stats['saida_fluxo']} (Eficiência: {eficiencia_fluxo:.1f}%) | Tempo: {stats['saida_tempo']}")
        print("-" * 60)

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
# Atualizar Módulo 8 para usar o Juiz e retornar resultados
def modulo_8_simulador_elite_v2(df_backtest, lista_targets, modelos_cache, probs_cache, meta_modelos=None, 
                                capital_inicial=1000, valor_mao=100, alavancagem=5, min_conf=0.75, usar_meta=True,
                                corte_juiz=0.85, titulo_relatorio="RELATÓRIO DE PERFORMANCE INSTITUCIONAL"):
    
    comissao, slippage = 0.001, 0.0005
    custo_total_operacao = (comissao + slippage) * 2
    taxa_financiamento = 0.0003 / 96   
    resultados_por_target = {}
    for tgt in lista_targets:
        if tgt not in modelos_cache: continue        
        # 🚀 VERIFICAÇÃO DE ATIVAÇÃO DO JUIZ (CORREÇÃO DE NOMES)
        juiz_key = next((k for k in meta_modelos.keys() if k == tgt or k.endswith(tgt)), None) if meta_modelos else None
        juiz_ativo = usar_meta and juiz_key is not None   
        capital = capital_inicial
        stats = {'sinais_total': 0, 'trades_total': 0, 'trades_buy': 0, 'trades_sell': 0, 'wins_buy': 0, 'wins_sell': 0, 'loss_buy': 0, 'loss_sell': 0, 'saida_tp': 0, 'saida_sl': 0, 'saida_fluxo': 0, 'saida_tempo': 0, 'lucro_bruto': 0, 'prejuizo_bruto': 0, 'hold_time_total': 0, 'fluxo_win': 0, 'vetados_juiz': 0}      
        pico_capital = capital_inicial
        drawdown_max = 0
        posicionado = False
        index_saida = 0
        
        n = min(len(df_backtest), len(probs_cache[tgt]))
        df_sim = df_backtest.iloc[:n].copy()
        probs = probs_cache[tgt][:n]
        classes = list(modelos_cache[tgt].classes_)
        
        if 'atr14' not in df_sim.columns:
            df_sim['atr14'] = (df_sim['high'] - df_sim['low']).rolling(14).mean().bfill()

        for i in range(n - 10):
            if posicionado:
                if i >= index_saida: posicionado = False
                else: continue

            p_up = probs[i][classes.index(1)] if 1 in classes else 0
            p_down = probs[i][classes.index(-1)] if -1 in classes else (probs[i][classes.index(0)] if 0 in classes and len(classes)==2 else 0)
            
            direcao = 0
            if p_up >= min_conf: direcao = 1
            elif p_down >= min_conf: direcao = -1
            
            if direcao != 0:
                stats['sinais_total'] += 1
                
                # 🛡️ FILTRO META-LABELING (O JUIZ)
                if juiz_ativo:
                    try:
                        # 🚀 SINCRONIZAÇÃO TOTAL: Usa as mesmas colunas do treino
                        conf_val = max(p_up, p_down)
                        cols_juiz = meta_modelos_features[tgt]
                        
                        # Prepara a linha de dados
                        df_row = df_sim.iloc[[i]].copy()
                        df_row['meta_conf_primaria'] = conf_val
                        
                        # Garante que todas as colunas existam (preenche com 0 se faltar)
                        for c in cols_juiz:
                            if c not in df_row.columns: 
                                df_row[c] = 0
                            
                        X_meta_input = df_row[cols_juiz].values.reshape(1, -1)
                        
                        # Predição de probabilidade de acerto
                        prob_acerto = meta_modelos[juiz_key].predict_proba(X_meta_input)[0][1]
                        
                        # DEBUG TEMPORÁRIO (Apenas para os primeiros 5 sinais de cada target)
                        if stats['sinais_total'] <= 5:
                            print(f"    [DEBUG JUIZ] {tgt} | Prob. Acerto: {prob_acerto:.4f} | Corte: {corte_juiz:.2f}")
                        
                        # 🛡️ FILTRAGEM REAL
                        if prob_acerto < corte_juiz: 
                            stats['vetados_juiz'] += 1
                            continue 
                    except Exception as e:
                        print(f"    [ERRO CRÍTICO JUIZ] {tgt}: {str(e)}")
                        # Se houver erro, vamos vetar por segurança
                        stats['vetados_juiz'] += 1
                        continue

                # Execução do Trade (Igual ao anterior)
                preco_entrada = df_sim['close'].iloc[i]
                atr = df_sim['atr14'].iloc[i]
                sl = preco_entrada - (direcao * atr * 1.5)
                tp = preco_entrada + (direcao * atr * 3.0)
                
                pnl_trade = 0
                motivo_saida = "TEMPO"
                for j in range(i + 1, min(i + 11, n)):
                    low_j, high_j, close_j = df_sim['low'].iloc[j], df_sim['high'].iloc[j], df_sim['close'].iloc[j]
                    
                    # 🛡️ SAÍDA POR FLUXO (APENAS SE EXISTIR DADOS DE AGRESSÃO)
                    if 'cum_delta' in df_sim.columns and df_sim['cum_delta'].iloc[j] != 0:
                        delta_j = df_sim['cum_delta'].iloc[j]
                        if (direcao == 1 and delta_j < 0) or (direcao == -1 and delta_j > 0):
                            pnl_trade = ((close_j / preco_entrada) - 1) * direcao
                            motivo_saida = "FLUXO"; index_saida = j; break
                    if (direcao == 1 and low_j <= sl) or (direcao == -1 and high_j >= sl):
                        pnl_trade = ((sl / preco_entrada) - 1) * direcao
                        motivo_saida = "SL"; index_saida = j; break
                    if (direcao == 1 and high_j >= tp) or (direcao == -1 and low_j <= tp):
                        pnl_trade = ((tp / preco_entrada) - 1) * direcao
                        motivo_saida = "TP"; index_saida = j; break
                    if j == min(i + 10, n - 1):
                        pnl_trade = ((close_j / preco_entrada) - 1) * direcao
                        motivo_saida = "TEMPO"; index_saida = j; break

                pnl_liquido = (pnl_trade * alavancagem) - custo_total_operacao - taxa_financiamento
                
                # 🚀 CORREÇÃO: O lucro deve ser baseado no capital atual se quisermos juros compostos,
                # ou fixo se usarmos valor_mao. Vamos usar valor_mao para ser conservador como solicitado.
                lucro_usd = valor_mao * pnl_liquido
                
                stats['trades_total'] += 1
                if direcao == 1:
                    stats['trades_buy'] += 1
                    if lucro_usd > 0: stats['wins_buy'] += 1
                    else: stats['loss_buy'] += 1
                else:
                    stats['trades_sell'] += 1
                    if lucro_usd > 0: stats['wins_sell'] += 1
                    else: stats['loss_sell'] += 1
                
                if motivo_saida == "TP": stats['saida_tp'] += 1
                elif motivo_saida == "SL": stats['saida_sl'] += 1
                elif motivo_saida == "FLUXO": 
                    stats['saida_fluxo'] += 1
                    if lucro_usd > 0: stats['fluxo_win'] += 1
                else: stats['saida_tempo'] += 1
                
                stats['hold_time_total'] += (index_saida - i)
                if lucro_usd > 0: stats['lucro_bruto'] += lucro_usd
                else: stats['prejuizo_bruto'] += abs(lucro_usd)

                # 🚀 ATUALIZAÇÃO DE CAPITAL REAL
                capital = float(capital + lucro_usd)
                posicionado = True
                
                if capital > pico_capital: 
                    pico_capital = capital
                
                dd = (pico_capital - capital) / max(1, pico_capital)
                if dd > drawdown_max: 
                    drawdown_max = dd
                
                if capital <= 0: 
                    capital = 0
                    # print(f"  [ALERTA] Capital zerado no target {tgt}!") # Comentado para não poluir o log do shuffle test
                    break

        retorno_total = ((capital / capital_inicial) - 1) * 100
        
        # 🚀 CÁLCULO DE PROFIT FACTOR REAL (SOMA GANHOS / SOMA PERDAS)
        ganhos_reais = float(stats['lucro_bruto'])
        perdas_reais = abs(float(stats['prejuizo_bruto']))
        
        if perdas_reais > 0:
            profit_factor = ganhos_reais / perdas_reais
        else:
            # 🚀 CORREÇÃO INSTITUCIONAL: Se perdas = 0, PF é teoricamente infinito. Usamos um valor alto.
            profit_factor = 999.99 if ganhos_reais > 0 else 0
        
        avg_hold = (stats['hold_time_total'] / stats['trades_total']) if stats['trades_total'] > 0 else 0
        payoff = (stats['lucro_bruto'] / max(1, stats['wins_buy'] + stats['wins_sell'])) / (stats['prejuizo_bruto'] / max(1, stats['loss_buy'] + stats['loss_sell'])) if stats['prejuizo_bruto'] > 0 else 0
        eficiencia_fluxo = (stats['fluxo_win'] / stats['saida_fluxo'] * 100) if stats['saida_fluxo'] > 0 else 0

        # Montar o dicionário de resultados
        resultados = {
            'target': tgt,
            'capital_inicial': capital_inicial,
            'capital_final': capital,
            'lucro_prejuizo': capital - capital_inicial,
            'retorno_total': retorno_total,
            'profit_factor': profit_factor,
            'drawdown_max': drawdown_max * 100,
            'payoff_ratio': payoff,
            'trades_executados': stats['trades_total'],
            'sinais_vetados': stats['vetados_juiz'],
            'titulo_relatorio': titulo_relatorio
        }
        
        resultados_por_target[tgt] = resultados
        
        # Imprimir o relatório apenas se não for o shuffle test
        if "SHUFFLE TEST" not in titulo_relatorio:
            print(f"\n" + "="*60)
            print(f"   {titulo_relatorio}: {tgt}")
            print("="*60)
            print(f"  [FINANCEIRO]")
            print(f"  💰 Capital Inicial  : ${capital_inicial:,.2f}")
            print(f"  💰 Lucro/Prejuízo   : ${resultados['lucro_prejuizo']:,.2f}")
            print(f"  💰 Capital Final    : ${capital:,.2f}")
            print(f"  📈 Retorno Total    : {retorno_total:+.2f}%")
            print(f"  📊 Profit Factor    : {profit_factor:.2f}")
            print("-" * 60)
            print(f"  [RISCO]")
            print(f"  📉 Drawdown Máximo  : {resultados['drawdown_max']:.2f}%")
            print(f"  ⚖️  Payoff Ratio     : {payoff:.2f}")
            print(f"  ⏱️  Hold Time Médio : {avg_hold:.1f} candles")
            print("-" * 60)
            print(f"  [FUNIL DE EXECUÇÃO]")
            print(f"  🎯 Sinais Gerados   : {stats['sinais_total']}")
            print(f"  🛡️  Vetados pelo Juiz: {stats['vetados_juiz']}")
            print(f"  ⚡ Trades Executados: {stats['trades_total']}")
            print(f"  📈 Utilização Sinal : {(stats['trades_total']/max(1, stats['sinais_total'])*100):.1f}%")
            print("-" * 60)
            print(f"  [DETALHAMENTO]")
            print(f"  🟢 COMPRAS: {stats['trades_buy']} (Wins: {stats['wins_buy']} | Loss: {stats['loss_buy']})")
            print(f"  🔴 VENDAS : {stats['trades_sell']} (Wins: {stats['wins_sell']} | Loss: {stats['loss_sell']})")
            print(f"  🚪 SAÍDAS : TP: {stats['saida_tp']} | SL: {stats['saida_sl']} | Fluxo: {stats['saida_fluxo']} | Tempo: {stats['saida_tempo']}")
            print(f"  🌊 Eficiência Fluxo: {eficiencia_fluxo:.1f}%")
            print("="*60 + "\n")
            
    return resultados_por_target

# 🛡️ MÓDULO 10 - TESTE DE ROBUSTEZ E VALIDAÇÃO FINAL
def modulo_10_teste_robustez(df_all, lista_targets, modelos_cache, probs_cache, meta_modelos_cache):
    print("\n" + "="*60)
    print("MÓDULO 10 — VALIDAÇÃO DE ROBUSTEZ (SHUFFLE TEST)")
    print("="*60)
    
    rodar = input_render("\nDeseja rodar a simulação financeira realista e o Shuffle Test? (s/n): ").strip().lower()
    if rodar != 's': return
    
    # 1. Coletar parâmetros do usuário
    try:
        capital_inicial = float(input_render("Capital Inicial (USD) [ex: 1000]: ") or 1000)
        valor_mao = float(input_render("Valor de cada entrada (USD) [ex: 100]: ") or 100)
        alavancagem = float(input_render("Alavancagem (x) [ex: 5]: ") or 5)
        min_conf = float(input_render("Confiança Mínima (0.50 a 0.99) [ex: 0.75]: ") or 0.75)
        
        # 🚀 INPUT INTELIGENTE: Aceita 's', 'n' ou o valor do rigor diretamente
        meta_input = input_render("Usar Juiz? (s/n ou digite o rigor ex: 0.85): ").strip().lower()
        if meta_input in ['s', 'sim']:
            usar_meta = True
            corte_juiz = float(input_render("Rigor do Juiz (0.50 a 0.99) [ex: 0.85]: ") or 0.85)
        elif meta_input in ['n', 'nao', 'não']:
            usar_meta = False
            corte_juiz = 0.85
        else:
            # Se digitou um número direto
            try:
                corte_juiz = float(meta_input)
                usar_meta = True
            except:
                usar_meta, corte_juiz = True, 0.85
    except:
        capital_inicial, valor_mao, alavancagem, min_conf, usar_meta, corte_juiz = 1000, 100, 5, 0.75, True, 0.85
        
    # 2. Executar Backtest Real
    print("\n" + "="*60)
    print(">>> 1/2: EXECUTANDO BACKTEST REAL (MODELO TREINADO) <<<")
    print("="*60)
    resultados_reais = modulo_8_simulador_elite_v2(
        df_all, lista_targets, modelos_cache, probs_cache, meta_modelos_cache,
        capital_inicial, valor_mao, alavancagem, min_conf, usar_meta,
        corte_juiz, titulo_relatorio="RELATÓRIO DE PERFORMANCE INSTITUCIONAL"
    )
    
    # 3. Executar Shuffle Test
    print("\n" + "="*60)
    print(">>> 2/2: EXECUTANDO SHUFFLE TEST (MODELO ALEATÓRIO) <<<")
    print("="*60)
    probs_cache_shuffled = realizar_teste_robustez(probs_cache)
    
    resultados_shuffled = modulo_8_simulador_elite_v2(
        df_all, lista_targets, modelos_cache, probs_cache_shuffled, meta_modelos_cache,
        capital_inicial, valor_mao, alavancagem, min_conf, usar_meta,
        corte_juiz, titulo_relatorio="RELATÓRIO DE PERFORMANCE (SHUFFLE TEST)"
    )
    
    # 4. Comparação e Relatório Final
    print("\n" + "█"*60)
    print("█ MÓDULO 10 — RELATÓRIO DE ROBUSTEZ E VALIDAÇÃO FINAL")
    print("█"*60)
    
    # 🚀 COMPARAR TODOS OS TARGETS (INCLUINDO K)
    targets_para_comparar = list(resultados_reais.keys())
    
    for tgt in targets_para_comparar:
        if tgt not in resultados_reais: continue
        
        real = resultados_reais[tgt]
        shuf = resultados_shuffled.get(tgt, {})
        
        lucro_real = real['lucro_prejuizo']
        lucro_shuf = shuf.get('lucro_prejuizo', 0)
        trades_real = real['trades_executados']
        trades_shuf = shuf.get('trades_executados', 0)
        
        # Cálculo do Score de Robustez
        # Score = (Lucro Real - Lucro Shuffle) / Lucro Real * 100
        # Se o lucro real for negativo, a robustez é calculada de forma diferente
        if lucro_real > 0:
            robustez_score = max(0, (lucro_real - lucro_shuf) / lucro_real * 100)
        else:
            # Se o modelo real perde, e o shuffle perde menos, o score é baixo.
            # Se o modelo real perde, e o shuffle ganha, o score é muito baixo.
            # Usamos a diferença absoluta para penalizar modelos perdedores.
            robustez_score = 100 - (abs(lucro_real - lucro_shuf) / capital_inicial * 100)
            robustez_score = max(0, robustez_score)
            
        print(f"\n--- VALIDAÇÃO DE ROBUSTEZ PARA TARGET: {tgt} ---")
        print(f"  💰 Lucro Real (Modelo Treinado) : ${lucro_real:,.2f} (Trades: {trades_real})")
        print(f"  🎲 Lucro Shuffle (Modelo Aleatório): ${lucro_shuf:,.2f} (Trades: {trades_shuf})")
        print(f"  📈 Retorno Real: {real['retorno_total']:+.2f}% | Drawdown: {real['drawdown_max']:.2f}%")
        print(f"  📊 Retorno Shuffle: {shuf.get('retorno_total', 0):+.2f}% | Drawdown: {shuf.get('drawdown_max', 0):.2f}%")
        print(f"  🛡️  SCORE DE ROBUSTEZ: {robustez_score:.2f}%")
        
        if robustez_score >= 70:
            print("  ✅ ROBUSTEZ ALTA: O modelo tem uma vantagem estatística clara sobre o ruído.")
        elif robustez_score >= 40:
            print("  ⚠️ ROBUSTEZ MÉDIA: O modelo tem alguma vantagem, mas o overfitting é uma preocupação.")
        else:
            print("  ❌ ROBUSTEZ BAIXA: O modelo está próximo de um gerador de sinais aleatórios. Overfitting provável.")
            
    print("█"*60 + "\n")


# Executar Módulo 10 com Meta-Labeling e Shuffle Test
modulo_10_teste_robustez(df_all, lista_targets, modelos_cache, probs_cache, meta_modelos_cache)



# ========================================================================
# 🟢 BLOCOS DE ELITE — ADICIONADOS AO FINAL (SEM MUTILAR O ORIGINAL)
# ========================================================================

def otimizar_hiperparametros_v25(X_train, y_train, X_val, y_val, n_trials=30):
    try:
        import optuna
    except ImportError:
        return None
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def detectar_regimes_mercado_v25(df, n_regimes=4):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    regime_features = [c for c in ['vol_realized', 'rsi_14', 'atr14', 'slope20'] if c in df.columns]
    if not regime_features:
        df['temp_ret'] = df['close'].pct_change(20)
        regime_features = ['temp_ret']
    X_regime = df[regime_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df['market_regime'] = kmeans.fit_predict(X_scaled)
    return df

def treinar_meta_model_v25(X_train, y_train, X_val, y_val, model_primario):
    from sklearn.ensemble import RandomForestClassifier
    y_proba_val = model_primario.predict_proba(X_val)
    y_pred_val = np.argmax(y_proba_val, axis=1)
    y_meta = (y_pred_val == y_val).astype(int)
    conf_val = np.max(y_proba_val, axis=1).reshape(-1, 1)
    X_meta = np.hstack([X_val, conf_val])
    meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    meta_model.fit(X_meta, y_meta)
    return meta_model

def executar_backtest_realista_v25(df, y_pred, y_proba, X_test=None, meta_model=None, h=10, capital_inicial=10000.0):
    capital = capital_inicial
    comissao, slippage = 0.001, 0.0005
    trades = []
    for i in range(len(y_pred)):
        conf = np.max(y_proba[i])
        if conf < 0.55: continue
        if meta_model is not None and X_test is not None:
            X_meta_row = np.hstack([X_test[i], conf]).reshape(1, -1)
            if meta_model.predict(X_meta_row)[0] == 0: continue
        pred = y_pred[i]
        if pred == 0: continue
        pos_size = 0.05 
        entrada_real = df.iloc[i]["close"] * (1 + slippage)
        if i + h >= len(df): break
        saida_real = df.iloc[i + h]["close"] * (1 - slippage)
        pnl_liquido = ((saida_real - entrada_real) / entrada_real) - (comissao * 2)
        capital += capital * pos_size * pnl_liquido * 10
        trades.append(pnl_liquido)
    print(f"\n>>> BACKTEST REALISTA FINALIZADO. Capital Final: ${capital:,.2f}")
    return capital, trades
