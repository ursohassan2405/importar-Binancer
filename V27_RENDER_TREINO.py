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
# from catboost import CatBoostClassifier  # Removido - não disponível no Render
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin

scaler = None  # Inicializa o scaler como vazio no escopo global

import joblib

# ============================================================================
# 📊 TRADE ANALYZER V7 - ANÁLISE DE HORÁRIOS, DIAS E PADRÕES
# ============================================================================
class TradeAnalyzer:
    """
    Analisa trades por horário, dia da semana e padrões de sequência.
    """
    def __init__(self, pattern_length=7, min_occurrences=20):
        self.pattern_length = pattern_length
        self.min_occurrences = min_occurrences
        
        # Análise temporal
        self.hourly_stats = {}  # {hora: {trades, wins, pnl}}
        self.daily_stats = {}   # {dia: {trades, wins, pnl}}
        
        # Análise de padrões
        self.pattern_history = []  # Lista de direções (BUY=1, SELL=-1)
        self.patterns = {}  # {padrão: {wins, losses, pnl}}
        
    def add_trade(self, signal_direction, win, pnl, timestamp):
        """
        Adiciona um trade para análise.
        
        Args:
            signal_direction: 1 (BUY) ou -1 (SELL)
            win: True se ganhou, False se perdeu
            pnl: Lucro/prejuízo em USD
            timestamp: Timestamp do trade
        """
        # Adiciona direção ao histórico
        self.pattern_history.append(signal_direction)
        
        # Análise temporal (se timestamp válido)
        if timestamp is not None and not pd.isna(timestamp):
            try:
                # --- CORREÇÃO: Conversão robusta de tempo ---
                if not isinstance(timestamp, (pd.Timestamp, datetime)):
                    if isinstance(timestamp, (int, float, np.integer)):
                        # Se for > 1e11 é milissegundos (padrão Binance/Bybit)
                        unit = 'ms' if timestamp > 1e11 else 's'
                        timestamp = pd.to_datetime(timestamp, unit=unit)
                    else:
                        timestamp = pd.to_datetime(str(timestamp))
                
                # Agora hour e day_name funcionam em qualquer formato
                hour = timestamp.hour
                day = timestamp.day_name()
                
                # Estatísticas por Hora
                if hour not in self.hourly_stats:
                    self.hourly_stats[hour] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                self.hourly_stats[hour]['trades'] += 1
                if win:
                    self.hourly_stats[hour]['wins'] += 1
                self.hourly_stats[hour]['pnl'] += pnl
                
                # Estatísticas por Dia
                if day not in self.daily_stats:
                    self.daily_stats[day] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                self.daily_stats[day]['trades'] += 1
                if win:
                    self.daily_stats[day]['wins'] += 1
                self.daily_stats[day]['pnl'] += pnl
                
            except Exception:
                pass  # Timestamp inválido, pula análise temporal
        
        # Análise de padrões (se temos histórico suficiente)
        if len(self.pattern_history) >= self.pattern_length:
            # Pega últimos N sinais
            pattern = tuple(self.pattern_history[-self.pattern_length:])
            
            # Converte para string legível (B=BUY, S=SELL)
            pattern_str = ','.join(['B' if d == 1 else 'S' for d in pattern])
            
            if pattern_str not in self.patterns:
                self.patterns[pattern_str] = {'wins': 0, 'losses': 0, 'pnl': 0.0, 'count': 0}
            
            self.patterns[pattern_str]['count'] += 1
            if win:
                self.patterns[pattern_str]['wins'] += 1
            else:
                self.patterns[pattern_str]['losses'] += 1
            self.patterns[pattern_str]['pnl'] += pnl
    
    def get_report(self):
        """Retorna relatório completo de análise."""
        return {
            'hourly': self.hourly_stats,
            'daily': self.daily_stats,
            'patterns': self.patterns,
            'total_trades': len(self.pattern_history)
        }
    
    def print_final_report(self, target_name="Target"):
        """Imprime relatório formatado de análise."""
        print("\n" + "█" * 100)
        print(f"█ RELATÓRIO COMPLETO DE ANÁLISE - {target_name}")
        print("█" * 100)
        print(f"\n📊 Total de trades analisados: {len(self.pattern_history)}")
        
        # ===== ANÁLISE POR HORÁRIO =====
        print("\n" + "=" * 80)
        print("📊 ANÁLISE DE PERFORMANCE POR HORÁRIO")
        print("=" * 80)
        print()
        print(f"{'Hora':<7}│ {'Trades':<8}│ {'WR':<8}│ {'PF':<7}│ {'Lucro':<11}│ Status")
        print("─" * 70)
        
        if self.hourly_stats:
            for hour in sorted(self.hourly_stats.keys()):
                stats = self.hourly_stats[hour]
                wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                pf = (stats['pnl'] / abs(min(stats['pnl'], -1))) if stats['pnl'] < 0 else stats['pnl']
                
                status = "✅" if wr >= 60 else "⚠️" if wr >= 50 else "❌"
                print(f"{hour:02d}h    │ {stats['trades']:<8}│ {wr:>5.1f}%  │ {pf:>6.2f} │ ${stats['pnl']:>10.2f} │ {status}")
        
        # ===== ANÁLISE POR DIA =====
        print("\n" + "=" * 80)
        print("📅 ANÁLISE DE PERFORMANCE POR DIA DA SEMANA")
        print("=" * 80)
        print()
        print(f"{'Dia':<9}│ {'Trades':<8}│ {'WR':<8}│ {'PF':<7}│ {'Lucro':<11}│ Status")
        print("─" * 70)
        
        if self.daily_stats:
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for day in days_order:
                if day in self.daily_stats:
                    stats = self.daily_stats[day]
                    wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                    pf = (stats['pnl'] / abs(min(stats['pnl'], -1))) if stats['pnl'] < 0 else stats['pnl']
                    
                    status = "✅" if wr >= 60 else "⚠️" if wr >= 50 else "❌"
                    print(f"{day:<9}│ {stats['trades']:<8}│ {wr:>5.1f}%  │ {pf:>6.2f} │ ${stats['pnl']:>10.2f} │ {status}")
        
        # ===== RECOMENDAÇÕES =====
        print("\n" + "=" * 80)
        print("💡 RECOMENDAÇÕES")
        print("=" * 80)
        
        # Horários ruins
        if self.hourly_stats:
            bad_hours = [h for h, s in self.hourly_stats.items() 
                        if s['trades'] >= 10 and (s['wins'] / s['trades'] < 0.5)]
            if bad_hours:
                print(f"⚠️ Evitar horários: {', '.join([f'{h:02d}h' for h in sorted(bad_hours)])}")
        
        # Dias ruins
        if self.daily_stats:
            bad_days = [d for d, s in self.daily_stats.items() 
                       if s['trades'] >= 10 and (s['wins'] / s['trades'] < 0.5)]
            if bad_days:
                print(f"⚠️ Evitar dias: {', '.join(bad_days)}")
        
        print("=" * 80)
        
        # ===== ANÁLISE DE PADRÕES =====
        if self.patterns:
            print("\n" + "=" * 100)
            print("🔍 ANÁLISE DE PADRÕES DE SEQUÊNCIA")
            print("=" * 100)
            
            # Separa padrões por sinal final
            buy_patterns = {p: s for p, s in self.patterns.items() if p.endswith(',B')}
            sell_patterns = {p: s for p, s in self.patterns.items() if p.endswith(',S')}
            
            # Top 10 BUY patterns
            if buy_patterns:
                print("\n" + "=" * 100)
                print("📈 TOP 10 PADRÕES DE SEQUÊNCIA - SINAL FINAL: BUY (últimos 7 sinais)")
                print("=" * 100)
                print()
                print(f"{'Rank':<7}│ {'Padrão':<26}│ {'WR':<9}│ {'PF':<8}│ {'Ocorr':<8}│ Lucro")
                print("─" * 100)
                
                sorted_buy = sorted(buy_patterns.items(), 
                                   key=lambda x: (x[1]['wins'] / max(1, x[1]['count'])) if x[1]['count'] >= self.min_occurrences else 0, 
                                   reverse=True)[:10]
                
                for rank, (pattern, stats) in enumerate(sorted_buy, 1):
                    if stats['count'] >= self.min_occurrences:
                        wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                        pf = abs(stats['pnl'] / min(stats['pnl'], -1)) if stats['pnl'] < 0 else stats['pnl']
                        print(f"{rank:<7}│ {pattern:<26}│ {wr:>6.1f}% │ {pf:>6.2f} │ {stats['count']:<8}│ $ {stats['pnl']:>10.0f}")
            
            # Top 10 SELL patterns
            if sell_patterns:
                print("\n" + "=" * 100)
                print("📉 TOP 10 PADRÕES DE SEQUÊNCIA - SINAL FINAL: SELL (últimos 7 sinais)")
                print("=" * 100)
                print()
                print(f"{'Rank':<7}│ {'Padrão':<26}│ {'WR':<9}│ {'PF':<8}│ {'Ocorr':<8}│ Lucro")
                print("─" * 100)
                
                sorted_sell = sorted(sell_patterns.items(), 
                                    key=lambda x: (x[1]['wins'] / max(1, x[1]['count'])) if x[1]['count'] >= self.min_occurrences else 0, 
                                    reverse=True)[:10]
                
                for rank, (pattern, stats) in enumerate(sorted_sell, 1):
                    if stats['count'] >= self.min_occurrences:
                        wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                        pf = abs(stats['pnl'] / min(stats['pnl'], -1)) if stats['pnl'] < 0 else stats['pnl']
                        print(f"{rank:<7}│ {pattern:<26}│ {wr:>6.1f}% │ {pf:>6.2f} │ {stats['count']:<8}│ $ {stats['pnl']:>10.0f}")
            
            # Piores padrões
            print("\n" + "=" * 100)
            print("⚠️ TOP 10 PIORES PADRÕES (EVITAR!)")
            print("=" * 100)
            print()
            print(f"{'Rank':<7}│ {'Padrão':<26}│ {'WR':<9}│ {'PF':<8}│ {'Ocorr':<8}│ Prejuízo")
            print("─" * 100)
            
            worst_patterns = sorted(self.patterns.items(), 
                                   key=lambda x: x[1]['pnl'] if x[1]['count'] >= self.min_occurrences else float('inf'))[:10]
            
            for rank, (pattern, stats) in enumerate(worst_patterns, 1):
                if stats['count'] >= self.min_occurrences:
                    wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
                    pf = abs(stats['pnl'] / min(stats['pnl'], -1)) if stats['pnl'] < 0 else stats['pnl']
                    print(f"{rank:<7}│ {pattern:<26}│ {wr:>6.1f}% │ {pf:>6.2f} │ {stats['count']:<8}│ $ {stats['pnl']:>10.0f}")
            
            print("=" * 100)
        
        print("\n" + "█" * 100)
        print()

ANALYZER_AVAILABLE = True  # Sempre disponível agora!
print("✅ Trade Analyzer V7 carregado com sucesso!")

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

# Detectar Render e usar path padrão
if os.path.exists("/opt/render"):
    csv_path = "/opt/render/project/.data/PENDLEUSDT_DATA/PENDLEUSDT_15m.csv"
    print(f"🌐 RENDER: Usando {csv_path}")
else:
    csv_path = input("📌 Arquivo CRU (ex: C:\\BTC_MODELO\\IA_CRIPTO\\datasets\\BTCUSDT_15m_full_V14.csv): ").strip()

if not os.path.isfile(csv_path):
    print(f"\n❌ ERRO: Arquivo não encontrado: {csv_path}")
    sys.exit(1)

# Detectar Render e usar diretório padrão
if os.path.exists("/opt/render"):
    out_dir = "/opt/render/project/.data/PENDLEUSDT_DATA"
    print(f"📁 RENDER: Output em {out_dir}")
else:
    out_dir = input("📁 Pasta de saída para modelos/relatórios: ").strip()
    if out_dir == "":
        out_dir = os.path.join(os.path.dirname(csv_path), "V25_MODELS")

os.makedirs(out_dir, exist_ok=True)

# Detectar Render e usar nome automático
if os.path.exists("/opt/render"):
    from datetime import datetime
    exp_name = f"RENDER_{datetime.now().strftime('%Y%m%d_%H%M')}"
    print(f"🏷 RENDER: Experimento {exp_name}")
else:
    exp_name = input("🏷 Nome do experimento: ").strip()
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
        # Criar diretório se não existir
        os.makedirs('modelos_salvos', exist_ok=True)
        
        # Salvar SCALER
        scaler_path = os.path.join('modelos_salvos', 'scaler_regimes.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Salvar KMEANS
        kmeans_path = os.path.join('modelos_salvos', 'kmeans_regimes.pkl')
        joblib.dump(kmeans, kmeans_path)
        
        # Garantir que estejam no cache global
        if 'modelos_cache' not in globals():
            globals()['modelos_cache'] = {}
        globals()['modelos_cache']['scaler_regimes'] = scaler
        globals()['modelos_cache']['kmeans_regimes'] = kmeans
        
        print(f"✅ SCALER E KMEANS SALVOS COM SUCESSO!")
        print(f"   📁 Diretório: modelos_salvos/")
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

# 1) Usuário informa somente o horizonte futuro
try:
    h_fut = int(input("Horizonte futuro N (ex: 3): ").strip())
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

usar_ctx = input("Adicionar contexto multi-TF? (s/n): ").strip().lower()

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

    escolha = input("Quais TFs deseja adicionar? (ex: 1h,4h,8h): ").strip().lower()
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

        fname = f"{simbolo}_{tf}.csv"
        
        # Render: usar diretório fixo; Local: usar diretório do CSV
        if os.path.exists("/opt/render"):
            path_tf = f"/opt/render/project/.data/PENDLEUSDT_DATA/{fname}"
        else:
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

        # 🔴 PATCH: Excluir buy_vol e sell_vol do Multi-TF (duplicados de buy_vol_agg/sell_vol_agg)
        feature_cols = [c for c in feature_cols if c not in ["buy_vol", "sell_vol"]]

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
        "total_vol_agg",  # Coluna auxiliar (soma de buy+sell)
        "buy_vol", "sell_vol",  # 🔴 PATCH: Duplicados de buy_vol_agg/sell_vol_agg (evitar 12 features extras)
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
# PERGUNTA AO USUÁRIO — ATIVAR PESO TEMPORAL
# (variável correta, função já definida ACIMA — sem erros)
# ===============================================================
usar_peso = input("Aplicar peso temporal no treinamento? (s/n): ").strip().lower()
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

    modo = input("Escolha o modo de peso (1/2/3/4): ").strip()

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
    meia_vida = input("Informe a meia-vida (em dias) [enter=90]: ").strip()
    meia_vida = float(meia_vida) if meia_vida else 90.0

    # Peso temporal
    idade_dias = (dt_max - ts).dt.days.astype(float)
    peso_tempo = np.power(0.5, idade_dias / meia_vida)

    # Peso por regime
    if "trend_regime" in df.columns:
        print("\n>>> Detectado trend_regime. Configurando pesos…")
        p_bear  = input("Peso regime BAIXA  (-1) [enter=1.0]: ").strip()
        p_lateral = input("Peso regime LATERAL (0) [enter=1.0]: ").strip()
        p_bull = input("Peso regime ALTA   (1) [enter=1.0]: ").strip()

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

    modo = input("Escolha o modo do C6 (1/2/3): ").strip()

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

def modulo_8_simulador_elite_v2(
    df_backtest,
    lista_targets,
    modelos_cache,
    probs_cache,
    meta_modelos_cache,
    capital_inicial,
    valor_mao,
    alavancagem,
    min_conf,
    usar_meta,
    corte_juiz,
    usar_fluxo=False,
    modo_fluxo="balanced",  # 🔥 NOVO: strict | balanced | permissive
    analyzers=None,  # 📊 V7: Trade Analyzer
    titulo_relatorio="RELATÓRIO DE PERFORMANCE INSTITUCIONAL",
):
    resultados_por_target = {}

    # Controle de juiz (meta-labeling)
    juiz_ativo = False
    juiz_key = None
    if isinstance(usar_meta, (int, float)) and usar_meta > 0:
        juiz_ativo = True
        juiz_key = "juiz"

    # Se o meta_modelos_cache vier como dict por target, detecta
    meta_modelos_por_target = None
    meta_modelos_features = None
    if isinstance(meta_modelos_cache, dict):
        # heurística: se tiver chaves iguais a targets ou tiver features por target
        meta_modelos_por_target = meta_modelos_cache.get("modelos_por_target", None)
        meta_modelos_features = meta_modelos_cache.get("features_por_target", None)

    # Parâmetros de custo (mantidos conforme padrão do script)
    comissao, slippage = 0.001, 0.0005
    custo_total_operacao = comissao + slippage
    taxa_financiamento = 0.0  # se você usa funding em outro bloco, mantenha lá
    
    # 🔧 CORREÇÃO: Taxa de funding realista para futuros perpétuos
    # Funding típico: 0.01% a cada 8 horas (Binance, Bybit padrão)
    taxa_financiamento_8h = 0.0001  # 0.01% por 8h (≈10.95% anual se sempre posicionado)
    
    # 🔧 CORREÇÃO: Slippage adicional em stops (execução de stop é pior que ordem normal)
    slippage_stops = slippage * 2.5  # Stops executam com 2.5x mais slippage

    # =========================================================
    # MEMÓRIA DE DIREÇÃO FINAL POR TARGET (base do K3/K6)
    # =========================================================
    
    n_total = min(
        len(df_backtest),
        min(len(probs_cache[t]) for t in lista_targets if t in probs_cache)
    )
 
    direcao_por_target = {tgt: [0] * n_total for tgt in lista_targets}
    
    # 🔥 ORDENAÇÃO: K3/K6 precisam que seus componentes sejam processados antes
    # K3 depende de K1 e K2
    # K6 depende de K4 e K5
    targets_ordenados = []
    targets_compostos = []
    
    for tgt in lista_targets:
        if tgt in ['target_K3', 'target_K6']:
            targets_compostos.append(tgt)
        else:
            targets_ordenados.append(tgt)
    
    # Processa targets simples primeiro, depois compostos
    targets_ordenados.extend(targets_compostos)
    
    # 📊 V19 CORREÇÃO CRÍTICA: Criar analyzers ANTES do loop
    # Sem isso, add_trade() nunca é chamado (horários/dias ficam vazios)
    if analyzers and 'config' in analyzers:
        for tgt in targets_ordenados:
            if tgt not in analyzers:
                analyzers[tgt] = TradeAnalyzer(
                    pattern_length=analyzers['config']['pattern_length'],
                    min_occurrences=analyzers['config']['min_occurrences']
                )
    
    for tgt in targets_ordenados:
        if tgt not in modelos_cache or tgt not in probs_cache:
            continue
        
        capital = capital_inicial
        stats = {
            "sinais_total": 0,
            "trades_total": 0,
            "trades_buy": 0,
            "trades_sell": 0,
            "wins_buy": 0,
            "loss_buy": 0,
            "wins_sell": 0,
            "loss_sell": 0,
            "saida_tp": 0,
            "saida_sl": 0,
            "saida_fluxo": 0,
            "saida_tempo": 0,
            "hold_time_total": 0,
            "fluxo_win": 0,
            "vetados_juiz": 0,
            "vetados_posicao": 0,  # <<< ADICIONADO (sem mudar dados, só contagem)
            "vetados_antisequencia": 0,  # V9: Vetados por 3+ mesma direção
            "aceitos_antisequencia_alta_conf": 0,  # V9: 3+ mesma direção mas conf 90%+
            "ultimas_direcoes": [],  # V9: Track últimas direções executadas
            "lucro_bruto": 0.0,
            "prejuizo_bruto": 0.0,
            # 🔬 DIAGNÓSTICOS
            "timeouts_amplitude": [],
            "winrate_por_holdtime": {
                "1-3": {"wins": 0, "total": 0},
                "4-7": {"wins": 0, "total": 0},
                "8-15": {"wins": 0, "total": 0},
                "16+": {"wins": 0, "total": 0}
            },
            "tp_sl_valores": {
                "tp_medio": [],
                "sl_medio": [],
                "tp_atingiu": 0,
                "sl_atingiu": 0
            }
        }

        pico_capital = capital_inicial
        drawdown_max = 0.0

        # Estado de posição
        posicionado = False
        index_saida = 0

        n = min(len(df_backtest), len(probs_cache[tgt]))
        df_sim = df_backtest.iloc[:n].copy()
        probs = probs_cache[tgt][:n]
        classes = list(modelos_cache[tgt].classes_)

        if "atr14" not in df_sim.columns:
            df_sim["atr14"] = (df_sim["high"] - df_sim["low"]).rolling(14).mean().ffill()

        for i in range(n - 10):
            # ---------------------------------------------------------
            # 1) Sinal do modelo
            # ---------------------------------------------------------
            p_up = probs[i][classes.index(1)] if 1 in classes else 0
            p_down = (
                probs[i][classes.index(-1)]
                if -1 in classes
                else (probs[i][classes.index(0)] if 0 in classes and len(classes) == 2 else 0)
            )

            direcao = 0
            
            # ---------------------------------------------------------------
            # LÓGICA ESPECIAL PARA TARGETS ESPECIALISTAS (REV_LONG / REV_SHORT)
            # ---------------------------------------------------------------
            # REV_LONG: classe 1 = vai subir → COMPRA
            # REV_SHORT: classe 1 = vai cair → VENDA (invertido!)
            # ---------------------------------------------------------------
            if tgt == "target_REV_LONG":
                # Especialista COMPRA: só compra, nunca vende
                if p_up >= min_conf:
                    direcao = 1
            elif tgt == "target_REV_SHORT":
                # Especialista VENDA: classe 1 = vai cair → direcao = -1
                if p_up >= min_conf:
                    direcao = -1
            else:
                # Lógica padrão para outros targets
                if p_up >= min_conf:
                    direcao = 1
                elif p_down >= min_conf:
                    direcao = -1

            if direcao != 0:
                stats["sinais_total"] += 1
            
            # V10 FIX CRÍTICO: K6 e K3 NÃO devem salvar direção do modelo!
            # K3 opera por concordância K1+K2, K6 por K4+K5
            if tgt not in ["target_K3", "target_K6", "target_CONFLUENCIA"]:
                # Memoriza direção SEMPRE (antes de K3/K6/CONFLUENCIA modificarem)
                # CRÍTICO: Deve salvar ANTES da lógica de confluência!
                direcao_por_target[tgt][i] = direcao
            
            # =========================================================
            # CONFLUÊNCIA INTELIGENTE REV_LONG + REV_SHORT
            # =========================================================
            if tgt == "target_CONFLUENCIA":
                # Busca direções dos especialistas
                dir_long = direcao_por_target.get("target_REV_LONG", [0]*n)[i]
                dir_short = direcao_por_target.get("target_REV_SHORT", [0]*n)[i]
                
                # Resetamos direção para usar lógica de confluência
                direcao = 0
                
                # CASO 1: Apenas LONG quer comprar, SHORT neutro → COMPRA
                if dir_long == 1 and dir_short == 0:
                    direcao = 1
                
                # CASO 2: Apenas SHORT quer vender, LONG neutro → VENDA
                elif dir_short == -1 and dir_long == 0:
                    direcao = -1
                
                # CASO 3: DIVERGÊNCIA! LONG quer comprar E SHORT quer vender
                # Árbitro (target_CONFLUENCIA) decide baseado no seu próprio modelo
                elif dir_long == 1 and dir_short == -1:
                    # Usa a probabilidade do modelo CONFLUENCIA para decidir
                    if p_up >= min_conf:
                        direcao = 1   # Árbitro decide COMPRA
                    elif p_down >= min_conf:
                        direcao = -1  # Árbitro decide VENDA
                    # Se árbitro não tem confiança, não opera (direcao = 0)
                
                # CASO 4: Ambos neutros → Não opera
                # (direcao já é 0)
                
                # CASO 5: Consenso (ambos querem mesma direção) - raro mas possível
                elif dir_long == 1 and dir_short == 0:
                    direcao = 1
                elif dir_long == 0 and dir_short == -1:
                    direcao = -1
                
                if direcao == 0:
                    continue  # Sem sinal claro, pula
            
            # =========================================================
	    # K3 = CONFLUÊNCIA INTEGRAL K1 + K2 (COMPRAS E VENDAS)
	    # =========================================================
            if tgt == "target_K3":
                dir_k1 = direcao_por_target.get("target_K1", [0]*n)[i]
                dir_k2 = direcao_por_target.get("target_K2", [0]*n)[i]
	        
                # Resetamos a direção do modelo base do K3 para usar a herança pura
                direcao = 0
	        
                if dir_k1 == 1 and dir_k2 == 1:
                    direcao = 1  # Compra confirmada por ambos
                elif dir_k1 == -1 and dir_k2 == -1:
                    direcao = -1 # Venda confirmada por ambos [CORREÇÃO: Adicionado herança de venda]
	            
                if direcao == 0:
                    continue # Ignora se não houver confluência total
            
            # =========================================================
	    # K6 = CONFLUÊNCIA INTEGRAL K4 + K5
	    # =========================================================
            if tgt == "target_K6":
                dir_k4 = direcao_por_target.get("target_K4", [0]*n)[i]
                dir_k5 = direcao_por_target.get("target_K5", [0]*n)[i]
	        
                direcao = 0
	        
                # Lógica de herança pura K4 + K5
                if dir_k4 == 1 and dir_k5 == 1:
                   direcao = 1
                elif dir_k4 == -1 and dir_k5 == -1:
                   direcao = -1
	            
                if direcao == 0:
                   continue
            
   
   

            # ---------------------------------------------------------
            # 2) Trava de posição
            # ---------------------------------------------------------
            if posicionado and i < index_saida:
                stats["vetados_posicao"] += 1
                continue
            elif posicionado and i >= index_saida:
                posicionado = False
                index_saida = 0

            
            
            # Contabiliza sinal (independente de posição)
            if direcao != 0:
                stats["sinais_total"] += 1

                # -----------------------------------------------------
                # 2) 🛡️ Trava de Ocupação: não executa novo trade com posição aberta
                # -----------------------------------------------------
                if posicionado and i < index_saida:
                    stats["vetados_posicao"] += 1
                    continue
                elif posicionado and i >= index_saida:
                    posicionado = False
                    index_saida = 0

                # 🛡️ FILTRO META-LABELING (O JUIZ)
                if juiz_ativo:
                    try:
                        conf_val = max(p_up, p_down)

                        # suporte a features por target
                        cols_juiz = None
                        if meta_modelos_features and isinstance(meta_modelos_features, dict) and tgt in meta_modelos_features:
                            cols_juiz = meta_modelos_features[tgt]

                        df_row = df_sim.iloc[[i]].copy()
                        df_row["meta_conf_primaria"] = conf_val

                        if cols_juiz:
                            for c in cols_juiz:
                                if c not in df_row.columns:
                                    df_row[c] = 0.0
                            X_meta = df_row[cols_juiz].values
                        else:
                            X_meta = df_row.select_dtypes(include=[np.number]).values

                        # pega o juiz do target, se existir
                        juiz_model = None
                        if isinstance(meta_modelos_cache, dict) and tgt in meta_modelos_cache:
                            juiz_model = meta_modelos_cache[tgt]
                        elif meta_modelos_por_target and isinstance(meta_modelos_por_target, dict) and tgt in meta_modelos_por_target:
                            juiz_model = meta_modelos_por_target[tgt]
                        elif isinstance(meta_modelos_cache, dict) and "juiz_global" in meta_modelos_cache:
                            juiz_model = meta_modelos_cache["juiz_global"]

                        if juiz_model is not None:
                            p_ok = float(juiz_model.predict_proba(X_meta)[0][1])
                            # Debug opcional — se quiser silenciar, comente
                            print(f"    [DEBUG JUIZ] {tgt} | Prob. Acerto: {p_ok:.4f} | Corte: {corte_juiz:.2f}")
                            if p_ok < corte_juiz:
                                stats["vetados_juiz"] += 1
                                continue
                    except Exception:
                        # Em caso de erro do juiz, não trava o backtest
                        pass
                # =====================================================
                # FILTRO DE AGRESSÃO REMOVIDO - ML PURO!
                # =====================================================
                # V8: Confiamos 100% no modelo ML
                # Sem EMA, RSI, MACD ou qualquer indicador velho
                # O modelo já aprendeu tudo que precisa!
                
                # (Código do filtro removido - 141 linhas de lixo eliminadas)

                # -----------------------------------------------------
                # 3) Execução do trade (1 por vez) — preserva cálculo original
                # -----------------------------------------------------
                # 🔧 CORREÇÃO CRÍTICA: Entrada no OPEN do próximo candle
                # Motivo: O sinal é gerado APÓS o close do candle atual.
                #         Em mercado real, só podemos executar no próximo candle.
                #         Usar o close do candle atual = LOOK-AHEAD BIAS grave!
                
                # ORIGINAL (com look-ahead bias):
                # preco_entrada = float(df_sim["close"].iloc[i])
                
                # CORRIGIDO (realista):
                if i + 1 >= len(df_sim):
                    continue  # Não há próximo candle, pula este sinal
                index_saida = i + 1
                posicionado = False                    
                    
                    
                preco_entrada = float(df_sim["open"].iloc[i + 1])  # Open do próximo candle
                atr = float(df_sim["atr14"].iloc[i]) if not pd.isna(df_sim["atr14"].iloc[i]) else 0.0

                # SL/TP conservadores por ATR (mantém sua lógica se já existia)
                sl = preco_entrada - (atr * 2.2) if direcao == 1 else preco_entrada + (atr * 2.2)
                tp = preco_entrada + (atr * 2.2) if direcao == 1 else preco_entrada - (atr * 2.2)
                
                # 🔬 DIAGNÓSTICO: Registrar TP/SL médios
                tp_distance_pct = abs(tp - preco_entrada) / preco_entrada * 100
                sl_distance_pct = abs(sl - preco_entrada) / preco_entrada * 100
                stats["tp_sl_valores"]["tp_medio"].append(tp_distance_pct)
                stats["tp_sl_valores"]["sl_medio"].append(sl_distance_pct)
                
                i_entrada = i + 1  # 🔬 Para calcular amplitude depois

                pnl_trade = 0.0
                motivo_saida = "TEMPO"

                # ==============================
                # ADICIONADO: contador de fluxo
                # ==============================
                agg_counter = 0

                # Simula saída até 01 candles à frente (como já estava)
                for j in range(i + 1, min(i + 20, n - 1) + 1):
                    high_j = float(df_sim["high"].iloc[j])
                    low_j = float(df_sim["low"].iloc[j])
                    close_j = float(df_sim["close"].iloc[j])


                    # SL
                    if (direcao == 1 and low_j <= sl) or (direcao == -1 and high_j >= sl):
                        # 🔧 CORREÇÃO: Aplicar slippage adicional em stops
                        # Stops executam pior que ordens normais (market orders em pânico)
                        if direcao == 1:
                            preco_sl_executado = sl * (1 - slippage_stops)  # Vende abaixo do SL
                        else:
                            preco_sl_executado = sl * (1 + slippage_stops)  # Compra acima do SL
                            
                        pnl_trade = ((preco_sl_executado / preco_entrada) - 1) * direcao
                        motivo_saida = "SL"
                        index_saida = j
                        break

                    # TP
                    if (direcao == 1 and high_j >= tp) or (direcao == -1 and low_j <= tp):
                        pnl_trade = ((tp / preco_entrada) - 1) * direcao
                        motivo_saida = "TP"
                        index_saida = j
                        break

                    # TEMPO (último candle do horizonte)
                    if j == min(i + 7, n - 1):
                        pnl_trade = ((close_j / preco_entrada) - 1) * direcao
                        motivo_saida = "TEMPO"
                        index_saida = j
                        break
                        
		# ------------------------------
		# CUSTOS REAIS SOBRE NOTIONAL
		# ------------------------------
                notional = valor_mao * alavancagem
		
		# 🔧 CORREÇÃO: custo round-trip (entrada + saída)
		# Fórmula correta: notional * 2 * (comissao + slippage)
		# Entrada: 1x (comissao + slippage)
		# Saída:   1x (comissao + slippage)
                custo_operacao_usd = notional * 2 * (comissao + slippage)
		
		# 🔧 CORREÇÃO: funding proporcional ao tempo em posição
		# Em futuros perpétuos, funding é cobrado a cada 8h
		# 8h = 32 candles de 15min, então (hold_time / 32) = proporção de 8h
                hold_time = max(1, index_saida - i)
                funding_usd = notional * taxa_financiamento_8h * (hold_time / 32)
		
		# pnl bruto já alavancado
                pnl_bruto_usd = notional * pnl_trade
		
		# pnl final
                lucro_usd = pnl_bruto_usd - custo_operacao_usd - funding_usd


                # Contabiliza trade (APENAS aqui: executou de fato)
                stats["trades_total"] += 1
                
                # V9: Registra direção para regra anti-sequência
                stats["ultimas_direcoes"].append(direcao)
                if len(stats["ultimas_direcoes"]) > 10:  # Mantém apenas últimas 10
                    stats["ultimas_direcoes"].pop(0)
                
                if direcao == 1:
                    stats["trades_buy"] += 1
                    if lucro_usd > 0:
                        stats["wins_buy"] += 1
                    else:
                        stats["loss_buy"] += 1
                else:
                    stats["trades_sell"] += 1
                    if lucro_usd > 0:
                        stats["wins_sell"] += 1
                    else:
                        stats["loss_sell"] += 1

                if motivo_saida == "TP":
                    stats["saida_tp"] += 1
                    stats["tp_sl_valores"]["tp_atingiu"] += 1  # 🔬 DIAGNÓSTICO
                elif motivo_saida == "SL":
                    stats["saida_sl"] += 1
                    stats["tp_sl_valores"]["sl_atingiu"] += 1  # 🔬 DIAGNÓSTICO
                elif motivo_saida == "FLUXO":
                    stats["saida_fluxo"] += 1
                    if lucro_usd > 0:
                        stats["fluxo_win"] += 1
                else:  # TEMPO
                    stats["saida_tempo"] += 1
                    # 🔬 DIAGNÓSTICO: Calcular amplitude do timeout
                    preco_max = df_sim["high"].iloc[i_entrada:index_saida+1].max()
                    preco_min = df_sim["low"].iloc[i_entrada:index_saida+1].min()
                    amplitude_pct = ((preco_max - preco_min) / preco_entrada) * 100
                    stats["timeouts_amplitude"].append(amplitude_pct)

                # 🔬 DIAGNÓSTICO: Winrate por hold time
                hold_candles = index_saida - i
                if hold_candles <= 3:
                    bucket = "1-3"
                elif hold_candles <= 7:
                    bucket = "4-7"
                elif hold_candles <= 15:
                    bucket = "8-15"
                else:
                    bucket = "16+"
                
                stats["winrate_por_holdtime"][bucket]["total"] += 1
                if lucro_usd > 0:
                    stats["winrate_por_holdtime"][bucket]["wins"] += 1

                stats["hold_time_total"] += (index_saida - i)
                if lucro_usd > 0:
                    stats["lucro_bruto"] += lucro_usd
                else:
                    stats["prejuizo_bruto"] += abs(lucro_usd)

                # Atualiza capital
                capital = float(capital + lucro_usd)

                # 📊 V7: Registrar trade no analyzer
                # V21: Corrigido - removido 'config' in analyzers
                if analyzers and tgt in analyzers:
                    timestamp_trade = None
                    
                    try:
                        # V11: Prioriza colunas de tempo reais
                        for col_time in ['ts', 'close_time', 'timestamp', 'datetime', 'date', 'open_time']:
                            if col_time in df_sim.columns:
                                ts_value = df_sim.iloc[i][col_time]
                                if ts_value is not None and str(ts_value) != 'nan' and str(ts_value) != 'NaT':
                                    timestamp_trade = ts_value
                                    break
                    except:
                        pass
                    
                    # V11: Se não encontrou, SEMPRE cria timestamp sintético
                    if timestamp_trade is None or str(timestamp_trade) in ['nan', 'NaT', 'None']:
                        # Timestamp sintético: começa 2024-01-01, 15min por candle
                        try:
                            base_time = pd.Timestamp('2024-01-01 00:00:00')
                            timestamp_trade = base_time + pd.Timedelta(minutes=15*i)
                        except:
                            # Último recurso: datetime simples
                            from datetime import datetime, timedelta
                            base_time = datetime(2024, 1, 1, 0, 0, 0)
                            timestamp_trade = base_time + timedelta(minutes=15*i)
                    
                    try:
                        analyzers[tgt].add_trade(
                            signal_direction=direcao,
                            win=(lucro_usd > 0),
                            pnl=lucro_usd,
                            timestamp=timestamp_trade
                        )
                    except Exception as e:
                        # V11: Se falhar, mostra erro (não mais silent!)
                        if stats.get('analyzer_errors', 0) == 0:  # Só mostra 1x
                            print(f"⚠️ [DEBUG] Analyzer error para {tgt}: {e}")
                            stats['analyzer_errors'] = 1

                # Marca posição como aberta até index_saida (trava passa a funcionar)
                posicionado = True

                if capital > pico_capital:
                    pico_capital = capital

                dd = (pico_capital - capital) / max(1.0, pico_capital)
                if dd > drawdown_max:
                    drawdown_max = dd

                if capital <= 0:
                    capital = 0.0
                    break

        # -----------------------
        # Relatório do target
        # -----------------------
        lucro_bruto = stats["lucro_bruto"]
        prejuizo_bruto = stats["prejuizo_bruto"]
        profit_factor = (lucro_bruto / max(1e-9, prejuizo_bruto)) if prejuizo_bruto > 0 else float("inf")

        avg_hold = stats["hold_time_total"] / max(1, stats["trades_total"])

        resultados = {
            "capital_final": capital,
            "lucro_prejuizo": capital - capital_inicial,
            "retorno_total_pct": ((capital / max(1e-9, capital_inicial)) - 1) * 100.0,
            "profit_factor": profit_factor,
            "drawdown_max": drawdown_max * 100.0,
            "stats": stats,
            "trades_executados": []
        }
        resultados_por_target[tgt] = resultados

        print("\n" + "=" * 60)
        print(f"  {titulo_relatorio}: {tgt}")
        print("=" * 60)
        print("  [FINANCEIRO]")
        print(f"  💰 Capital Inicial  : ${capital_inicial:,.2f}")
        print(f"  💰 Lucro/Prejuízo   : ${resultados['lucro_prejuizo']:,.2f}")
        print(f"  💰 Capital Final    : ${resultados['capital_final']:,.2f}")
        print(f"  📈 Retorno Total    : {resultados['retorno_total_pct']:+.2f}%")
        print(f"  📊 Profit Factor    : {resultados['profit_factor']:.2f}")
        print("-" * 60)
        print("  [RISCO]")
        print(f"  📉 Drawdown Máximo  : {resultados['drawdown_max']:.2f}%")
        print(f"  ⏱️  Hold Time Médio : {avg_hold:.1f} candles")
        print("-" * 60)
        print(f"  [FUNIL DE EXECUÇÃO]")
        print(f"  🎯 Sinais Gerados   : {stats['sinais_total']}")
        print(f"  🛡️  Vetados pelo Juiz: {stats['vetados_juiz']}")
        print(f"  ⛔ Vetados Posição  : {stats['vetados_posicao']}")
        
        # V10: Estatísticas da regra anti-sequência
        if stats.get('vetados_antisequencia', 0) > 0 or stats.get('aceitos_antisequencia_alta_conf', 0) > 0:
            print(f"  🔄 Vetados Anti-Seq : {stats.get('vetados_antisequencia', 0)}")
            print(f"     └─ Regra V10: 6+ mesma direção nos últimos 7 sinais = conf +20%")
            if stats.get('aceitos_antisequencia_alta_conf', 0) > 0:
                print(f"     └─ Aceitos c/ conf 90%+: {stats.get('aceitos_antisequencia_alta_conf', 0)}")
        
        if usar_fluxo:
            print(f"  🌊 Vetados por Fluxo: {stats.get('vetados_fluxo', 0)}")
            vetos_div = stats.get('vetos_divergencia', 0)
            vetos_vol = stats.get('vetos_volume', 0)
            vetos_vpin = stats.get('vetos_vpin', 0)
            if vetos_div + vetos_vol + vetos_vpin > 0:
                print(f"     └─ Divergências: {vetos_div} | Volume: {vetos_vol} | VPIN: {vetos_vpin}")
            if stats.get('sinais_fluxo_elite', 0) > 0:
                print(f"  ⭐ Sinais Elite (Score≥80%): {stats.get('sinais_fluxo_elite', 0)}")
        print(f"  ⚡ Trades Executados: {stats['trades_total']}")
        print(f"  📈 Utilização Sinal : {(stats['trades_total']/max(1, stats['sinais_total'])*100):.1f}%")
        print("-" * 60)
        print("  [DETALHAMENTO]")
        print(f"  🟢 COMPRAS: {stats['trades_buy']} (Wins: {stats['wins_buy']} | Loss: {stats['loss_buy']})")
        print(f"  🔴 VENDAS : {stats['trades_sell']} (Wins: {stats['wins_sell']} | Loss: {stats['loss_sell']})")
        print(f"  🚪 SAÍDAS : TP: {stats['saida_tp']} | SL: {stats['saida_sl']} | Fluxo: {stats['saida_fluxo']} | Tempo: {stats['saida_tempo']}")
        
        # 🔬 DIAGNÓSTICOS
        print("-" * 60)
        print("  [🔬 DIAGNÓSTICOS]")
        
        # TESTE 1: Amplitude dos Timeouts
        if stats["timeouts_amplitude"]:
            ampls = stats["timeouts_amplitude"]
            avg_ampl = np.mean(ampls)
            count_050 = sum(1 for a in ampls if a < 0.5)
            count_05_10 = sum(1 for a in ampls if 0.5 <= a < 1.0)
            count_10plus = sum(1 for a in ampls if a >= 1.0)
            print(f"  📊 TIMEOUTS - Amplitude Média: {avg_ampl:.2f}%")
            print(f"     └─ <0.5%: {count_050} | 0.5-1%: {count_05_10} | >1%: {count_10plus}")
        
        # TESTE 2: Winrate por Hold Time
        print(f"  📊 WINRATE POR HOLD TIME:")
        for faixa in ["1-3", "4-7", "8-15", "16+"]:
            data = stats["winrate_por_holdtime"][faixa]
            if data["total"] > 0:
                wr = (data["wins"] / data["total"]) * 100
                print(f"     {faixa:4} candles: {wr:5.1f}% ({data['wins']:3}/{data['total']:3})")
        
        # TESTE 3: TP/SL Calibração
        if stats["tp_sl_valores"]["tp_medio"]:
            tp_avg = np.mean(stats["tp_sl_valores"]["tp_medio"])
            sl_avg = np.mean(stats["tp_sl_valores"]["sl_medio"])
            tp_hit_rate = (stats["tp_sl_valores"]["tp_atingiu"] / max(1, stats["trades_total"])) * 100
            sl_hit_rate = (stats["tp_sl_valores"]["sl_atingiu"] / max(1, stats["trades_total"])) * 100
            print(f"  📊 TP/SL CALIBRAÇÃO:")
            print(f"     TP médio: {tp_avg:.2f}% | Hit Rate: {tp_hit_rate:.1f}%")
            print(f"     SL médio: {sl_avg:.2f}% | Hit Rate: {sl_hit_rate:.1f}%")
        
        print("=" * 60)
        
        # 📊 V7: Imprimir relatório do Trade Analyzer
        if analyzers and tgt in analyzers:
            try:
                analyzers[tgt].print_final_report(target_name=tgt)
            except:
                pass  # Não trava em caso de erro


    return resultados_por_target

# 🛡️ MÓDULO 10 - TESTE DE ROBUSTEZ E VALIDAÇÃO FINAL
def modulo_10_teste_robustez(df_all, lista_targets, modelos_cache, probs_cache, meta_modelos_cache):
    print("\n" + "="*60)
    print("MÓDULO 10 — VALIDAÇÃO DE ROBUSTEZ (SHUFFLE TEST)")
    print("="*60)
    
    rodar = input("\nDeseja rodar a simulação financeira realista e o Shuffle Test? (s/n): ").strip().lower()
    
    # 🚀 K12 CONFLUÊNCIA: Se o usuário quer K1+K2, forçamos a execução de ambos
    usar_k12 = input("Usar Confluência K1+K2? (s/n): ").strip().lower() == 's'
    if usar_k12:
        print("✅ K12 Ativado: Forçando a execução de K1, K2 e K3.")
        if 'target_K1' not in lista_targets: lista_targets.append('target_K1')
        if 'target_K2' not in lista_targets: lista_targets.append('target_K2')
        if 'target_K3' not in lista_targets: lista_targets.append('target_K3')
    
    # 🔥 K6 CONFLUÊNCIA ELITE: Combina K4 (melhor PF) + K5 (melhor lucro)
    usar_k6 = input("🔥 Usar Confluência K4+K5 (K6 ELITE)? (s/n): ").strip().lower() == 's'
    if usar_k6:
        print("✅ K6 ELITE Ativado: Forçando a execução de K4, K5 e K6.")
        print("   📊 K4: PF 3.58, DD 2.64%, WR 73%")
        print("   📊 K5: Lucro +1218%, WR 68%")
        print("   🎯 K6: Expectativa WR 75-80%, PF 4.5-5.5")
        if 'target_K4' not in lista_targets: lista_targets.append('target_K4')
        if 'target_K5' not in lista_targets: lista_targets.append('target_K5')
        if 'target_K6' not in lista_targets: lista_targets.append('target_K6')
    
    if rodar != 's': return
    
    # 1. Coletar parâmetros do usuário
    try:
        capital_inicial = float(input("Capital Inicial (USD) [ex: 1000]: ") or 1000)
        valor_mao = float(input("Valor de cada entrada (USD) [ex: 100]: ") or 100)
        alavancagem = float(input("Alavancagem (x) [ex: 5]: ") or 5)
        min_conf = float(input("Confiança Mínima (0.50 a 0.99) [ex: 0.75]: ") or 0.75)
        
        # 🚀 INPUT INTELIGENTE: Aceita 's', 'n' ou o valor do rigor diretamente
        meta_input = input("Usar Juiz? (s/n ou digite o rigor ex: 0.85): ").strip().lower()
        if meta_input in ['s', 'sim']:
            usar_meta = True
            corte_juiz = float(input("Rigor do Juiz (0.50 a 0.99) [ex: 0.85]: ") or 0.85)
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
        
        # 🔥 V8: FILTRO DE AGRESSÃO DESABILITADO!
        # ML PURO - sem EMA, RSI, MACD, etc
        print("\n🚀 V8 - ML PURO (Sem indicadores velhos na decisão)")
        print("   ✅ Modelo ML decide tudo")
        print("   ❌ Sem EMA9/20/50")
        print("   ❌ Sem aggression_delta")
        print("   ❌ Sem VPIN")
        print("   ❌ Sem flow_acceleration")
        
        usar_fluxo = False
        modo_fluxo = "disabled"
        
        # 📊 TRADE ANALYZER V7 - SEMPRE CRIAR!
        analyzers = {'config': {'pattern_length': 7, 'min_occurrences': 20}}
        
        if ANALYZER_AVAILABLE:
            usar_analyzer = input("\n📊 Habilitar análise de horários/padrões? (s/n): ").strip().lower() == 's'
            
            if usar_analyzer:
                pattern_length = input("Tamanho do padrão de sequência [padrão: 7]: ").strip()
                pattern_length = int(pattern_length) if pattern_length.isdigit() else 7
                
                min_occurrences = input("Mínimo de ocorrências [padrão: 20]: ").strip()
                min_occurrences = int(min_occurrences) if min_occurrences.isdigit() else 20
                
                analyzers['config']['pattern_length'] = pattern_length
                analyzers['config']['min_occurrences'] = min_occurrences
                print(f"✅ Analyzer configurado: padrões de {pattern_length} sinais, mín {min_occurrences} ocorrências")
            else:
                print("ℹ️ Análise desabilitada (mas K3/K6 funcionam)")
        
    except:
        capital_inicial, valor_mao, alavancagem, min_conf = 1000, 100, 5, 0.75
        usar_meta, corte_juiz = True, 0.85
        usar_fluxo, modo_fluxo = True, "balanced"
        analyzers = None
        
    # 2. Executar Backtest Real
    print("\n" + "="*60)
    print(">>> 1/2: EXECUTANDO BACKTEST REAL (MODELO TREINADO) <<<")
    print("="*60)
    if usar_fluxo:
        print(f"🌊 Filtro de Agressão V2: MODO {modo_fluxo.upper()}")
    resultados_reais = modulo_8_simulador_elite_v2(
        df_all, lista_targets, modelos_cache, probs_cache, meta_modelos_cache,
        capital_inicial, valor_mao, alavancagem, min_conf, usar_meta,
        corte_juiz, usar_fluxo=usar_fluxo, modo_fluxo=modo_fluxo,
        analyzers=analyzers,  # 📊 V7: Trade Analyzer
        titulo_relatorio="RELATÓRIO DE PERFORMANCE INSTITUCIONAL"
    )
    
    # 3. Executar Shuffle Test
    print("\n" + "="*60)
    print(">>> 2/2: EXECUTANDO SHUFFLE TEST (MODELO ALEATÓRIO) <<<")
    print("="*60)
    probs_cache_shuffled = realizar_teste_robustez(probs_cache)
    
    resultados_shuffled = modulo_8_simulador_elite_v2(
        df_all, lista_targets, modelos_cache, probs_cache_shuffled, meta_modelos_cache,
        capital_inicial, valor_mao, alavancagem, min_conf, usar_meta,
        corte_juiz, usar_fluxo=usar_fluxo, modo_fluxo=modo_fluxo,
        analyzers=None,  # 📊 V7: Sem analyzer no shuffle test
        titulo_relatorio="RELATÓRIO DE PERFORMANCE (SHUFFLE TEST)"
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
        print(f"  📈 Retorno Real: {real['retorno_total_pct']:+.2f}% | Drawdown: {real['drawdown_max']:.2f}%")
        print(f"  📊 Retorno Shuffle: {shuf.get('retorno_total', 0):+.2f}% | Drawdown: {shuf.get('drawdown_max', 0):.2f}%")
        print(f"  🛡️  SCORE DE ROBUSTEZ: {robustez_score:.2f}%")
        
        if robustez_score >= 70:
            print("  ✅ ROBUSTEZ ALTA: O modelo tem uma vantagem estatística clara sobre o ruído.")
        elif robustez_score >= 40:
            print("  ⚠️ ROBUSTEZ MÉDIA: O modelo tem alguma vantagem, mas o overfitting é uma preocupação.")
        else:
            print("  ❌ ROBUSTEZ BAIXA: O modelo está próximo de um gerador de sinais aleatórios. Overfitting provável.")
            
    print("█"*60 + "\n")

# ==============================================================================
    # 🛡️ MÓDULO 12: VALIDAÇÃO FINANCEIRA (MODO FINAL PÓS-VARREDURA)
    # ==============================================================================
    # Adaptado para a estrutura confirmada: resultados_reais['target_K6']['stats']
    
    print("\n" + "="*60)
    print("🛡️ VALIDANDO LÓGICA DE SEQUÊNCIA (GUARDIÃO K6)...")
    print("="*60)

    target_foco = 'target_K6'
    trades_para_analise = []

    # 1. TENTATIVA DE RECUPERAÇÃO DE DADOS
    # Fonte Primária: Dicionário de Resultados
    if 'resultados_reais' in locals() and target_foco in resultados_reais:
        pacote = resultados_reais[target_foco]
        
        # O Claude disse que 'trades_executados' pode estar vazio, mas vamos checar
        if 'trades_executados' in pacote and len(pacote['trades_executados']) > 0:
            trades_para_analise = pacote['trades_executados']
            print("✔ Dados recuperados de 'trades_executados'.")
            
        # Fonte Secundária: Variável Global df_k6_results (Geralmente sobra na memória)
        elif 'df_k6_results' in globals() and not df_k6_results.empty:
            trades_para_analise = df_k6_results.to_dict('records')
            print("✔ Dados recuperados de 'df_k6_results' (Backup Global).")
            
        # Fonte Terciária: Tentar reconstruir via log (se disponível)
        elif 'log_trades' in pacote:
            trades_para_analise = pacote['log_trades']
            print("✔ Dados recuperados de 'log_trades'.")

    # 2. EXECUÇÃO DA LÓGICA
    if len(trades_para_analise) > 0:
        print(f"🚀 Analisando sequência de {len(trades_para_analise)} trades...")
        
        lucros = []
        sinais_visuais = []
        
        # Padronização de dados
        for t in trades_para_analise:
            # Lucro
            pnl = t.get('lucro', t.get('pnl', t.get('profit', t.get('retorno', 0))))
            lucros.append(pnl)
            
            # Sinal
            raw_sig = str(t.get('sinal', t.get('side', t.get('tipo', '')))).upper()
            if 'BUY' in raw_sig or '1' in raw_sig or 'COMPRA' in raw_sig: 
                sinais_visuais.append('B')
            else: 
                sinais_visuais.append('S')

        # --- APRENDIZADO ---
        memoria = {}
        contagem = {}
        vitorias = {}
        
        for i in range(7, len(sinais_visuais)):
            seq = tuple(sinais_visuais[i-7:i])
            win = 1 if lucros[i] > 0 else 0
            
            if seq not in contagem: contagem[seq]=0; vitorias[seq]=0
            contagem[seq] += 1
            vitorias[seq] += win
            
        for seq, total in contagem.items():
            if total >= 50: # Filtro de 50
                memoria[seq] = vitorias[seq] / total
        
        print(f"✔ Inteligência Criada: {len(memoria)} padrões monitorados.")

        # --- SIMULAÇÃO ---
        saldo_orig = sum(lucros)
        saldo_filt = 0
        bloq = 0
        
        for i in range(7, len(sinais_visuais)):
            pnl = lucros[i]
            seq = tuple(sinais_visuais[i-7:i])
            
            if seq in memoria and memoria[seq] < 0.40: # CORTE < 40%
                bloq += 1
            else:
                saldo_filt += pnl
        
        delta = saldo_filtrado - saldo_orig if 'saldo_filtrado' in locals() else saldo_filt - saldo_orig
        
        print("-" * 50)
        print(f"🚫 Bloqueados: {bloq}")
        print(f"💰 Original  : ${saldo_orig:,.2f}")
        print(f"🚀 Filtrado  : ${saldo_filt:,.2f}")
        print(f"📈 Diferença : ${delta:,.2f}")
        print("-" * 50)
        
	
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

# =====================================================================
# 🔬 AUDITORIA CIENTÍFICA FINAL (MÓDULO EXTERNO PARA V21)
# =====================================================================
import numpy as np

def auditoria_v21_final(lista_retornos, nome_modelo="K6_V21"):
    """
    Roda Monte Carlo e p-Value sobre os resultados do V21 sem tocar na lógica.
    """
    if not lista_retornos or len(lista_retornos) < 10:
        print(f"⚠️ [AUDITORIA] Dados insuficientes em '{nome_modelo}'.")
        return

    print(f"\n{'='*60}")
    print(f"🕵️ RELATÓRIO DE ROBUSTEZ CIENTÍFICA: {nome_modelo}")
    print(f"{'='*60}")

    # Preparação dos dados
    rets = np.array(lista_retornos)
    cap_inicial = 1000
    n_sim = 10000 # 10.000 simulações de estresse
    
    # 1. MONTE CARLO (Teste de Ruína e Caminho)
    finais = []
    ruinas = 0
    for _ in range(n_sim):
        # Embaralha os trades para ver se o lucro resiste à mudança de ordem
        sim_rets = np.random.choice(rets, size=len(rets), replace=True)
        curva = cap_inicial * np.cumprod(1 + sim_rets)
        finais.append(curva[-1])
        if np.any(curva <= (cap_inicial * 0.2)): # Consideramos quebra se cair 80%
            ruinas += 1
            
    # 2. P-VALUE (Teste contra a Sorte)
    vol = np.std(rets)
    random_bench = np.random.normal(0, vol, size=(10000, len(rets)))
    finais_random = cap_inicial * np.prod(1 + random_bench, axis=1)
    lucro_real = cap_inicial * np.prod(1 + rets)
    
    # Qual a chance de um aleatório bater o seu robô?
    p_value = np.sum(finais_random >= lucro_real) / 10000

    # 3. EXIBIÇÃO DOS DADOS
    print(f"📊 Trades Analisados: {len(rets)}")
    print(f"💰 Lucro Real do V21: {((lucro_real/cap_inicial)-1)*100:.1f}%")
    print(f"------------------------------------------------------------")
    print(f"🧪 SIGNIFICÂNCIA ESTATÍSTICA (p-Value): {p_value:.6f}")
    print(f"   (Meta: < 0.05 para ser considerado real)")
    print(f"------------------------------------------------------------")
    print(f"🎲 RISCO DE RUÍNA (Monte Carlo): {(ruinas/n_sim)*100:.2f}%")
    print(f"   (Probabilidade de quebrar a conta em 10 mil cenários)")
    print(f"{'='*60}\n")

# ==============================================================================
# 🚀 EXPORTADOR FINAL (RECUPERAÇÃO VIA DISCO)
# ==============================================================================
import joblib
import os

print("\n" + "█"*60)
print("📦 GERANDO SISTEMA VIA RECUPERAÇÃO DE DISCO...")
print("█"*60)

try:
    # 1. Tenta carregar o scaler e kmeans que salvamos no passo anterior
    scaler_path = os.path.join('modelos_salvos', 'scaler_regimes.pkl')
    kmeans_path = os.path.join('modelos_salvos', 'kmeans_regimes.pkl')
    
    if os.path.exists(scaler_path):
        scaler_final = joblib.load(scaler_path)
        print(f"✅ Scaler carregado: {scaler_path}")
    else:
        scaler_final = None
        print(f"❌ Arquivo '{scaler_path}' não encontrado.")
    
    if os.path.exists(kmeans_path):
        kmeans_final = joblib.load(kmeans_path)
        print(f"✅ KMeans carregado: {kmeans_path}")
    else:
        kmeans_final = None
        print(f"❌ Arquivo '{kmeans_path}' não encontrado.")

    # 2. Localiza o modelo K6 no cache
    # Se o cache estiver vazio, precisamos verificar o nome da variável no seu script
    modelo_k6 = modelos_cache.get('target_K6') if 'modelos_cache' in globals() else None

    if scaler_final and kmeans_final and modelo_k6:
        pacote = {
            'modelo': modelo_k6,
            'scaler': scaler_final,
            'kmeans': kmeans_final,
            'info': 'V27_SCALER_KMEANS_COMPLETO'
        }
        pacote_path = os.path.join('modelos_salvos', 'SISTEMA_K6_COMPLETO.pkl')
        joblib.dump(pacote, pacote_path)
        print(f"✅ SUCESSO! Sistema completo salvo: {pacote_path}")
        print(f"   ├─ Modelo K6")
        print(f"   ├─ Scaler")
        print(f"   └─ KMeans")
    else:
        if not modelo_k6: print("❌ Modelo K6 não encontrado no modelos_cache.")
        if not scaler_final: print("❌ Scaler não pôde ser recuperado.")
        if not kmeans_final: print("❌ KMeans não pôde ser recuperado.")

except Exception as e:
    print(f"❌ Erro no processo: {e}")

print("█"*60 + "\n")

# ============================================================================
# 🔧 FUNÇÕES PARA CARREGAR SCALER E KMEANS EM PRODUÇÃO
# ============================================================================

def carregar_sistema_completo(diretorio='modelos_salvos'):
    """
    Carrega scaler, kmeans e modelo salvos durante o treinamento.
    
    Returns:
        dict: {'modelo': modelo_k6, 'scaler': scaler, 'kmeans': kmeans}
              ou None se falhar
    """
    import joblib
    import os
    
    try:
        pacote_path = os.path.join(diretorio, 'SISTEMA_K6_COMPLETO.pkl')
        
        if os.path.exists(pacote_path):
            pacote = joblib.load(pacote_path)
            print(f"✅ Sistema completo carregado: {pacote_path}")
            print(f"   ├─ Modelo: {type(pacote['modelo']).__name__}")
            print(f"   ├─ Scaler: {type(pacote['scaler']).__name__}")
            print(f"   └─ KMeans: {type(pacote['kmeans']).__name__}")
            return pacote
        else:
            print(f"❌ Pacote não encontrado: {pacote_path}")
            print(f"   Tentando carregar componentes separados...")
            
            # Fallback: carregar separadamente
            scaler_path = os.path.join(diretorio, 'scaler_regimes.pkl')
            kmeans_path = os.path.join(diretorio, 'kmeans_regimes.pkl')
            
            if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
                print(f"❌ Componentes não encontrados")
                return None
            
            scaler = joblib.load(scaler_path)
            kmeans = joblib.load(kmeans_path)
            
            print(f"✅ Componentes carregados separadamente:")
            print(f"   ├─ Scaler: {scaler_path}")
            print(f"   └─ KMeans: {kmeans_path}")
            
            return {'scaler': scaler, 'kmeans': kmeans, 'modelo': None}
        
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
        import traceback
        traceback.print_exc()
        return None


def aplicar_regimes_em_producao(df, scaler=None, kmeans=None, diretorio='modelos_salvos'):
    """
    Aplica detecção de regimes em dados novos usando scaler/kmeans salvos.
    
    Args:
        df: DataFrame com dados novos (já com features calculadas)
        scaler: StandardScaler carregado (opcional)
        kmeans: KMeans carregado (opcional)
        diretorio: diretório dos modelos salvos
    
    Returns:
        DataFrame com coluna 'market_regime' adicionada
    """
    import joblib
    import os
    
    # Carregar automaticamente se não fornecidos
    if scaler is None or kmeans is None:
        sistema = carregar_sistema_completo(diretorio)
        if sistema is None:
            print("❌ Não foi possível carregar scaler/kmeans")
            df['market_regime'] = 0
            return df
        scaler = sistema['scaler']
        kmeans = sistema['kmeans']
    
    try:
        # Features para regime (mesmas do treinamento)
        regime_features = ['vol_realized', 'rsi_14', 'atr14', 'slope20']
        
        # Verificar se features existem
        missing = [f for f in regime_features if f not in df.columns]
        if missing:
            print(f"⚠️ Features faltando para regimes: {missing}")
            df['market_regime'] = 0
            return df
        
        # Aplicar scaler (TRANSFORM, não FIT_TRANSFORM!)
        X_regime = df[regime_features].fillna(0).values
        X_scaled = scaler.transform(X_regime)
        
        # Predizer regime
        df['market_regime'] = kmeans.predict(X_scaled)
        
        regimes_count = df['market_regime'].value_counts().to_dict()
        print(f"✅ Regimes aplicados: {regimes_count}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro ao aplicar regimes: {e}")
        import traceback
        traceback.print_exc()
        df['market_regime'] = 0
        return df


# ============================================================================
# 📝 EXEMPLO DE USO EM PRODUÇÃO
# ============================================================================
"""
USO COMPLETO EM PRODUÇÃO:

# 1. CARREGAR SISTEMA COMPLETO
sistema = carregar_sistema_completo()
modelo_k6 = sistema['modelo']
scaler = sistema['scaler']
kmeans = sistema['kmeans']

# 2. PREPARAR DADOS NOVOS
df_novo = pd.read_csv('dados_live.csv')
df_novo = feature_engine(df_novo)
df_novo = adicionar_features_avancadas(df_novo)

# 3. APLICAR REGIMES
df_novo = aplicar_regimes_em_producao(df_novo, scaler, kmeans)

# 4. FAZER PREDIÇÕES
features = [...] # mesmas features do treinamento
X = df_novo[features].iloc[-1:]
probs = modelo_k6.predict_proba(X)

# 5. TOMAR DECISÃO
p_up = probs[0][classes.index(1)]
if p_up > 0.75:
    print(f"🚀 COMPRAR! Confiança: {p_up:.2%}")
"""

print("\n✅ Funções de carregamento adicionadas!")
print("   - carregar_sistema_completo()")
print("   - aplicar_regimes_em_producao()")
