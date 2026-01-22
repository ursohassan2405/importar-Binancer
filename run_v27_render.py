#!/usr/bin/env python3
"""
WRAPPER PARA RODAR V27 NO RENDER
Lê TUDO do config JSON
"""

import os
import sys
import json
import subprocess

# Detectar Render
if not os.path.exists("/opt/render/project"):
    print("❌ Não está no Render!")
    sys.exit(1)

# Carregar config
CONFIG_PATHS = [
    "/data/config_v27_render.json",
    "/opt/render/project/src/config_v27_render.json",
    "./config_v27_render.json",
]

CONFIG_PATH = None
for path in CONFIG_PATHS:
    if os.path.exists(path):
        CONFIG_PATH = path
        break

if not CONFIG_PATH:
    print("❌ Config não encontrado!")
    sys.exit(1)

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("🌐 RENDER: Rodando V27 com config automática")
print(f"   Config: {CONFIG_PATH}")
print("=" * 80)
print()

# Preparar inputs DIRETO do config
simbolo = config.get("simbolo", "PENDLEUSDT")  # NEARUSDT do seu config
pasta_saida = config.get("pasta_saida", f"/data/{simbolo}_DATA")

inputs = []
inputs.append(f"{pasta_saida}/{simbolo}_15m.csv")  # csv_path
inputs.append(pasta_saida)  # out_dir
inputs.append(f"RENDER_{os.urandom(4).hex()}")  # exp_name
inputs.append(str(config.get("horizonte", 5)))  # horizonte
inputs.append("s" if config.get("multi_tf", True) else "n")  # multi_tf
inputs.append(",".join(config.get("tfs", ["30m","1h","4h","8h","1d"])))  # tfs
inputs.append("s" if config.get("peso_temporal", True) else "n")  # peso_temporal
inputs.append(str(config.get("modo_peso", 1)))  # modo_peso
inputs.append("s" if config.get("k12", True) else "n")  # k12
inputs.append("s" if config.get("k6", True) else "n")  # k6
inputs.append("s" if config.get("rodar_simulacao", True) else "n")  # simulacao
inputs.append("s" if config.get("rodar_shuffle", True) else "n")  # shuffle
inputs.append(str(config.get("capital_inicial", 1000)))  # capital
inputs.append(str(config.get("valor_entrada", 100)))  # valor_entrada
inputs.append(str(config.get("alavancagem", 10)))  # alavancagem
inputs.append(str(config.get("confianca_minima", 0.70)))  # confianca
inputs.append(str(config.get("usar_juiz", 0)))  # juiz
inputs.append("s" if config.get("analise_padroes", True) else "n")  # padroes
inputs.append(str(config.get("tamanho_padrao", 7)))  # tamanho
inputs.append(str(config.get("minimo_ocorrencias", 20)))  # ocorrencias

# Juntar tudo
input_string = "\n".join(inputs) + "\n"

# DEBUG: Mostrar inputs que serão passados
print("🔍 DEBUG - INPUTS QUE SERÃO PASSADOS:")
print("=" * 80)
for i, inp in enumerate(inputs, 1):
    print(f"{i:2d}. {inp}")
print("=" * 80)
print()

# Rodar V27
print(f"🚀 Iniciando V27 com {simbolo}...\n")

try:
    process = subprocess.Popen(
        ["python3", "/opt/render/project/src/V27_COM_REVERSAO_CORRIGIDO.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    stdout, _ = process.communicate(input=input_string, timeout=3600)
    print(stdout)
    sys.exit(process.returncode)
    
except subprocess.TimeoutExpired:
    print("❌ TIMEOUT!")
    process.kill()
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

