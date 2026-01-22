#!/usr/bin/env python3
"""
WRAPPER PARA RODAR V27 NO RENDER
Lê TUDO do config JSON

ORDEM CORRETA DOS INPUTS DO V27:
1. csv_path
2. out_dir
3. exp_name
4. horizonte
5. multi_tf (s/n)
6. tfs (só se multi_tf == s)
7. peso_temporal (s/n)
8. modo_peso (só se peso_temporal == s)
9. rodar_simulacao (s/n)
10. k12 (s/n)
11. k6 (s/n)
12. capital_inicial (só se rodar_simulacao == s)
13. valor_entrada
14. alavancagem
15. confianca_minima
16. usar_juiz (s/n ou valor direto como 0.85)
17. analise_padroes (s/n)
18. tamanho_padrao (só se analise_padroes == s)
19. minimo_ocorrencias (só se analise_padroes == s)
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
simbolo = config.get("simbolo", "PENDLEUSDT")
pasta_saida = config.get("pasta_saida", f"/data/{simbolo}_DATA")

# ============================================================
# ORDEM CORRETA DOS INPUTS (CONFORME V27 ESPERA)
# ============================================================
inputs = []

# 1. csv_path
inputs.append(f"{pasta_saida}/{simbolo}_15m.csv")

# 2. out_dir
inputs.append(pasta_saida)

# 3. exp_name
inputs.append(f"RENDER_{os.urandom(4).hex()}")

# 4. horizonte
inputs.append(str(config.get("horizonte", 5)))

# 5. multi_tf (s/n)
multi_tf = config.get("multi_tf", True)
inputs.append("s" if multi_tf else "n")

# 6. tfs (SÓ SE multi_tf == True)
if multi_tf:
    inputs.append(",".join(config.get("tfs", ["30m", "1h", "4h", "8h", "1d"])))

# 7. peso_temporal (s/n)
peso_temporal = config.get("peso_temporal", True)
inputs.append("s" if peso_temporal else "n")

# 8. modo_peso (SÓ SE peso_temporal == True)
if peso_temporal:
    inputs.append(str(config.get("modo_peso", 1)))

# 9. rodar_simulacao (s/n) - VINHA ERRADO NA POSIÇÃO 11!
rodar_simulacao = config.get("rodar_simulacao", True)
inputs.append("s" if rodar_simulacao else "n")

# 10. k12 (s/n)
inputs.append("s" if config.get("k12", True) else "n")

# 11. k6 (s/n)
inputs.append("s" if config.get("k6", True) else "n")

# 12-19. Parâmetros da simulação (SÓ SE rodar_simulacao == True)
if rodar_simulacao:
    # 12. capital_inicial
    inputs.append(str(config.get("capital_inicial", 1000)))
    
    # 13. valor_entrada
    inputs.append(str(config.get("valor_entrada", 100)))
    
    # 14. alavancagem
    inputs.append(str(config.get("alavancagem", 10)))
    
    # 15. confianca_minima
    inputs.append(str(config.get("confianca_minima", 0.70)))
    
    # 16. usar_juiz (pode ser s/n ou valor direto)
    usar_juiz = config.get("usar_juiz", 0)
    if usar_juiz == 0 or usar_juiz == "n" or usar_juiz == False:
        inputs.append("n")
    elif usar_juiz == "s" or usar_juiz == True:
        inputs.append("s")
        inputs.append(str(config.get("rigor_juiz", 0.85)))  # Input adicional se s
    else:
        # Valor numérico direto (ex: 0.85)
        inputs.append(str(usar_juiz))
    
    # 17. analise_padroes (s/n)
    analise_padroes = config.get("analise_padroes", True)
    inputs.append("s" if analise_padroes else "n")
    
    # 18-19. Parâmetros do analyzer (SÓ SE analise_padroes == True)
    if analise_padroes:
        inputs.append(str(config.get("tamanho_padrao", 7)))
        inputs.append(str(config.get("minimo_ocorrencias", 20)))

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
