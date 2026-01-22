#!/usr/bin/env python3
"""
WRAPPER PARA RODAR V27 NO RENDER (VERSÃO CORRIGIDA - ANTI-LOOPING)
"""

import os
import sys
import json
import subprocess

# 1. Detectar Ambiente e Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOME_SCRIPT_ALVO = "V27_COM_REVERSAO_CORRIGIDO.py"

# Lista de caminhos possíveis para o script principal
possiveis_caminhos = [
    os.path.join(BASE_DIR, NOME_SCRIPT_ALVO),
    os.path.join(BASE_DIR, "src", NOME_SCRIPT_ALVO),
    os.path.join("/opt/render/project/src", NOME_SCRIPT_ALVO)
]

script_final = None
for caminho in possiveis_caminhos:
    if os.path.exists(caminho):
        script_final = caminho
        break

if not script_final:
    print(f"❌ ERRO CRÍTICO: Não encontrei {NOME_SCRIPT_ALVO} em nenhuma dessas pastas:")
    for p in possiveis_caminhos: print(f"   -> {p}")
    sys.exit(1)

# 2. Carregar Configuração JSON
CONFIG_PATHS = [
    "/data/config_v27_render.json",
    os.path.join(BASE_DIR, "config_v27_render.json"),
    os.path.join(BASE_DIR, "src", "config_v27_render.json")
]

config_path = None
for path in CONFIG_PATHS:
    if os.path.exists(path):
        config_path = path
        break

if not config_path:
    print("❌ Config JSON não encontrado!")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

print("=" * 80)
print("🌐 RENDER: Wrapper Ativo")
print(f"   Script Alvo: {script_final}")
print(f"   Config:      {config_path}")
print("=" * 80)

# 3. Montar a lista de inputs (Ordem exata do V27)
simbolo = config.get("simbolo", "NEARUSDT")
pasta_saida = config.get("pasta_saida", f"/data/{simbolo}_DATA")

inputs = [
    f"{pasta_saida}/{simbolo}_15m.csv",      # 1. csv_path
    pasta_saida,                             # 2. out_dir
    f"RENDER_{os.urandom(4).hex()}",        # 3. exp_name
    str(config.get("horizonte", 5)),         # 4. horizonte
    "s" if config.get("multi_tf", True) else "n" # 5. multi_tf
]

if config.get("multi_tf", True):
    inputs.append(",".join(config.get("tfs", ["30m", "1h", "4h", "8h", "1d"]))) # 6. tfs

inputs.append("s" if config.get("peso_temporal", True) else "n") # 7. peso_temporal

if config.get("peso_temporal", True):
    inputs.append(str(config.get("modo_peso", 1))) # 8. modo_peso

rodar_sim = config.get("rodar_simulacao", True)
inputs.append("s" if rodar_sim else "n") # 9. rodar_simulacao
inputs.append("s" if config.get("k12", True) else "n") # 10. k12
inputs.append("s" if config.get("k6", True) else "n") # 11. k6

if rodar_sim:
    inputs.extend([
        str(config.get("capital_inicial", 1000)), # 12
        str(config.get("valor_entrada", 100)),   # 13
        str(config.get("alavancagem", 10)),      # 14
        str(config.get("confianca_minima", 0.7)),# 15
    ])
    
    usar_juiz = config.get("usar_juiz", 0)
    if usar_juiz in [0, "n", False]:
        inputs.append("n") # 16
    else:
        inputs.append("s") # 16
        inputs.append(str(usar_juiz)) # Sub-input se s
    
    analise_p = config.get("analise_padroes", True)
    inputs.append("s" if analise_p else "n") # 17
    
    if analise_p:
        inputs.append(str(config.get("tamanho_padrao", 7))) # 18
        inputs.append(str(config.get("minimo_ocorrencias", 20))) # 19

input_string = "\n".join(inputs) + "\n"

# 4. Execução com Stream de Log (Evita Looping e Timeout)
print(f"🚀 Iniciando V27 para {simbolo}...\n")

try:
    process = subprocess.Popen(
        ["python3", "-u", script_final], # -u força logs em tempo real
        stdin=subprocess.PIPE,
        stdout=sys.stdout, # Envia direto para o console do Render
        stderr=sys.stderr,
        text=True
    )
    
    # Enviar inputs e aguardar
    process.communicate(input=input_string)
    sys.exit(process.returncode)

except Exception as e:
    print(f"❌ ERRO NO WRAPPER: {e}")
    sys.exit(1)