#!/usr/bin/env python3
import os
import shutil
from datetime import datetime

# Configurações
BASE_DATA = "/data"
SIMBOLO_ATUAL = "NEARUSDT"
HOJE = datetime.now().strftime("%Y-%m-%d")

print("="*80)
print(f"🧹 FAXINA GERAL NO DISCO RENDER - {datetime.now().strftime('%d/%m/%Y')}")
print("="*80)

if not os.path.exists(BASE_DATA):
    print(f"❌ Diretório {BASE_DATA} não encontrado.")
    exit()

# Listar todas as pastas e ficheiros em /data
itens = os.listdir(BASE_DATA)

for item in itens:
    caminho = os.path.join(BASE_DATA, item)
    
    # 1. Se for uma pasta de outro símbolo (ex: PENDLEUSDT_DATA), apaga TUDO
    if os.path.isdir(caminho) and SIMBOLO_ATUAL not in item:
        try:
            shutil.rmtree(caminho)
            print(f"🗑️  PASTA REMOVIDA (Outro ativo): {item}")
        except Exception as e:
            print(f"⚠️  Erro ao remover pasta {item}: {e}")
            
    # 2. Se for a pasta do símbolo atual (NEAR), faz limpeza seletiva
    elif os.path.isdir(caminho) and SIMBOLO_ATUAL in item:
        print(f"📂 ANALISANDO PASTA ATUAL: {item}")
        for f in os.listdir(caminho):
            arq_path = os.path.join(caminho, f)
            mtime = os.path.getmtime(arq_path)
            data_mod = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            
            # APAGA se for CSV (pesado) ou se NÃO for um PKL de hoje
            eh_pkl_hoje = f.endswith(".pkl") and data_mod == HOJE
            if not eh_pkl_hoje:
                try:
                    os.remove(arq_path)
                    print(f"   🗑️  Removido ficheiro antigo/pesado: {f}")
                except Exception as e:
                    print(f"   ⚠️  Erro ao remover {f}: {e}")
            else:
                print(f"   ✅ PRESERVADO (Modelo de hoje): {f}")

    # 3. Se forem ficheiros soltos em /data (como .zip ou configs antigas)
    elif os.path.isfile(caminho):
        if not item.endswith(".json"): # Preserva ficheiros .json de configuração
            try:
                os.remove(caminho)
                print(f"🗑️  FICHEIRO SOLTO REMOVIDO: {item}")
            except Exception as e:
                print(f"⚠️  Erro ao remover ficheiro {item}: {e}")

print("-" * 80)
print("✨ Limpeza concluída! Disco pronto para novos downloads.")
print("="*80)