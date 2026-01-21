#!/usr/bin/env python3
"""
PROCURAR CSVS - ONDE DIABOS ESTÃO?
"""
import os

print("=" * 80)
print("🔍 PROCURANDO CSVs, PKLs, ZIPs")
print("=" * 80)

# Procurar só arquivos de DADOS
extensions = ['.csv', '.pkl', '.zip', '.json', '.txt']

print("\n📂 Procurando em /opt/render/project/...")

for root, dirs, files in os.walk('/opt/render/project'):
    # Ignorar bibliotecas
    if '.venv' in root or 'site-packages' in root or 'python' in root.lower():
        continue
    
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            full_path = os.path.join(root, file)
            try:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                print(f"\n✅ ENCONTRADO:")
                print(f"   Arquivo: {file}")
                print(f"   Tamanho: {size_mb:.2f} MB")
                print(f"   PATH COMPLETO: {full_path}")
            except:
                pass

print("\n" + "=" * 80)
print("✅ FIM")
print("=" * 80)
