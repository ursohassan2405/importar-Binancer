#!/usr/bin/env python3
"""
PROCURAR ARQUIVOS PENDLEUSDT ESPECIFICAMENTE
"""
import os
import subprocess

print("=" * 80)
print("🔍 PROCURANDO ARQUIVOS PENDLEUSDT")
print("=" * 80)

# Usar comando find do Linux (mais rápido)
print("\n📂 Procurando com 'find'...\n")

try:
    # Procurar arquivos com PENDLE no nome
    result = subprocess.run(
        ['find', '/opt/render/project', '-name', '*PENDLE*', '-type', 'f'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.stdout:
        files = result.stdout.strip().split('\n')
        print(f"✅ ENCONTRADOS {len(files)} ARQUIVOS:\n")
        for f in files:
            try:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                print(f"📄 {os.path.basename(f)} ({size_mb:.2f} MB)")
                print(f"   PATH: {f}\n")
            except:
                print(f"📄 {f}\n")
    else:
        print("❌ NENHUM ARQUIVO PENDLEUSDT ENCONTRADO!")
        
except Exception as e:
    print(f"❌ Erro: {e}")

# Também procurar CSVs genéricos
print("\n" + "=" * 80)
print("🔍 PROCURANDO QUALQUER CSV...")
print("=" * 80 + "\n")

try:
    result = subprocess.run(
        ['find', '/opt/render/project', '-name', '*.csv', '-type', 'f'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.stdout:
        files = result.stdout.strip().split('\n')
        # Filtrar bibliotecas
        files = [f for f in files if '.venv' not in f and 'site-packages' not in f and 'node_modules' not in f]
        
        if files:
            print(f"✅ ENCONTRADOS {len(files)} CSVs:\n")
            for f in files:
                try:
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    print(f"📄 {os.path.basename(f)} ({size_mb:.2f} MB)")
                    print(f"   PATH: {f}\n")
                except:
                    pass
        else:
            print("❌ NENHUM CSV ENCONTRADO!")
    else:
        print("❌ NENHUM CSV ENCONTRADO!")
        
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n" + "=" * 80)
print("✅ FIM")
print("=" * 80)
