#!/usr/bin/env python3
"""
Script para DELETAR completamente o diretório de dados no Render
"""
import os
import shutil

print("="*80)
print("🧹 LIMPEZA TOTAL DO DISCO RENDER")
print("="*80)
print()

# Diretório a deletar
DIR_DATA = "/data/PENDLEUSDT_DATA"
ZIP_FILE = "/data/PENDLEUSDT_COMPLETO.zip"

# Deletar diretório
if os.path.exists(DIR_DATA):
    print(f"⚠️  Diretório encontrado: {DIR_DATA}")
    
    # Listar o que será deletado
    try:
        files = os.listdir(DIR_DATA)
        print(f"    Contém {len(files)} arquivo(s):")
        for f in files:
            full_path = os.path.join(DIR_DATA, f)
            size_mb = os.path.getsize(full_path) / (1024*1024)
            print(f"      - {f} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"    Erro ao listar: {e}")
    
    print()
    print(f"🗑️  DELETANDO DIRETÓRIO COMPLETO...")
    
    try:
        shutil.rmtree(DIR_DATA)
        print(f"✅ DELETADO: {DIR_DATA}")
    except Exception as e:
        print(f"❌ ERRO ao deletar: {e}")
else:
    print(f"✅ Diretório não existe: {DIR_DATA}")

print()

# Deletar ZIP se existir
if os.path.exists(ZIP_FILE):
    size_mb = os.path.getsize(ZIP_FILE) / (1024*1024)
    print(f"⚠️  ZIP encontrado: {ZIP_FILE} ({size_mb:.2f} MB)")
    
    try:
        os.remove(ZIP_FILE)
        print(f"✅ DELETADO: {ZIP_FILE}")
    except Exception as e:
        print(f"❌ ERRO ao deletar: {e}")
else:
    print(f"✅ ZIP não existe: {ZIP_FILE}")

print()
print("="*80)
print("🎉 LIMPEZA CONCLUÍDA!")
print("="*80)
print()
print("Agora você pode rodar o DataManager com disco limpo.")
