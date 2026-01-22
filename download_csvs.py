#!/usr/bin/env python3
"""
DOWNLOAD DE CSVs INDIVIDUAIS DO RENDER
Faz upload de cada CSV para Catbox e retorna links
"""

import os
import requests
from datetime import datetime

print("=" * 80)
print("📥 DOWNLOAD DE CSVs DO RENDER")
print("=" * 80)
print()

# Detectar se está no Render
if os.path.exists("/data"):
    SOURCE_DIR = "/data/PENDLEUSDT_DATA"
    print(f"✅ Render detectado!")
    print(f"📂 Diretório: {SOURCE_DIR}")
else:
    print("❌ Não está no Render!")
    exit(1)

print()

# Verificar se diretório existe
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Diretório não existe: {SOURCE_DIR}")
    exit(1)

# Listar CSVs
files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')])

print(f"📊 CSVs encontrados: {len(files)}")
print()

links = {}

for csv_file in files:
    file_path = os.path.join(SOURCE_DIR, csv_file)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    print(f"📤 Enviando {csv_file} ({size_mb:.2f} MB)...")
    
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://catbox.moe/user/api.php',
                data={'reqtype': 'fileupload'},
                files={'fileToUpload': f},
                timeout=300
            )
        
        if response.status_code == 200:
            link = response.text.strip()
            links[csv_file] = link
            print(f"   ✅ {link}")
        else:
            print(f"   ❌ Erro: {response.status_code}")
            links[csv_file] = "ERRO"
    
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        links[csv_file] = "ERRO"
    
    print()

# Resumo final
print("=" * 80)
print("📋 RESUMO - LINKS PARA DOWNLOAD")
print("=" * 80)
for csv_file, link in links.items():
    print(f"{csv_file}:")
    print(f"  {link}")
    print()
print("=" * 80)

