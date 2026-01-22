#!/usr/bin/env python3
"""
DOWNLOAD DE TODOS OS ARQUIVOS DO RENDER
Busca RECURSIVAMENTE em /data/ e faz ZIP de tudo
"""

import os
import zipfile
from datetime import datetime

print("=" * 80)
print("📥 DOWNLOAD DE TODOS OS ARQUIVOS DO RENDER")
print("=" * 80)
print()

# Detectar se está no Render
if os.path.exists("/data"):
    SOURCE_DIR = "/data"
    print(f"✅ Render detectado!")
    print(f"📂 Buscando em: {SOURCE_DIR}")
else:
    print("❌ Não está no Render!")
    exit(1)

print()

# Buscar TODOS os arquivos recursivamente
all_files = []
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(('.csv', '.pkl', '.json', '.zip')):
            file_path = os.path.join(root, file)
            all_files.append(file_path)

print(f"📊 Arquivos encontrados: {len(all_files)}")
print()

# Agrupar por tipo
csvs = [f for f in all_files if f.endswith('.csv')]
pkls = [f for f in all_files if f.endswith('.pkl')]
jsons = [f for f in all_files if f.endswith('.json')]
zips = [f for f in all_files if f.endswith('.zip')]

print(f"   CSVs: {len(csvs)}")
print(f"   PKLs: {len(pkls)}")
print(f"   JSONs: {len(jsons)}")
print(f"   ZIPs: {len(zips)}")
print()

# Criar ZIP com TUDO
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"RENDER_FULL_BACKUP_{timestamp}.zip"
zip_path = f"/data/{zip_name}"

print(f"📦 Criando ZIP: {zip_name}")
print()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for file_path in all_files:
        # Pular o próprio ZIP que está sendo criado
        if file_path == zip_path:
            continue
            
        # Criar nome relativo ao /data/
        arcname = os.path.relpath(file_path, SOURCE_DIR)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        print(f"   ✅ {arcname} ({size_mb:.2f} MB)")
        z.write(file_path, arcname=arcname)

zip_size = os.path.getsize(zip_path) / (1024 * 1024)
print()
print(f"✅ ZIP criado: {zip_size:.2f} MB")
print(f"📁 Path: {zip_path}")
print()

# Upload para Catbox
print("📤 Enviando para Catbox...")
try:
    import requests
    
    with open(zip_path, 'rb') as f:
        response = requests.post(
            'https://catbox.moe/user/api.php',
            data={'reqtype': 'fileupload'},
            files={'fileToUpload': f},
            timeout=600
        )
    
    if response.status_code == 200:
        link = response.text.strip()
        print()
        print("=" * 80)
        print("🎉 SUCESSO!")
        print("=" * 80)
        print(f"📥 LINK PARA DOWNLOAD:")
        print(link)
        print("=" * 80)
        print()
        print("📋 CONTEÚDO DO ZIP:")
        print(f"   Total de arquivos: {len(all_files)}")
        print(f"   Tamanho: {zip_size:.2f} MB")
        print("=" * 80)
    else:
        print(f"❌ Erro no upload: {response.status_code}")
        print(f"   ZIP salvo em: {zip_path}")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    print(f"   ZIP salvo em: {zip_path}")
    import traceback
    traceback.print_exc()
