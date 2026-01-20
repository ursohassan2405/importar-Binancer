#!/usr/bin/env python3
"""
AUDITORIA COMPLETA DO DISCO RENDER
Lista TODOS os arquivos gerados e verifica integridade
"""

import os
import json
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("🔍 AUDITORIA COMPLETA DO DISCO RENDER")
print("=" * 80)

# Detectar se está no Render
IS_RENDER = (
    os.environ.get('RENDER') or
    os.environ.get('RENDER_SERVICE_NAME') or
    os.path.exists('/opt/render/project') or
    'render.com' in os.environ.get('HOSTNAME', '')
)

if IS_RENDER:
    BASE_PATH = "/opt/render/project/.data"
    print(f"\n✅ RENDER DETECTADO!")
    print(f"📁 Diretório base: {BASE_PATH}")
else:
    BASE_PATH = "/home/claude"
    print(f"\n⚠️  AMBIENTE LOCAL DETECTADO")
    print(f"📁 Diretório base: {BASE_PATH}")

print("=" * 80)

# Função para formatar tamanho
def format_size(bytes_size):
    """Converte bytes para formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:7.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:7.2f} TB"

# Função para listar arquivos recursivamente
def list_files_recursive(directory):
    """Lista todos os arquivos recursivamente"""
    files_info = []
    
    if not os.path.exists(directory):
        return files_info
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Caminho relativo
                rel_path = os.path.relpath(filepath, directory)
                
                files_info.append({
                    'path': rel_path,
                    'full_path': filepath,
                    'size': size,
                    'size_str': format_size(size),
                    'modified': mtime_str
                })
            except Exception as e:
                print(f"⚠️  Erro ao acessar {filepath}: {e}")
    
    return files_info

# Listar todos os arquivos
print("\n📂 LISTANDO TODOS OS ARQUIVOS...\n")
all_files = list_files_recursive(BASE_PATH)

if not all_files:
    print("❌ NENHUM ARQUIVO ENCONTRADO!")
    print(f"   Verifique se o diretório {BASE_PATH} existe e tem arquivos.")
else:
    # Ordenar por caminho
    all_files.sort(key=lambda x: x['path'])
    
    # Organizar por tipo
    csvs = [f for f in all_files if f['path'].endswith('.csv')]
    pkls = [f for f in all_files if f['path'].endswith('.pkl')]
    jsons = [f for f in all_files if f['path'].endswith('.json')]
    zips = [f for f in all_files if f['path'].endswith('.zip')]
    txts = [f for f in all_files if f['path'].endswith('.txt')]
    outros = [f for f in all_files if not any(f['path'].endswith(ext) for ext in ['.csv', '.pkl', '.json', '.zip', '.txt'])]
    
    # RESUMO GERAL
    print("=" * 80)
    print("📊 RESUMO GERAL")
    print("=" * 80)
    print(f"Total de arquivos: {len(all_files)}")
    print(f"  📄 CSVs:     {len(csvs)}")
    print(f"  📦 PKLs:     {len(pkls)}")
    print(f"  📋 JSONs:    {len(jsons)}")
    print(f"  🗜️  ZIPs:     {len(zips)}")
    print(f"  📝 TXTs:     {len(txts)}")
    print(f"  ❓ Outros:   {len(outros)}")
    
    # Tamanho total
    total_size = sum(f['size'] for f in all_files)
    print(f"\n💾 Tamanho total: {format_size(total_size)}")
    
    # LISTAGEM DETALHADA POR TIPO
    
    # CSVs
    if csvs:
        print("\n" + "=" * 80)
        print("📄 ARQUIVOS CSV")
        print("=" * 80)
        for f in csvs:
            print(f"{f['size_str']:>12} | {f['modified']} | {f['path']}")
    
    # PKLs
    if pkls:
        print("\n" + "=" * 80)
        print("📦 ARQUIVOS PKL (MODELOS)")
        print("=" * 80)
        
        # Separar por tipo de modelo
        lgbm = [f for f in pkls if 'LGBM' in f['path'] or 'lgbm' in f['path']]
        xgb = [f for f in pkls if 'XGB' in f['path'] or 'xgb' in f['path']]
        outros_pkl = [f for f in pkls if f not in lgbm and f not in xgb]
        
        if lgbm:
            print(f"\n🌲 LGBM ({len(lgbm)} modelos):")
            for f in lgbm:
                print(f"  {f['size_str']:>12} | {f['modified']} | {f['path']}")
        
        if xgb:
            print(f"\n🌳 XGB ({len(xgb)} modelos):")
            for f in xgb:
                print(f"  {f['size_str']:>12} | {f['modified']} | {f['path']}")
        
        if outros_pkl:
            print(f"\n🔧 Outros PKLs ({len(outros_pkl)}):")
            for f in outros_pkl:
                print(f"  {f['size_str']:>12} | {f['modified']} | {f['path']}")
    
    # JSONs
    if jsons:
        print("\n" + "=" * 80)
        print("📋 ARQUIVOS JSON")
        print("=" * 80)
        for f in jsons:
            print(f"{f['size_str']:>12} | {f['modified']} | {f['path']}")
            
            # Se for manifesto, mostrar conteúdo
            if 'manifesto' in f['path'].lower():
                try:
                    with open(f['full_path'], 'r') as mf:
                        manifesto = json.load(mf)
                    print(f"  └─ Targets: {len(manifesto)}")
                    if manifesto:
                        first_target = list(manifesto.keys())[0]
                        n_features = manifesto[first_target].get('n_features', 'N/A')
                        print(f"  └─ Features: {n_features}")
                except Exception as e:
                    print(f"  └─ ⚠️  Erro ao ler: {e}")
    
    # ZIPs
    if zips:
        print("\n" + "=" * 80)
        print("🗜️  ARQUIVOS ZIP")
        print("=" * 80)
        for f in zips:
            print(f"{f['size_str']:>12} | {f['modified']} | {f['path']}")
    
    # TXTs
    if txts:
        print("\n" + "=" * 80)
        print("📝 ARQUIVOS TXT")
        print("=" * 80)
        for f in txts:
            print(f"{f['size_str']:>12} | {f['modified']} | {f['path']}")
    
    # Outros
    if outros:
        print("\n" + "=" * 80)
        print("❓ OUTROS ARQUIVOS")
        print("=" * 80)
        for f in outros:
            print(f"{f['size_str']:>12} | {f['modified']} | {f['path']}")
    
    # VERIFICAÇÃO DE INTEGRIDADE
    print("\n" + "=" * 80)
    print("✅ VERIFICAÇÃO DE INTEGRIDADE")
    print("=" * 80)
    
    # Arquivos esperados
    esperados = {
        'CSVs': ['15m.csv', '30m.csv', '1h.csv', '4h.csv', '8h.csv', '1d.csv'],
        'PKLs Essenciais': ['scaler_regimes.pkl', 'kmeans_regimes.pkl'],
        'JSONs': ['manifesto.json'],
        'TXTs': ['RELATORIO_FINAL.txt'],
        'ZIPs': ['.zip']  # Pelo menos um ZIP
    }
    
    problemas = []
    
    # Verificar CSVs
    csv_names = [f['path'] for f in csvs]
    for esperado in esperados['CSVs']:
        found = any(esperado in csv for csv in csv_names)
        if found:
            print(f"✅ CSV {esperado}: ENCONTRADO")
        else:
            print(f"❌ CSV {esperado}: FALTANDO")
            problemas.append(f"CSV {esperado} faltando")
    
    # Verificar PKLs essenciais
    pkl_names = [f['path'] for f in pkls]
    for esperado in esperados['PKLs Essenciais']:
        found = any(esperado in pkl for pkl in pkl_names)
        if found:
            print(f"✅ PKL {esperado}: ENCONTRADO")
        else:
            print(f"❌ PKL {esperado}: FALTANDO")
            problemas.append(f"PKL {esperado} faltando")
    
    # Verificar modelos (deve ter pelo menos 9 LGBM ou XGB)
    if len(pkls) >= 11:  # 9 modelos + 2 essenciais
        print(f"✅ Modelos: {len(pkls)-2} modelos encontrados (esperado ≥9)")
    else:
        print(f"⚠️  Modelos: {len(pkls)-2} modelos encontrados (esperado ≥9)")
        problemas.append(f"Faltam modelos (tem {len(pkls)-2}, esperado ≥9)")
    
    # Verificar manifesto
    if jsons:
        manifesto_found = any('manifesto' in j['path'].lower() for j in jsons)
        if manifesto_found:
            print(f"✅ Manifesto: ENCONTRADO")
        else:
            print(f"❌ Manifesto: FALTANDO")
            problemas.append("Manifesto faltando")
    else:
        print(f"❌ Manifesto: FALTANDO")
        problemas.append("Manifesto faltando")
    
    # Verificar ZIP
    if zips:
        print(f"✅ ZIP: ENCONTRADO ({len(zips)} arquivo(s))")
    else:
        print(f"⚠️  ZIP: NÃO ENCONTRADO")
        problemas.append("ZIP não encontrado")
    
    # Resultado final
    print("\n" + "=" * 80)
    if not problemas:
        print("🎉 TUDO OK! TODOS OS ARQUIVOS ESPERADOS FORAM ENCONTRADOS!")
    else:
        print("⚠️  PROBLEMAS ENCONTRADOS:")
        for p in problemas:
            print(f"   - {p}")
    print("=" * 80)

print("\n✅ AUDITORIA CONCLUÍDA!")
print("=" * 80)
