#!/usr/bin/env python3
"""
AUDIT SIMPLES - SÓ LISTA PATHS EXATOS
"""
import os

print("=" * 80)
print("🔍 LISTANDO TODOS OS ARQUIVOS E PATHS EXATOS")
print("=" * 80)

# Procurar em todos os lugares possíveis
dirs_to_check = [
    "/opt/render/project/.data",
    "/opt/render/project/src",
    "/opt/render/project",
    "/opt/render",
]

for base_dir in dirs_to_check:
    if not os.path.exists(base_dir):
        print(f"\n❌ {base_dir} - NÃO EXISTE")
        continue
    
    print(f"\n✅ {base_dir} - EXISTE")
    print("-" * 80)
    
    try:
        for root, dirs, files in os.walk(base_dir):
            # Mostrar diretórios
            if dirs:
                print(f"\n📁 {root}/")
                for d in sorted(dirs):
                    print(f"   📂 {d}/")
            
            # Mostrar arquivos
            if files:
                if not dirs:
                    print(f"\n📁 {root}/")
                for f in sorted(files):
                    full_path = os.path.join(root, f)
                    try:
                        size = os.path.getsize(full_path)
                        size_mb = size / (1024 * 1024)
                        print(f"   📄 {f} ({size_mb:.2f} MB)")
                        print(f"      PATH EXATO: {full_path}")
                    except Exception as e:
                        print(f"   📄 {f} (erro: {e})")
    except Exception as e:
        print(f"   ⚠️ Erro ao listar: {e}")

print("\n" + "=" * 80)
print("✅ FIM DA LISTAGEM")
print("=" * 80)
