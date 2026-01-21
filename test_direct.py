import os

path = "/data/PENDLEUSDT_DATA"

print(f"Testando acesso direto a: {path}")
print()

if os.path.exists(path):
    print(f"✅ Diretório EXISTE!")
    print()
    files = os.listdir(path)
    print(f"Arquivos ({len(files)}):")
    for f in sorted(files):
        full = os.path.join(path, f)
        size_mb = os.path.getsize(full) / (1024*1024)
        print(f"  {f} ({size_mb:.2f} MB)")
else:
    print(f"❌ Diretório NÃO EXISTE!")
