import os
print("\n" + "="*40)
print("📂 EXPLORADOR DE DISCO RENDER")
if os.path.exists("/data"):
    arquivos = os.listdir("/data")
    for arq in arquivos:
        tamanho = os.path.getsize(f"/data/{arq}") / (1024*1024)
        print(f"📄 Arquivo: {arq} | Tamanho: {tamanho:.2f} MB")
else:
    print("❌ Erro: Pasta /data não encontrada.")
print("="*40 + "\n")
