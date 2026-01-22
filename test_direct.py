import os
import pandas as pd
from datetime import datetime

path = "/data/NEARUSDT_DATA"

print(f"Testando acesso direto a: {path}")
print()

if os.path.exists(path):
    print(f"✅ Diretório EXISTE!")
    print()
    files = os.listdir(path)
    print(f"Arquivos ({len(files)}):")
    print()
    for f in sorted(files):
        full = os.path.join(path, f)
        size_mb = os.path.getsize(full) / (1024*1024)
        
        # Data de modificação
        mtime = os.path.getmtime(full)
        data_modificacao = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Contar linhas se for CSV
        linhas = "N/A"
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(full)
                linhas = f"{len(df):,} linhas"
                
                # Mostrar período se tiver coluna 'ts'
                if 'ts' in df.columns:
                    try:
                        ts_min = pd.to_datetime(df['ts'].min(), unit='ms')
                        ts_max = pd.to_datetime(df['ts'].max(), unit='ms')
                        periodo = f" | {ts_min.date()} até {ts_max.date()}"
                    except:
                        periodo = ""
                else:
                    periodo = ""
                    
                linhas = linhas + periodo
            except Exception as e:
                linhas = f"ERRO ao ler: {e}"
        
        print(f"  📄 {f}")
        print(f"     ├─ Tamanho: {size_mb:.2f} MB")
        print(f"     ├─ Modificado: {data_modificacao}")
        print(f"     └─ Conteúdo: {linhas}")
        print()
else:
    print(f"❌ Diretório NÃO EXISTE!")
