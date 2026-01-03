import pandas as pd
import os

caminho = "/data/RUNEUSDT_15m.csv"

print("\n" + "="*50)
print("🔍 INSPEÇÃO TÉCNICA DO ARQUIVO NO DISCO")
print("="*50)

if os.path.exists(caminho):
    # Lendo apenas as primeiras e últimas linhas para economizar memória
    df = pd.read_csv(caminho)
    
    print(f"✅ Arquivo Localizado: {caminho}")
    print(f"📊 Total de Linhas: {len(df)}")
    print(f"📅 Início dos Dados: {pd.to_datetime(df['ts'].min(), unit='ms')}")
    print(f"📅 Fim dos Dados:    {pd.to_datetime(df['ts'].max(), unit='ms')}")
    
    print("\n🛡️ VERIFICAÇÃO DE COLUNAS (INTELIGÊNCIA):")
    colunas = df.columns.tolist()
    print(f"Colunas encontradas: {colunas}")
    
    # Verificando se os dados de agressão/baleias estão preenchidos
    if 'buy_vol' in df.columns or 'delta' in df.columns:
        # Pega uma amostra onde o delta não é zero
        amostra_agressao = df[df['delta'] != 0].head(3)
        if not amostra_agressao.empty:
            print("\n✅ DADOS DE AGRESSÃO DETECTADOS!")
            print("Amostra de Delta (Agressão Líquida):")
            print(amostra_agressao[['ts', 'close', 'delta']].to_string(index=False))
        else:
            print("\n⚠️ AVISO: Colunas de agressão existem, mas estão zeradas.")
    else:
        print("\n❌ ERRO: Colunas de microestrutura (Delta/Baleias) NÃO encontradas!")

else:
    print(f"❌ ERRO: O arquivo {caminho} não foi encontrado no disco.")

print("="*50 + "\n")
