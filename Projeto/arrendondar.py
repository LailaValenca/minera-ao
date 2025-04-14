import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('feature_importances.csv')

# Converter as colunas para numéricas (forçando erros para NaN se for necessário)
df['Importance_RF'] = pd.to_numeric(df['Importance_RF'], errors='coerce')
df['Importance_XGB'] = pd.to_numeric(df['Importance_XGB'], errors='coerce')
df['Average_Importance'] = pd.to_numeric(df['Average_Importance'], errors='coerce')

# Definir o fator de escala
fator_escala = 1e12

# Aplicar o fator de escala e arredondar
df['Importance_RF'] = (df['Importance_RF'] / fator_escala).round(2)
df['Importance_XGB'] = (df['Importance_XGB'] / fator_escala).round(2)
df['Average_Importance'] = (df['Average_Importance'] / fator_escala).round(2)

# Verificar rapidamente
print(df)

# Salvar no novo arquivo
df.to_csv('feature_importances_formatado.csv', index=False)
