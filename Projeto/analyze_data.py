import pandas as pd
import numpy as np

# Carregar o dataset
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

# Informações gerais do dataset
print('INFORMAÇÕES GERAIS DO DATASET:')
print(f'Total de registros: {len(df)}')
print(f'Colunas do dataset: {df.columns.tolist()}')

# Estatísticas descritivas para colunas numéricas
print('\nESTATÍSTICAS DESCRITIVAS - COLUNAS NUMÉRICAS:')
numeric_cols = ['Year', 'Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
print(df[numeric_cols].describe())

# Valor máximo, mínimo, média e mediana para colunas numéricas específicas
print('\nINFORMAÇÕES ADICIONAIS:')
for col in ['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']:
    print(f'\n{col}:')
    print(f'Valor máximo: {df[col].max()}')
    print(f'Valor mínimo: {df[col].min()}')
    print(f'Média: {df[col].mean()}')
    print(f'Mediana: {df[col].median()}')

# Identificação de outliers para colunas numéricas
print('\nOUTLIERS (MÉTODO IQR):')
for col in ['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'\nOutliers em {col}: {len(outliers)}')
    if len(outliers) > 0:
        print(f'Limite inferior: {lower_bound}')
        print(f'Limite superior: {upper_bound}')

# Totalizações
print('\nTOTALIZAÇÕES:')
print(f'Perda financeira total: ${df["Financial Loss (in Million $)"].sum()} milhões')
print(f'Total de usuários afetados: {df["Number of Affected Users"].sum()}')
print(f'Tempo total de resolução de incidentes: {df["Incident Resolution Time (in Hours)"].sum()} horas')

# Estatísticas por tipo de ataque (nosso label)
print('\nESTATÍSTICAS POR TIPO DE ATAQUE:')
attack_stats = df.groupby('Attack Type').agg({
    'Financial Loss (in Million $)': ['sum', 'mean', 'median'],
    'Number of Affected Users': ['sum', 'mean', 'median'],
    'Incident Resolution Time (in Hours)': ['sum', 'mean', 'median']
})
print(attack_stats)

# Contagem de tipos de ataque
print('\nCONTAGEM DE TIPOS DE ATAQUE:')
print(df['Attack Type'].value_counts())

# Distribuição de ataques por ano
print('\nDISTRIBUIÇÃO DE ATAQUES POR ANO:')
year_attack_count = df.groupby(['Year', 'Attack Type']).size().unstack()
print(year_attack_count) 