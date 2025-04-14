import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Carregar os dados
print("Carregando os dados...")
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

# Preparar features (X) e target (y)
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Aplicar Label Encoding para a variável alvo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses codificadas: {list(zip(le.classes_, range(len(le.classes_))))}")

# Identificar colunas categóricas e numéricas
categorical_cols = ['Country', 'Target Industry', 'Attack Source', 
                   'Security Vulnerability Type', 'Defense Mechanism Used']
numerical_cols = ['Year', 'Financial Loss (in Million $)', 
                 'Number of Affected Users', 'Incident Resolution Time (in Hours)']

# Criar preprocessador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Criar pipeline com o modelo vencedor (Random Forest)
modelo = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         min_samples_split=5, random_state=42))
])

# Treinar o modelo com todos os dados
print("\nTreinando o modelo Random Forest com todos os dados...")
modelo.fit(X, y_encoded)
print("Modelo treinado com sucesso!")

# Função para simular um ataque e fazer a previsão
def classificar_ataque(dados):
    # Criar um DataFrame com os dados do ataque
    ataque_df = pd.DataFrame([dados])
    
    # Verificar se o DataFrame tem as colunas corretas
    for col in X.columns:
        if col not in ataque_df.columns:
            raise ValueError(f"Coluna '{col}' está faltando nos dados de entrada")
    
    # Fazer a previsão
    classe_prevista = modelo.predict(ataque_df)[0]
    probabilidades = modelo.predict_proba(ataque_df)[0]
    
    # Obter o nome da classe prevista
    nome_classe = le.inverse_transform([classe_prevista])[0]
    
    # Obter as probabilidades para cada classe
    prob_dict = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilidades)}
    
    return nome_classe, prob_dict

# Demonstração: Classificar alguns exemplos de ataques
print("\n========== ORÁCULO DE CLASSIFICAÇÃO DE ATAQUES ==========")

# Exemplo 1 - Um possível ataque de ransomware
exemplo1 = {
    'Country': 'USA',
    'Year': 2023,
    'Target Industry': 'Healthcare',
    'Financial Loss (in Million $)': 75.5,
    'Number of Affected Users': 450000,
    'Attack Source': 'Hacker Group',
    'Security Vulnerability Type': 'Unpatched Software',
    'Defense Mechanism Used': 'Firewall',
    'Incident Resolution Time (in Hours)': 48
}

# Exemplo 2 - Um possível ataque DDoS
exemplo2 = {
    'Country': 'Germany',
    'Year': 2024,
    'Target Industry': 'Banking',
    'Financial Loss (in Million $)': 60.2,
    'Number of Affected Users': 850000,
    'Attack Source': 'Nation-state',
    'Security Vulnerability Type': 'Zero-day',
    'Defense Mechanism Used': 'AI-based Detection',
    'Incident Resolution Time (in Hours)': 15
}

# Exemplo 3 - Um possível ataque de phishing
exemplo3 = {
    'Country': 'UK',
    'Year': 2022,
    'Target Industry': 'Retail',
    'Financial Loss (in Million $)': 25.8,
    'Number of Affected Users': 320000,
    'Attack Source': 'Unknown',
    'Security Vulnerability Type': 'Social Engineering',
    'Defense Mechanism Used': 'VPN',
    'Incident Resolution Time (in Hours)': 22
}

# Classificar os exemplos
for i, exemplo in enumerate([exemplo1, exemplo2, exemplo3], 1):
    try:
        print(f"\nExemplo {i}:")
        for k, v in exemplo.items():
            print(f"  {k}: {v}")
        
        tipo_ataque, probabilidades = classificar_ataque(exemplo)
        
        print(f"\n  => Tipo de ataque previsto: {tipo_ataque}")
        print("  => Probabilidades para cada tipo de ataque:")
        for tipo, prob in sorted(probabilidades.items(), key=lambda x: x[1], reverse=True):
            print(f"     {tipo}: {prob:.4f} ({prob*100:.2f}%)")
    
    except Exception as e:
        print(f"Erro ao classificar o exemplo {i}: {e}")

print("\n==========================================================")
print("Nota: Este oráculo usa o modelo Random Forest treinado com todos os dados do dataset.")
print("      As predições são baseadas nos padrões aprendidos a partir dos incidentes históricos.")