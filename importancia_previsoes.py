import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

print("Carregando os dados...")
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

# Preparar features (X) e target (y) - excluindo a coluna de países
X = df.drop(['Attack Type', 'Country'], axis=1)
y = df['Attack Type']

# Aplicar Label Encoding para a variável alvo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nClasses codificadas: {list(zip(le.classes_, range(len(le.classes_))))}")

# Identificar colunas categóricas e numéricas - sem incluir 'Country'
categorical_cols = ['Target Industry', 'Attack Source', 
                   'Security Vulnerability Type', 'Defense Mechanism Used']
numerical_cols = ['Year', 'Financial Loss (in Million $)', 
                 'Number of Affected Users', 'Incident Resolution Time (in Hours)']

# Criar preprocessador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Transformar os dados para extrair as features após one-hot encoding
X_preprocessed = preprocessor.fit_transform(X)

# Obter os nomes das features após one-hot encoding
one_hot_encoder = preprocessor.named_transformers_['cat']
cat_features = one_hot_encoder.get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numerical_cols, cat_features])

# Treinar o modelo Random Forest com todos os dados
print("\nTreinando o modelo Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, 
                                 min_samples_split=5, random_state=42)
rf_model.fit(X_preprocessed, y_encoded)

# Treinar o modelo XGBoost
print("\nTreinando o modelo XGBoost...")
xgb_model = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, 
                          subsample=0.8, random_state=42)
xgb_model.fit(X_preprocessed, y_encoded)

# Obter importância das features para ambos os modelos
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_model.feature_importances_,
    'XGBoost_Importance': xgb_model.feature_importances_
})

# Calcular importância média
importances_df['Average_Importance'] = (importances_df['RF_Importance'] + importances_df['XGBoost_Importance']) / 2
importances_df = importances_df.sort_values('Average_Importance', ascending=False)

# Salvar importâncias das features
importances_df.to_csv('dual_model_attack_type_importances.csv', index=False)
print("\nImportâncias das features para ambos os modelos salvas em 'dual_model_attack_type_importances.csv'")

# Classificar as importâncias por modelo
rf_top_features = importances_df.sort_values('RF_Importance', ascending=False)[['Feature', 'RF_Importance']].head(20)
xgb_top_features = importances_df.sort_values('XGBoost_Importance', ascending=False)[['Feature', 'XGBoost_Importance']].head(20)

rf_top_features.to_csv('rf_top_features.csv', index=False)
xgb_top_features.to_csv('xgb_top_features.csv', index=False)
print("Top features de cada modelo salvas em arquivos separados")

# Agrupar importâncias por categoria para cada modelo
categories_rf = {}
categories_xgb = {}

for feature, rf_imp, xgb_imp in zip(feature_names, rf_model.feature_importances_, xgb_model.feature_importances_):
    if feature in numerical_cols:
        category = 'Numerical_' + feature
    else:
        # Extrair a categoria da feature one-hot encoded (formato: categoria_valor)
        parts = feature.split('_')
        if len(parts) >= 2:
            category = parts[0]
        else:
            category = 'Outro'
    
    if category not in categories_rf:
        categories_rf[category] = 0
        categories_xgb[category] = 0
    
    categories_rf[category] += rf_imp
    categories_xgb[category] += xgb_imp

# Criar DataFrame com importância por categoria para ambos os modelos
category_df = pd.DataFrame({
    'Category': list(categories_rf.keys()),
    'RF_Importance': list(categories_rf.values()),
    'XGBoost_Importance': list(categories_xgb.values())
})
category_df['Average_Importance'] = (category_df['RF_Importance'] + category_df['XGBoost_Importance']) / 2
category_df = category_df.sort_values('Average_Importance', ascending=False)

# Salvar importâncias por categoria
category_df.to_csv('dual_model_category_importances.csv', index=False)
print("Importâncias agregadas por categoria para ambos os modelos salvas em 'dual_model_category_importances.csv'")

# Análise por tipo de ataque para ambos os modelos
attack_types = le.classes_
attack_type_importances_rf = {}
attack_type_importances_xgb = {}

print("\nAnalisando a importância das features para cada tipo de ataque em ambos os modelos...")

for i, attack_type in enumerate(attack_types):
    # Criar target binário para este tipo de ataque (1 para este tipo, 0 para outros)
    y_binary = np.zeros_like(y_encoded)
    y_binary[y_encoded == i] = 1
    
    # Treinar modelos para este tipo de ataque
    rf_model_binary = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_binary.fit(X_preprocessed, y_binary)
    
    xgb_model_binary = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model_binary.fit(X_preprocessed, y_binary)
    
    # Armazenar importâncias para este tipo de ataque
    attack_type_importances_rf[attack_type] = rf_model_binary.feature_importances_
    attack_type_importances_xgb[attack_type] = xgb_model_binary.feature_importances_

# Criar DataFrames para cada tipo de ataque
for attack_type in attack_types:
    # Criar DataFrame para este tipo de ataque
    attack_df = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': attack_type_importances_rf[attack_type],
        'XGBoost_Importance': attack_type_importances_xgb[attack_type]
    })
    attack_df['Average_Importance'] = (attack_df['RF_Importance'] + attack_df['XGBoost_Importance']) / 2
    attack_df = attack_df.sort_values('Average_Importance', ascending=False)
    
    # Salvar para este tipo de ataque
    safe_name = attack_type.replace('/', '_').replace(' ', '_').lower()
    attack_df.to_csv(f'dual_model_{safe_name}_importances.csv', index=False)

print(f"Importâncias específicas para cada tipo de ataque salvas em arquivos separados")

# Criar um arquivo resumo com as top 10 features para cada tipo de ataque em ambos os modelos
summary_rows = []

for attack_type in attack_types:
    # Criar DataFrame para este tipo de ataque
    attack_df = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': attack_type_importances_rf[attack_type],
        'XGBoost_Importance': attack_type_importances_xgb[attack_type]
    })
    attack_df['Average_Importance'] = (attack_df['RF_Importance'] + attack_df['XGBoost_Importance']) / 2
    
    # Top 10 features por importância média
    top_features = attack_df.sort_values('Average_Importance', ascending=False).head(10)
    
    for index, row in top_features.iterrows():
        summary_rows.append({
            'Attack_Type': attack_type,
            'Feature': row['Feature'],
            'RF_Importance': row['RF_Importance'],
            'XGBoost_Importance': row['XGBoost_Importance'],
            'Average_Importance': row['Average_Importance']
        })

# Criar DataFrame com o resumo
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('dual_model_top_features_summary.csv', index=False)
print("Resumo das top features por tipo de ataque salvo em 'dual_model_top_features_summary.csv'")

print("\nTodos os arquivos CSV foram gerados com sucesso!")