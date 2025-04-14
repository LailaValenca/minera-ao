import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Carregar os dados
print("Carregando os dados...")
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

# Verificar a distribuição da variável alvo
print("\nDistribuição da variável alvo (Attack Type):")
target_counts = df['Attack Type'].value_counts()
print(target_counts)

# Configurar X e y
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

# Definir pipelines para cada modelo
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         min_samples_split=5, random_state=42))
])

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, 
                               subsample=0.8, random_state=42))
])

# Definir métricas para validação cruzada
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}

# Configurar validação cruzada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definir modelos
models = {
    'Random Forest': rf_pipeline,
    'XGBoost': xgb_pipeline
}

# Executar validação cruzada para cada modelo
print("\nIniciando experimentos com validação cruzada de 10 folds...")
results = {}

for name, model in models.items():
    print(f"\nExecutando {name}...")
    cv_results = cross_validate(model, X, y_encoded, cv=cv, scoring=scoring, n_jobs=-1)
    
    results[name] = {
        'Acurácia': cv_results['test_accuracy'].mean(),
        'Precisão (Macro)': cv_results['test_precision_macro'].mean(),
        'Recall (Macro)': cv_results['test_recall_macro'].mean(),
        'F1-Measure (Macro)': cv_results['test_f1_macro'].mean()
    }

# Apresentar resultados da validação cruzada
print("\n========== RESULTADOS DA VALIDAÇÃO CRUZADA ==========")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Criar DataFrame para facilitar comparação
df_results = pd.DataFrame(results)
print("\nComparação entre os modelos:")
print(df_results)

# Dividir dados em treino e teste para avaliação final
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Avaliar modelos no conjunto de teste
test_results = {}
print("\n========== AVALIAÇÃO NO CONJUNTO DE TESTE ==========")

for name, model in models.items():
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    test_results[name] = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão (Macro)': precision_score(y_test, y_pred, average='macro'),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro'),
        'F1-Measure (Macro)': f1_score(y_test, y_pred, average='macro')
    }
    
    # Mostrar relatório de classificação
    print(f"\n{name} - Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# Apresentar resultados da avaliação no conjunto de teste
print("\n========== RESUMO DOS RESULTADOS DO CONJUNTO DE TESTE ==========")
for model_name, metrics in test_results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Criar DataFrame para facilitar comparação
df_test_results = pd.DataFrame(test_results)
print("\nComparação entre os modelos (conjunto de teste):")
print(df_test_results)

# Salvar resultados
df_results.to_csv('resultados_validacao_cruzada.csv')
df_test_results.to_csv('resultados_teste_simplificado.csv') 