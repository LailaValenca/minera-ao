import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Criar pasta para salvar as visualizações
os.makedirs('dashboard_bi', exist_ok=True)

# Configurar estilo das visualizações
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Definir cores para o dashboard
cores_dashboard = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']

# Configurar tamanho padrão das figuras
plt.figure(figsize=(12, 8))

# Carregar os dados
print("Carregando os dados...")
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

# Preparar features (X) e target (y)
X = df.drop('Attack Type', axis=1)
y = df['Attack Type']

# Aplicar Label Encoding para a variável alvo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

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

# Criar pipeline com o modelo Random Forest
modelo = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         min_samples_split=5, random_state=42))
])

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Treinar o modelo
print("Treinando o modelo...")
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)

print("Gerando visualizações para o dashboard...")

# 1. Distribuição das classes no dataset
plt.figure(figsize=(12, 6))
counts = df['Attack Type'].value_counts()
bars = plt.bar(counts.index, counts.values, color=cores_dashboard)
plt.title('Distribuição de Tipos de Ataques no Dataset', fontweight='bold')
plt.xlabel('Tipo de Ataque')
plt.ylabel('Quantidade')
plt.xticks(rotation=45, ha='right')

# Adicionar valores sobre as barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('dashboard_bi/01_distribuicao_classes.png', dpi=300, bbox_inches='tight')

# 2. Matriz de confusão estilizada
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Matriz de Confusão', fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Valor Previsto')
plt.tight_layout()
plt.savefig('dashboard_bi/02_matriz_confusao.png', dpi=300, bbox_inches='tight')

# 3. Medidas de desempenho por classe
report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report = df_report.iloc[:-3]  # Remover métricas agregadas

plt.figure(figsize=(14, 8))
metrics = ['precision', 'recall', 'f1-score']
df_plot = df_report[metrics]

ax = df_plot.plot(kind='bar', width=0.8, figsize=(14, 8), color=cores_dashboard[:3])
plt.title('Métricas de Desempenho por Tipo de Ataque', fontweight='bold')
plt.xlabel('Tipo de Ataque')
plt.ylabel('Valor')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Métrica')

# Adicionar valores sobre as barras
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=10)

plt.tight_layout()
plt.savefig('dashboard_bi/03_metricas_desempenho.png', dpi=300, bbox_inches='tight')

# 4. Principais features na classificação (feature importance)
feature_names = numerical_cols.copy()
for encoder_name, encoder_transformation, encoder_cols in preprocessor.transformers_:
    if encoder_name != 'num':
        feature_names.extend(encoder_cols)

# Extrair importância das features diretamente do classificador
rf_model = modelo.named_steps['classifier']

# Como não podemos obter diretamente as features do pipeline, vamos mostrar as principais features numéricas
feature_importance = rf_model.feature_importances_[:len(numerical_cols)]
idx = np.argsort(feature_importance)

plt.figure(figsize=(12, 6))
plt.barh(range(len(idx)), feature_importance[idx], align='center', color=cores_dashboard[0])
plt.yticks(range(len(idx)), [numerical_cols[i] for i in idx])
plt.title('Importância das Features Numéricas', fontweight='bold')
plt.xlabel('Importância Relativa')
plt.tight_layout()
plt.savefig('dashboard_bi/04_feature_importance.png', dpi=300, bbox_inches='tight')

# 5. Distribuição das previsões para cada classe
plt.figure(figsize=(14, 8))
for i, classe in enumerate(classes):
    # Selecionar amostras que realmente pertencem à classe atual
    indices = (y_test == i)
    # Obter probabilidades preditas para essa classe
    prob_classe = y_proba[indices, i]
    if len(prob_classe) > 0:  # Verificar se há amostras
        sns.kdeplot(prob_classe, label=classe, color=cores_dashboard[i % len(cores_dashboard)])

plt.title('Distribuição das Probabilidades de Previsão', fontweight='bold')
plt.xlabel('Probabilidade Predita')
plt.ylabel('Densidade')
plt.legend(title='Tipo de Ataque')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('dashboard_bi/05_distribuicao_probabilidades.png', dpi=300, bbox_inches='tight')

# 6. Taxa de acerto por indústria alvo
industry_performance = {}
industries = df['Target Industry'].unique()

for industry in industries:
    # Filtrar apenas registros dessa indústria no conjunto de teste
    industry_mask = X_test['Target Industry'] == industry
    if sum(industry_mask) > 0:  # Verificar se há amostras
        industry_actual = y_test[industry_mask]
        industry_pred = y_pred[industry_mask]
        industry_accuracy = sum(industry_actual == industry_pred) / len(industry_actual)
        industry_performance[industry] = industry_accuracy

# Ordenar por taxa de acerto
industry_performance = {k: v for k, v in sorted(industry_performance.items(), key=lambda item: item[1], reverse=True)}

plt.figure(figsize=(12, 6))
plt.bar(industry_performance.keys(), industry_performance.values(), color=cores_dashboard)
plt.title('Taxa de Acerto por Indústria Alvo', fontweight='bold')
plt.xlabel('Indústria')
plt.ylabel('Taxa de Acerto')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('dashboard_bi/06_acerto_por_industria.png', dpi=300, bbox_inches='tight')

# 7. Exemplo de oráculo com casos reais do conjunto de teste
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("ORÁCULO DE CLASSIFICAÇÃO - Exemplos Reais", fontsize=20, fontweight='bold', y=0.98)

# Selecionar 3 exemplos aleatórios
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), size=3, replace=False)

# Para cada exemplo
for i, idx in enumerate(sample_indices):
    amostra = X_test.iloc[idx]
    real_label = le.inverse_transform([y_test[idx]])[0]
    pred_label = le.inverse_transform([y_pred[idx]])[0]
    probs = y_proba[idx]
    acertou = real_label == pred_label
    
    # Criar texto para a amostra
    texto = f"Exemplo {i+1}:\n"
    texto += f"País: {amostra['Country']}\n"
    texto += f"Indústria Alvo: {amostra['Target Industry']}\n"
    texto += f"Perda Financeira: ${amostra['Financial Loss (in Million $)']} milhões\n"
    texto += f"Usuários Afetados: {amostra['Number of Affected Users']}\n"
    texto += f"Fonte do Ataque: {amostra['Attack Source']}\n\n"
    texto += f"Tipo Real: {real_label}\n"
    texto += f"Previsão: {pred_label}"
    
    # Cores para acerto/erro
    cor_borda = 'green' if acertou else 'red'
    
    # Texto da amostra
    axs[i, 0].text(0, 0.5, texto, fontsize=10, va='center')
    axs[i, 0].axis('off')
    
    # Gráfico de barras das probabilidades
    axs[i, 1].bar(classes, probs, color=cores_dashboard)
    axs[i, 1].set_title("Probabilidades", fontsize=10)
    axs[i, 1].set_ylim(0, 1)
    axs[i, 1].tick_params(axis='x', rotation=90)
    
    # Indicador de acerto/erro
    axs[i, 2].pie([1], colors=[cor_borda], wedgeprops=dict(width=0.3))
    axs[i, 2].text(0, 0, "ACERTO" if acertou else "ERRO", ha='center', va='center', fontweight='bold')
    axs[i, 2].axis('equal')

plt.tight_layout()
plt.savefig('dashboard_bi/07_exemplos_oraculo.png', dpi=300, bbox_inches='tight')

# 8. Dashboard completo
from matplotlib.gridspec import GridSpec

plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=plt.gcf())

# Título do dashboard
ax_title = plt.subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.5, "DASHBOARD - CLASSIFICAÇÃO DE ATAQUES CIBERNÉTICOS\nModelo: Random Forest", 
             fontsize=24, fontweight='bold', ha='center', va='center')

# Carregar e adicionar cada visualização
imagens = [
    '01_distribuicao_classes.png',
    '02_matriz_confusao.png', 
    '03_metricas_desempenho.png',
    '04_feature_importance.png',
    '05_distribuicao_probabilidades.png',
    '06_acerto_por_industria.png',
    '07_exemplos_oraculo.png'
]

posicoes = [
    gs[1, 0],
    gs[1, 1],
    gs[2, :],
    gs[3, 0],
    gs[3, 1]
]

for i, img_path in enumerate(imagens[:5]):
    img = plt.imread(f'dashboard_bi/{img_path}')
    ax = plt.subplot(posicoes[i])
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig('dashboard_bi/dashboard_completo.png', dpi=300, bbox_inches='tight')

print("Dashboard BI gerado com sucesso! Visualizações salvas na pasta 'dashboard_bi'.") 