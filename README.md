
# 🔐 CyberThreat Classifier & BI Dashboard

Este projeto combina análise de dados, visualização de BI e modelos de aprendizado de máquina para classificar **tipos de ataques cibernéticos** com base em um conjunto realista de dados globais de segurança cibernética (2015–2024). Além disso, o projeto conta com um **oráculo preditivo**, um **painel de visualização interativo** e uma análise detalhada de **importância de atributos**.

## 📁 Estrutura do Projeto

```
├── analyze_data.py                   # Análise exploratória e estatísticas descritivas
├── dashboard_bi_classificacao.py    # Geração do dashboard com visualizações e modelo de classificação
├── importancia_previsoes.py         # Análise de importância de variáveis usando Random Forest e XGBoost
├── oraculo_classificacao.py         # Oráculo de classificação para prever tipos de ataques com base em dados de entrada
├── dashboard_bi/                    # Pasta gerada com gráficos do dashboard
├── *.csv                            # Arquivos CSV com importâncias e top features
```

## 📊 Objetivo

Criar um modelo preditivo capaz de identificar o **tipo de ataque cibernético** com base em variáveis como:

- País
- Indústria alvo
- Fonte do ataque
- Tipo de vulnerabilidade
- Mecanismo de defesa utilizado
- Perdas financeiras
- Tempo de resolução
- Número de usuários afetados

## 📌 Funcionalidades

### ✅ `analyze_data.py`

- Estatísticas gerais e descritivas
- Detecção de outliers (IQR)
- Totalizações por perda, usuários afetados e tempo
- Agrupamentos por tipo de ataque e por ano

### 📊 `dashboard_bi_classificacao.py`

- Treinamento de modelo Random Forest (com pipeline)
- Geração de visualizações como:
  - Distribuição de tipos de ataques
  - Matriz de confusão
  - Métricas de desempenho por classe
  - Importância das features
  - Oráculo com exemplos reais
  - Acertos por setor
- Gera o dashboard completo como imagem

### 🔍 `importancia_previsoes.py`

- Compara as importâncias das features entre modelos Random Forest e XGBoost
- Salva as top features em arquivos `.csv`
- Gera análises por **categoria** e por **tipo de ataque**
- Cria um resumo das 10 features mais relevantes para cada classe

### 🧠 `oraculo_classificacao.py`

- Treina modelo Random Forest final com todos os dados
- Função `classificar_ataque(dados)` para prever o tipo de ataque
- Exemplos simulados de classificação com saída interpretável

## 🛠️ Requisitos

- Python 3.8+
- Bibliotecas:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - xgboost

Instalação das dependências:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## 💡 Como executar

1. Faça a análise exploratória:
   ```bash
   python analyze_data.py
   ```

2. Gere o dashboard e treine o modelo:
   ```bash
   python dashboard_bi_classificacao.py
   ```

3. Veja a importância das variáveis para cada modelo:
   ```bash
   python importancia_previsoes.py
   ```

4. Execute o oráculo para simulações:
   ```bash
   python oraculo_classificacao.py
   ```

## 📁 Dados

O projeto utiliza o dataset `Global_Cybersecurity_Threats_2015-2024.csv` (não incluso aqui). O arquivo deve estar na mesma pasta dos scripts.

## 📌 Exemplos de Uso do Oráculo

```python
exemplo = {
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
```

## 📊 Resultados

Todos os gráficos são salvos automaticamente na pasta `dashboard_bi/`, incluindo:

- `dashboard_completo.png`
- `01_distribuicao_classes.png`
- `03_metricas_desempenho.png`
- e outros...

## 📌 Licença

Este projeto é apenas para fins educacionais e acadêmicos.
