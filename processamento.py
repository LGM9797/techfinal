import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import pytz # Importe pytz
from sklearn.metrics import precision_score, recall_score

# 1. Carregamento dos Dados
try:
    df_itens = pd.read_csv("itens/itens/itens-parte3.csv")
    df_treino = pd.read_csv("files/treino/treino_parte1.csv")
    df_validacao = pd.read_csv("validacao.csv")
    print("Arquivos carregados com sucesso!")
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit()

def string_para_lista(s):
    """Converte uma string representando uma lista em uma lista Python."""
    if isinstance(s, list):
        return s  # Se já for uma lista, retorna sem modificar

    if pd.isna(s):  # Check for NaN values
        return []

    s = str(s).replace('[', '').replace(']', '').replace("'", '').strip()
    return [x.strip() for x in s.split(',') if x.strip()] # Tratamento para evitar strings vazias

def safe_float(value):
    """Tenta converter um valor para float; retorna NaN em caso de erro."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def safe_int(value):
    """Tenta converter um valor para inteiro; retorna NaN em caso de erro."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return np.nan

def safe_datetime(value):
    """Tenta converter um valor para datetime; retorna NaT em caso de erro."""
    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return pd.NaT

#3 - Explodir colunas do dataset treino
colunas_para_explodir = ['history', 'timestampHistory', 'numberOfClicksHistory', 'timeOnPageHistory', 'scrollPercentageHistory', 'pageVisitsCountHistory']

for col in colunas_para_explodir:
    df_treino[col] = df_treino[col].apply(string_para_lista)

# Explodir as colunas no dataframe de treino
df_treino_exploded = df_treino.explode(colunas_para_explodir)

# Converter colunas explodidas para os tipos corretos
df_treino_exploded['timestampHistory'] = df_treino_exploded['timestampHistory'].apply(safe_int)
df_treino_exploded['numberOfClicksHistory'] = df_treino_exploded['numberOfClicksHistory'].apply(safe_int)
df_treino_exploded['timeOnPageHistory'] = df_treino_exploded['timeOnPageHistory'].apply(safe_float)
df_treino_exploded['scrollPercentageHistory'] = df_treino_exploded['scrollPercentageHistory'].apply(safe_float)
df_treino_exploded['pageVisitsCountHistory'] = df_treino_exploded['pageVisitsCountHistory'].apply(safe_int)

# Converter timestamp para datetime e forçar para UTC
df_treino_exploded['timestampHistory'] = pd.to_datetime(df_treino_exploded['timestampHistory'], unit='ms', utc=True) # Força para UTC

# Encontrar data_limite_treino como Timestamp em UTC
data_limite_treino = df_treino_exploded['timestampHistory'].max()

# Explodir colunas do dataset de validação
df_validacao['history'] = df_validacao['history'].apply(string_para_lista)
df_validacao['timestampHistory'] = df_validacao['timestampHistory'].apply(lambda x: [safe_int(i) for i in x])

# Converter colunas explodidas para os tipos corretos
df_validacao['timestampHistory'] = df_validacao['timestampHistory'].apply(lambda x: [pd.to_datetime(i, unit='ms') for i in x])

# Exemplo de uso das funções no conjunto de dados de validação
df_validacao['history'] = df_validacao['history'].apply(string_para_lista)
df_validacao['timestampHistory'] = df_validacao['timestampHistory'].apply(lambda x: [safe_datetime(i) for i in x])

# Converter data/hora das matérias para UTC
df_itens['issued'] = pd.to_datetime(df_itens['issued'], utc=True) # Garante que 'issued' esteja em UTC
df_itens['modified'] = pd.to_datetime(df_itens['modified'], utc=True) # Garante que 'modified' esteja em UTC

# Feature 1: Popularidade (contagem de ocorrências em history)
noticias_populares = Counter(df_treino_exploded['history'])
df_itens['popularidade'] = df_itens['page'].map(noticias_populares).fillna(0)  # NaN vira 0
print("\nTop 10 notícias mais populares:")
print(df_itens.sort_values(by='popularidade', ascending=False).head(10)[['title', 'popularidade']])

# Feature 2: Recência (idade da matéria)
df_itens['idade_materia'] = (data_limite_treino - df_itens['issued']).dt.days
print("\nIdade das matérias (em dias):")
print(df_itens[['title', 'issued', 'idade_materia']].sample(5))

# Exemplo de uso das funções no conjunto de dados de validação
df_validacao['history'] = df_validacao['history'].apply(string_para_lista)
df_validacao['timestampHistory'] = df_validacao['timestampHistory'].apply(lambda x: [safe_datetime(i) for i in x])



K = 10  # Recomendar top 10 notícias
def recomendar_noticias(user_history, df_itens, K):
    """Recomenda as K notícias mais populares que o usuário não viu."""
    vistas = user_history  # Lista de noticias já vistas pelo usuário
    # Filtra para manter apenas notícias que o usuário NÃO viu.
    noticias_nao_vistas = df_itens[~df_itens['page'].isin(vistas)].copy()
    # Ordena as notícias não vistas pela popularidade (do maior para o menor)
    noticias_ordenadas = noticias_nao_vistas.sort_values(by='popularidade', ascending=False)
    # Retorna o top K
    top_k_noticias = noticias_ordenadas['page'].head(K).tolist()
    return top_k_noticias

# Avaliação do Modelo
def avaliar_modelo(df_validacao, df_itens, K):
    print('chegou aqui 1')
    """Avalia o modelo de recomendação usando Precision@K e Recall@K."""
    precisions = []
    recalls = []

    for index, row in df_validacao.iterrows():
        user_id = row['userId']
        user_history = row['history']
        # Simulate the next articles by taking the real history
        noticias_relevantes = user_history
        # Make the recommendations
        top_k_noticias = recomendar_noticias(user_history, df_itens, K)

        # Calcula precision e recall, tratando caso de listas vazias para evitar erros
        if top_k_noticias and noticias_relevantes:
            precision = len(set(top_k_noticias) & set(noticias_relevantes)) / len(top_k_noticias)
            recall = len(set(top_k_noticias) & set(noticias_relevantes)) / len(noticias_relevantes)
            precisions.append(precision)
            recalls.append(recall)
        else:
            # Se alguma das listas estiver vazia, precision e recall são 0
            precisions.append(0.0)
            recalls.append(0.0)
    print('chegou aqui 2')
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    return mean_precision, mean_recall


# Aplicar o modelo e avaliar
mean_precision, mean_recall = avaliar_modelo(df_validacao, df_itens, K)

print(f"Média Precision@{K}: {mean_precision:.4f}")
print(f"Média Recall@{K}: {mean_recall:.4f}")

