from fastapi import FastAPI
import pickle
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Carregar os modelos e os dados
with open("model/knn_model.pkl", "rb") as knn_file:
    knn = pickle.load(knn_file)

with open("model/kmeans_model.pkl", "rb") as kmeans_file:
    kmeans = pickle.load(kmeans_file)

with open("model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("model/history_mapping.json", "r", encoding="utf-8") as history_file:
    history_mapping = json.load(history_file)

# Criar a API com FastAPI
app = FastAPI()

# Estrutura para receber o input da API
class UserRequest(BaseModel):
    user_id: str

# Criar endpoint para recomendações
@app.post("/recommend")
def recomendar_paginas(request: UserRequest):
    user_id = request.user_id
    
    if user_id not in history_mapping or len(history_mapping[user_id]) == 0:
        # Cold-start para usuário novo: recomenda páginas com base no cluster do KMeans
        user_features = np.array([scaler.transform([[0]*9])])  # Usuário novo sem dados
        user_cluster = kmeans.predict(user_features.reshape(1, -1))[0]
        
        # Encontrar usuários no mesmo cluster
        cluster_members = [uid for uid, cluster in zip(history_mapping.keys(), kmeans.labels_) if cluster == user_cluster]
        
        recomendacoes = set()
        for member_id in cluster_members:
            recomendacoes.update(history_mapping.get(member_id, []))
        
        return {"user_id": user_id, "recommendations": list(recomendacoes)[:6]}
    
    # Caso contrário, usa o modelo KNN
    user_features = np.array([scaler.transform([[0]*9])])  # Usuário existente com histórico
    neighbors = knn.kneighbors(user_features.reshape(1, -1), n_neighbors=6, return_distance=False)[0]
    
    recomendacoes = set()
    for neighbor in neighbors:
        neighbor_id = list(history_mapping.keys())[int(neighbor)]
        recomendacoes.update(history_mapping.get(neighbor_id, []))
    
    return {"user_id": user_id, "recommendations": list(recomendacoes)[:6]}

# Executar a API no terminal com uvicorn
# Comando: uvicorn api_recommendation:app --reload