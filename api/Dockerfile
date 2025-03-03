FROM python:3.9-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY api_recommendation.py .
# COPY ../model/ ./model/

CMD ["uvicorn", "api_recommendation:app", "--host", "0.0.0.0", "--port", "8000"]
