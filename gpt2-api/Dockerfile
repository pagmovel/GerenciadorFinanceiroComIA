FROM python:3.9-slim

WORKDIR /app

# Instala as dependências da API
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY . .

# Exponha a porta da API
EXPOSE 5000

# Comando para iniciar a API
CMD ["python", "app.py"]