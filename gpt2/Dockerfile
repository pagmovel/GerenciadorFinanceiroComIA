FROM python:3.9-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    git iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências e instala as bibliotecas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código para o container
COPY . .

# Exponha a porta que o GPT-2 utilizará
EXPOSE 8000

# Comando para iniciar o serviço GPT-2
CMD ["python", "app.py"]