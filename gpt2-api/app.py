from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# URL do serviço GPT-2 (usando o nome do serviço no Docker Compose)
GPT2_URL = "http://gpt2:8000/generate"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)

    # Envia a consulta para o serviço GPT-2
    response = requests.post(GPT2_URL, json={"prompt": prompt, "max_length": max_length})
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Erro ao processar a requisição"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)