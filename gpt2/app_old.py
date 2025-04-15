from flask import Flask, request, jsonify
from transformers import pipeline, set_seed
from huggingface_hub import snapshot_download

# Faz o download completo do modelo "gpt2" e salva em cache/localmente.
print("Fazendo download do modelo GPT-2...")
# model_folder = snapshot_download(repo_id="gpt2")
model_folder = snapshot_download(repo_id="pierreguillou/gpt2-small-portuguese")
print("Modelo baixado em:", model_folder)

# Agora o pipeline usará o modelo já baixado localmente.
generator = pipeline('text-generation', model=model_folder)
set_seed(42)  # Define uma seed para reproducibilidade

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)

    # Gera texto com o modelo
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    response = {"response": output[0]['generated_text']}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)



# Faz o download do modelo em português
model_folder = snapshot_download(repo_id="pierreguillou/gpt2-small-portuguese")
generator = pipeline('text-generation', model=model_folder)
