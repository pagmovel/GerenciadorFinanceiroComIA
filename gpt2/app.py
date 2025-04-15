from flask import Flask, request, jsonify
from transformers import pipeline, set_seed

print("Carregando o modelo BLOOM...")
# Pode escolher uma versão menor se os recursos forem limitados:
model_name = "bigscience/bloom-560m"  # ou "bigscience/bloom" para a versão completa, se o hardware permitir
generator = pipeline("text-generation", model=model_name)
set_seed(42)

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)

    output = generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    response = {"response": output[0]['generated_text']}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
