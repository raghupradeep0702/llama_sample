from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

llm = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    input_text = data['input_text']
    generated_text = []

    for word in llm(input_text, stream=True):
        generated_text.append(word)

    return jsonify({'generated_text': ''.join(generated_text)})

if __name__ == '__main__':
    app.run(debug=True)
