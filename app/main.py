from flask import Flask, request, render_template, jsonify
import requests
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Ollama API endpoint
OLLAMA_API = os.getenv("OLLAMA_HOST", "http://localhost:11434")
logging.debug(f"OLLAMA_API set to: {OLLAMA_API}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    logging.debug(f"Received prompt: {prompt}")
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }
    logging.debug(f"Sending to Ollama: {payload}")
    try:
        response = requests.post(f"{OLLAMA_API}/api/generate", json=payload)
        logging.debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        result = response.json().get('response', 'Error: No response')
        logging.debug(f"Response content: {result}")
        return jsonify({"response": result})
    except Exception as e:
        logging.error(f"Error calling Ollama: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)