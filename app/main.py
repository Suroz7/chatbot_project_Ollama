from flask import Flask, request, render_template, Response, stream_with_context, jsonify
import requests
import os
import logging
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
logging.basicConfig(level=logging.DEBUG)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434"
logging.debug(f"OLLAMA_API set to: {OLLAMA_API}")

# Global variable to store context as a list of message pairs
conversation_context = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat/stream')
def chat_stream():
    prompt = request.args.get('prompt', '')
    logging.debug(f"Received prompt for streaming: {prompt}")
    
    def generate(prompt):
        global conversation_context
        
        # Build conversation history as text
        conversation_text = ""
        for entry in conversation_context:
            conversation_text += f"Human: {entry['human']}\n"
        
        # Build the payload with the conversation history in the prompt
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": f"{conversation_text}Be polite and just answer.\nHuman: {prompt}\nAssistant: ",
            "stream": True,
            "context_window": 4096
        }
        
        logging.debug(f"Payload sent to Ollama: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(f"{OLLAMA_API}/api/generate", json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get('response', '')
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Store the conversation pair after getting full response
            if full_response:
                conversation_context.append({
                    'human': prompt,
                    'assistant': full_response
                })
                logging.debug(f"Updated conversation context: {conversation_context}")
            
            if not full_response:
                logging.warning("No response content received from stream")
                        
        except Exception as e:
            logging.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate(prompt)), mimetype='text/event-stream')

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    global conversation_context
    conversation_context = []
    return jsonify({"status": "success", "message": "Conversation history cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)