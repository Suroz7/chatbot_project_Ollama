from flask import Flask, request, Response, render_template, stream_with_context, jsonify
import datetime
import requests
import os
import logging
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Ollama API endpoint
OLLAMA_API = os.environ.get('OLLAMA_API', 'http://localhost:11434')
logging.info(f"OLLAMA_API set to: {OLLAMA_API}")

# Google Custom Search API credentials
GOOGLE_API_KEY = "AIzaSyCMRicu4UnrCmn-8rr29GUqpB_NUzf-k3c"  # Replace with your API key
GOOGLE_CSE_ID = "564c4d8b512164bb9"  # Replace with your Search Engine ID

# Global conversation context
conversation_context = []

# Serve external index.html
@app.route('/')
def index():
    return render_template('index.html')

# Check if web search is needed
def should_search_web(prompt):
    trigger_words = [
        "news", "current", "latest", "update", "weather", "stocks", "stock price",
        "market", "price of", "what is happening", "recent events", "today's",
        "who is", "what is", "when did", "where is", "how to", "election", "covid",
        "pandemic", "score", "release date", "breaking news", "headlines", "trending",
        "now", "forecast", "temperature", "conditions", "finance", "economy", "trade",
        "value of", "what’s new", "this week", "yesterday", "tomorrow", "event",
        "sports", "game update", "results", "politics", "health", "vaccine", "stats",
        "time of", "location", "directions", "tutorial", "announcement"
    ]
    result = any(word.lower() in prompt.lower() for word in trigger_words)
    logging.debug(f"Web search check for '{prompt[:30]}...': {result}")
    return result

# Perform Google API search
def web_search(query, num_results=3):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": num_results
        }
        logging.info(f"Searching Google API: '{query}'")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        snippets = [
            f"{item.get('title', 'No title')}\n{item.get('snippet', 'No snippet')}"
            for item in search_results.get("items", [])
        ]
        combined_snippets = "\n\n".join(snippets)[:2000]  # Limit to 2000 chars
        logging.debug(f"Extracted {len(snippets)} snippets: {combined_snippets}")
        return {"combined_snippets": combined_snippets, "search_engine": "Google API"}
    except Exception as e:
        logging.error(f"Google API search failed: {e}")
        return {"error": str(e)}

# Chat streaming endpoint
@app.route('/api/chat/stream')
def chat_stream():
    prompt = request.args.get('prompt', '')
    logging.info(f"Received prompt: '{prompt[:50]}...'")

    def generate(prompt):
        global conversation_context

        # Web search if needed
        web_data = None
        web_info = ""
        if prompt.strip() and should_search_web(prompt):
            web_data = web_search(prompt)
            if "combined_snippets" in web_data:
                web_info = f"\n\n{web_data['search_engine']} search results:\n{web_data['combined_snippets']}\n\n"
                yield f"data: {json.dumps({'chunk': f'Using {web_data['search_engine']} search results...\n\n'})}\n\n"
            elif "error" in web_data:
                yield f"data: {json.dumps({'chunk': f'Web search failed: {web_data['error']}\n\n'})}\n\n"

        # History only if explicitly referenced
        history_triggers = ["earlier", "before", "previous", "last", "history"]
        conversation_text = "".join(
            f"Human: {entry['human']}\nAssistant: {entry['assistant']}\n"
            for entry in conversation_context
        ) if any(word in prompt.lower() for word in history_triggers) else ""

        # Build prompt
        final_prompt = f"{prompt}{web_info}"
        full_prompt = (
            f"You are a helpful AI assistant. who is very good at having an engaging conversation\n\n"
            f"History (for explicit reference only):\n{conversation_text or '(No history used)'}\n\n"
             f"Instructions: Do not answer from history   Answer only: '{prompt}'. Use web results if provided.\n\n"
            f"Current question: {final_prompt}\nAssistant: "
            
        )

        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": full_prompt,
            "stream": True,
            "context_window": 4096
        }

        try:
            logging.debug(f"Sending to Ollama: {OLLAMA_API}/api/generate")
            response = requests.post(f"{OLLAMA_API}/api/generate", json=payload, stream=True)
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get('response', '')
                    if chunk:
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk, 'web_used': bool(web_data)})}\n\n"
            if full_response:
                conversation_context.append({'human': prompt, 'assistant': full_response})
                logging.info(f"Context updated: {len(conversation_context)} entries")
            elif not prompt.strip():
                yield f"data: {json.dumps({'chunk': 'Hey bro, what\'s up?'})}\n\n"
            else:
                logging.warning("No response from Ollama")
                yield f"data: {json.dumps({'chunk': 'Nothing came back, bro. Try again?'})}\n\n"
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            yield f"data: {json.dumps({'error': f'Error: {e}'})}\n\n"

    return Response(stream_with_context(generate(prompt)), mimetype='text/event-stream')

# Clear and download routes remain unchanged...
@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    global conversation_context
    logging.info("Clearing conversation history")
    conversation_length = len(conversation_context)
    conversation_context = []
    return jsonify({
        "status": "success",
        "message": f"Cleared {conversation_length} messages, bro!"
    })

@app.route('/api/chat/download', methods=['GET'])
def download_chat():
    global conversation_context
    logging.info("Downloading chat history")
    chat_history = {
        'conversations': conversation_context,
        'timestamp': datetime.datetime.now().isoformat()
    }
    response = jsonify(chat_history)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=chat_history.json'
    return response

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    logging.info(f"Starting Flask app (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)