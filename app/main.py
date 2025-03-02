from flask import Flask, request, Response, render_template, stream_with_context, jsonify
import datetime
import os
import logging
import json
import requests  # Use requests instead of ollama client

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Ollama configuration with external endpoint
OLLAMA_API = 'https://joly.work.gd'
MODEL_NAME = "deepseek-r1:1.5b"
logging.info(f"API endpoint set to: {OLLAMA_API}")
logging.info(f"Model set to: {MODEL_NAME}")

# Google Custom Search API credentials
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyCMRicu4UnrCmn-8rr29GUqpB_NUzf-k3c")
GOOGLE_CSE_ID = os.environ.get('GOOGLE_CSE_ID', "564c4d8b512164bb9")

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
        "who is the current", "what is the current", "when did", "where is", "how to", "election", "covid",
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
        combined_snippets = "\n\n".join(snippets)[:2000]
        logging.debug(f"Extracted {len(snippets)} snippets: {combined_snippets}")
        return {"combined_snippets": combined_snippets, "search_engine": "Google API"}
    except Exception as e:
        logging.error(f"Google API search failed: {e}")
        return {"error": str(e)}

# Chat streaming endpoint with direct API calls
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
                search_message = f'Using {web_data["search_engine"]} search results...\n\n'
                yield f"data: {json.dumps({'chunk': search_message})}\n\n"
            elif "error" in web_data:
                error_message = f'Web search failed: {web_data["error"]}\n\n'
                yield f"data: {json.dumps({'chunk': error_message})}\n\n"

        # Build conversation history as a reference string
        history_text = "".join(
            f"{entry['role'].capitalize()}: {entry['content']}\n"
            for entry in conversation_context
        ) if conversation_context else "No prior conversation."

        # Create full prompt with system instructions, history and current question
        full_prompt = (
            f"You are a helpful AI assistant with a memory of conversation history with the user. "
            f"Below is the conversation history for reference:\n\n{history_text}\n\n"
            f"User's current question: {prompt}\n\n"
            f"{web_info if web_info else ''}"
            f"Please provide a helpful response."
        )
        
        # Direct API call to /api/generate endpoint
        api_url = f"{OLLAMA_API}/api/generate"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": True,
            "context_window": 4096
        }

        try:
            logging.info(f"Sending generate request to API at {api_url}")
            response = requests.post(api_url, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line)
                        # Extract response from the generate API format
                        chunk = chunk_data.get('response', '')
                        if chunk:
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk, 'web_used': bool(web_data)})}\n\n"
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse API response: {e}")
            
            if full_response:
                conversation_context.append({"role": "user", "content": prompt})
                conversation_context.append({"role": "assistant", "content": full_response})
            elif not prompt.strip():
                yield f"data: {json.dumps({'chunk': 'Hey there! How can I help you today?'})}\n\n"
            else:
                logging.warning("No response from API")
                error_response = "Sorry, I couldn't generate a response. Please try again."
                yield f"data: {json.dumps({'chunk': error_response})}\n\n"
        except Exception as e:
            logging.error(f"API error: {e}")
            error_message = f"Error: {e}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(stream_with_context(generate(prompt)), mimetype='text/event-stream')

# Clear conversation history
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

# Download chat history
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
    debug_mode = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    logging.info(f"Starting Flask app (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)