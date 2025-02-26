from flask import Flask, request, Response, render_template, stream_with_context, jsonify
import datetime
import os
import logging
import json
import ollama

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Ollama configuration
OLLAMA_MODEL = "deepseek-r1:1.5b"
logging.info(f"Ollama model set to: {OLLAMA_MODEL}")

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
        import requests
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

# Chat streaming endpoint with Ollama
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

        # Build conversation history as a reference string
        history_text = "".join(
            f"{entry['role'].capitalize()}: {entry['content']}\n"
            for entry in conversation_context
        ) if conversation_context else "No prior conversation."

        # System prompt with history as reference
        system_prompt = (
            f"You are a helpful AI assistant who is very good at having an engaging conversation. "
            f"Below is the conversation history for reference only—DO NOT answer questions from it unless explicitly asked. "
            f"Focus solely on the current user prompt and answer it directly. Use web results if provided.\n\n"
            f"Conversation History (reference only):\n{history_text}\n\n"
        )

        # Current prompt with web info
        full_prompt = f"{prompt}{web_info}"

        # Prepare messages—system prompt with history, then current user prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Answer this: {full_prompt}"}
        ]

        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options={"context_window": 4096}
            )
            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                if content:
                    full_response += content
                    yield f"data: {json.dumps({'chunk': content, 'web_used': bool(web_data)})}\n\n"
            
            if full_response:
                conversation_context.append({"role": "user", "content": prompt})
                conversation_context.append({"role": "assistant", "content": full_response})
            elif not prompt.strip():
                yield f"data: {json.dumps({'chunk': 'Hey bro, what\'s up?'})}\n\n"
            else:
                logging.warning("No response from Ollama")
                yield f"data: {json.dumps({'chunk': 'Nothing came back, bro. Try again?'})}\n\n"
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            yield f"data: {json.dumps({'error': f'Error: {e}'})}\n\n"

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
    debug_mode = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    logging.info(f"Starting Flask app (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)