# Generative AI interactive Chatbot with Ollama and Docker

A simple chatbot project using locally run Ollama models and Docker for API integrations.

## Setup
1. Install Docker and Python 3.9+.
2. Clone this repo and `cd` into `chatbot_project/`.
3. Run `docker-compose up --build`.
4. Pull a model in the Ollama container: `docker exec -it chatbot_project-ollama-1 bash`, then `ollama pull llama3`.
5. Visit `http://localhost:5000` to chat.

## Directory Structure
- `app/`: Flask app with API and UI.
- `docker/`: Dockerfiles for Ollama and Flask app.
- `docker-compose.yml`: Orchestrates services.

## Shutdown
Run `docker-compose down`
