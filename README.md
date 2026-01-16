# Stock Investment AI Service

AI Service for Multi-Agent AI Stock Investment System built with Python and FastAPI using Clean Architecture.

## Architecture

This project follows Clean Architecture principles with the following layers:

- **api/** - API Layer (FastAPI routes, models)
- **application/** - Application Layer (Use cases, services)
- **domain/** - Domain Layer (Entities, interfaces)
- **infrastructure/** - Infrastructure Layer (LLM clients, vector store, message queue)
- **shared/** - Shared utilities and exceptions

## Prerequisites

- Python 3.11+
- Blackbox API Key
- Qdrant (Vector Database)

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```
   BLACKBOX_API_KEY=yourkey
   QDRANT_URL=http://localhost:6333
   RABBITMQ_CONNECTION_STRING=amqp://guest:guest@localhost:5672/
   ```

4. Run the service:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

## API Endpoints

- `POST /api/summarize` - Summarize news articles
- `POST /api/analyze-event` - Analyze corporate events
- `POST /api/forecast` - Generate stock forecasts
- `POST /api/qa` - Answer questions with RAG
- `POST /api/parse-alert` - Parse natural language alert requests

## Docker

Build and run with Docker:
```bash
docker build -f docker/Dockerfile -t ai-service .
docker run -p 8000:8000 --env-file .env ai-service
```

