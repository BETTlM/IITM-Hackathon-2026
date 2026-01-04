# Visa Benefits Finder

Multi-agent system for retrieving and explaining Visa card benefits using local LLMs. Uses LangGraph for agent orchestration, Ollama for text generation, and ChromaDB for vector search.

## Architecture

- Backend: FastAPI server with 7-agent LangGraph workflow
- Frontend: Next.js React application
- LLM: Ollama with llama3.2 model
- Vector DB: ChromaDB with HuggingFace embeddings
- No external APIs required

## Requirements

- Python 3.11+
- Node.js 18+
- Ollama installed and running
- llama3.2 model pulled in Ollama

## Installation

### Backend

```bash
cd backend
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If dependency resolution fails, use the staged installer:

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### Frontend

```bash
cd frontend
npm install
```

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull required model
ollama pull llama3.2

# Verify
ollama list
```

### 2. Ingest Benefits Data

```bash
cd backend
python ingest_simple.py
```

This creates the vector database at `./visa_db` from `data/benefits.txt`.

If you encounter scipy issues:

```bash
./fix_scipy.sh
python ingest_simple.py
```

### 3. Configure Frontend API URL

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, set this to your backend URL.

## Running

### Start Backend

```bash
cd backend
python main.py
```

Server runs on `http://localhost:8000`

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on `http://localhost:3000`

## API Usage

### POST /benefits

Request:
```json
{
  "card_number": "4111-****-****-1111",
  "user_context": "student",
  "preferred_language": "en",
  "location": "Chennai"
}
```

Response:
```json
{
  "status": "success",
  "card_tier": "Classic",
  "recommended_benefit": {
    "explanation": "...",
    "source_chunks": [...],
    "scores": {...}
  },
  "recommendations": [...],
  "all_benefits": [...],
  "disclaimers": [...],
  "metadata": {
    "bin_validated": true,
    "rag_grounded": true,
    "compliance_approved": true
  }
}
```

### Card Number Format

Must be masked: `4XXX-****-****-XXXX`

Valid examples:
- `4111-****-****-1111`
- `4222-****-****-2222`
- `4333-****-****-3333`

### User Context Options

- `student`
- `travel`
- `dining_entertainment_shopping`
- `services`

### Language Options

- `en` (English)
- `ta` (Tamil)

## Workflow

1. Supervisor Agent: Orchestrates workflow
2. Card Intelligence Agent: Validates card format, determines tier
3. Benefit Retrieval Agent: RAG-based semantic search
4. Explanation Agent: Generates plain-language explanations
5. Recommendation Agent: Ranks benefits by relevance
6. Language Agent: Translates if needed
7. Compliance Agent: Validates output, adds disclaimers

## Configuration

### Change LLM Model

Edit `backend/app/graph.py`:

```python
llm = ChatOllama(model="your-model", temperature=0.3)
```

### Change Embedding Model

Edit `backend/ingest.py` and `backend/app/graph.py`:

```python
embeddings = HuggingFaceEmbeddings(model_name="your-model")
```

### CORS Configuration

Edit `backend/main.py`:

```python
cors_origins = os.getenv("CORS_ORIGINS", "*")
```

Set `CORS_ORIGINS` environment variable to restrict origins.

## Troubleshooting

### Ollama not responding

```bash
ollama list
curl http://localhost:11434/api/tags
```

Ensure Ollama service is running and llama3.2 is pulled.

### Vector database missing

```bash
cd backend
python ingest_simple.py
```

Verify `./visa_db` directory exists.

### No benefits returned

- Check `data/benefits.txt` has content
- Verify vector database was created
- Ensure embedding model matches between ingest and retrieval

### Frontend can't connect to backend

- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Check browser console for CORS errors

## Project Structure

```
backend/
├── app/
│   └── graph.py          # LangGraph workflow
├── data/
│   └── benefits.txt      # Source documents
├── ingest.py             # Data ingestion
├── ingest_simple.py      # Simplified ingestion
├── main.py               # FastAPI server
└── requirements.txt      # Dependencies

frontend/
├── app/
│   ├── page.tsx          # Main page
│   └── layout.tsx        # Root layout
├── components/           # React components
├── contexts/            # React contexts
└── lib/                 # Utilities
```

## Constraints

- Masked cards only (format: `4XXX-****-****-XXXX`)
- No data persistence
- Awareness-only (no transactions)
- RAG-grounded (all output from source documents)
- Compliance validation on all responses

## API Documentation

Swagger UI available at `http://localhost:8000/docs` when backend is running.
