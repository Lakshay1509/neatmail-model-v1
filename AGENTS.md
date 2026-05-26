# AGENTS.md

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, PINECONE_API_KEY, DASHBOARD_API_KEY
```

Required env vars:
- `OPENAI_API_KEY` — used for GPT-4o-mini (classification) and text-embedding-3-small (embeddings)
- `PINECONE_API_KEY` — Pinecone vector DB (index `neatmail-corrections` auto-created on startup)
- `DASHBOARD_API_KEY` — Bearer-like API key; all endpoints require `X-API-Key` header

## Running

```bash
python main.py                       # dev
uvicorn main:app --host 0.0.0.0 --port 8000   # or Docker
docker build -t neatmail . && docker run -p 8000:8000 --env-file .env neatmail
```

## Architecture

Single-file FastAPI app (`main.py`). Two endpoints:

- `POST /classify` — classifies an email into one of the provided tags using GPT-4o-mini with few-shot corrections from Pinecone
- `POST /correct` — stores a user correction as a vector embedding for future few-shot prompts

No tests, no lint config, no CI. No codegen or migrations.
