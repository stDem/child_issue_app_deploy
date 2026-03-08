# Child-Issue AI (Ollama) — React prototype

This is a **simple architecture** prototype:
- FastAPI backend calls your local **Ollama** (embeddings + llama3)
- React + SCSS UI with:
  - issue list + details
  - merged "same error pattern" row
  - root causes row
  - interactive world map with plant markers
  - risk + trend + per-plant bar chart
  - removable child relations (trash icon)
  - download enriched JSON

## Data
See `data/`:
- `MDBB_parts_usage.json` (expanded, includes equivalence groups with different material numbers)
- `Issues_train.json` (expanded history with RCA hints)
- `Issues_test_errors.json` (test set: missing children including cross-material cases)
- `sample_leads.json` (subset for UI)

## Run backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

## Ollama models
```bash
ollama pull nomic-embed-text:latest
ollama pull llama3:latest
```
