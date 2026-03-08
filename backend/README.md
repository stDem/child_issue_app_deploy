# Backend (FastAPI)

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn server:app --reload --port 8000
```

If Ollama is not on localhost:
```bash
export OLLAMA_URL="http://<host>:11434"
```
