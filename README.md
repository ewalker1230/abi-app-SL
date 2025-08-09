## ABI — Agentic Business Intelligence

Built as a solution for Hatch-a-thon 2025. 

A lightweight Streamlit app to explore your data with natural‑language questions. Upload files, index them locally with Chroma, and ask questions to get concise answers and simple charts.

### What it does
- Ingests Excel (.xlsx/.xls) and text (.txt) files; CSV is also supported
- Stores processed content locally in `chroma_db/`
- Lets you ask questions and get AI‑assisted answers
- Can generate basic visualizations on request
- Optionally logs conversations and session metadata to Redis

### Agentic behavior
- Agents run multi‑step analyses (“Analyze All Data”, “Quality Assessment”, “Query Suggestions”).
- Retrieval adapts context size by intent/complexity.
- Optional Redis provides session memory.

### Quick start
1) Python 3.11 recommended
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Provide your API key (env or .env)
```bash
export OPENAI_API_KEY=your_key
```
4) Run the app
```bash
streamlit run main.py
```

Then upload data and use the chat input. You can also trigger built‑in agents like “Analyze All Data”, “Quality Assessment”, and “Query Suggestions”.

### Docker (optional)
The included `docker-compose.yml` starts Postgres (optional), Redis (for session/history), and the app.
```bash
docker compose up --build
```
App: http://localhost:8501

### Notes
- Vector index persists in `chroma_db/`. If it becomes read‑only/corrupted, use the in‑app “Reset Database” control.
- An admin dashboard for performance metrics is available via `pages/1_Admin_Dashboard.py` (disabled by default).

