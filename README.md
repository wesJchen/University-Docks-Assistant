## RAG Simulation

This project is a minimal Retrieval-Augmented Generation (RAG) setup that:

- **Indexes** a PDF into a Redis vector database.
- **Retrieves** relevant chunks from Redis for a user question.
- **Generates** an answer using a local Ollama model via `langchain-ollama`.

The core pieces are:

- `scripts/import_script.py` – builds the Redis vector index from a PDF.
- `scripts/retrieval_script.py` – retrieves relevant context for a question from Redis.
- `main.py` – calls retrieval, then uses an Ollama LLM to answer the question using that context.
- `docker-compose.yml` – runs a Redis Stack instance for vector search.

---

## Prerequisites

- **Python** 3.10+ (project currently using 3.12 in `.venv`).
- **Docker** and **docker-compose** installed.
- **Ollama** installed locally and running, with the model `qwen2.5-coder:7b` pulled:

```bash
ollama pull qwen2.5-coder:7b
```

---

## Installation

1. **Create and activate a virtual environment** (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

3. **Start Redis Stack** (for vector search and RedisInsight):

```bash
docker-compose up -d
```

This exposes:

- Redis on `localhost:6379`
- RedisInsight on `http://localhost:8001`

---

## 1. Index your PDF into Redis

Edit `scripts/import_script.py` and set the PDF path:

```python
loader = PyPDFLoader("your_file_goes_here.pdf")  # Replace with your file path
```

Then run:

```bash
python scripts/import_script.py
```

What this does:

- Loads the PDF.
- Splits it into chunks with `RecursiveCharacterTextSplitter`.
- Vectorizes each chunk with `HFTextVectorizer` (`sentence-transformers/all-MiniLM-L6-v2`).
- Stores text + embeddings into a Redis vector index called `rag_index`.

---

## 2. (Optional) Test retrieval only

You can quickly verify that retrieval works by running:

```bash
python scripts/retrieval_script.py
```

This will:

- Ask Redis for the top matches (default 3) for a demo question.
- Print the retrieved context to the console.

---

## 3. Ask questions via the RAG pipeline

The main entry point is `main.py`. It:

- Prompts you for a question.
- Calls `get_context_for_question` from `scripts.retrieval_script` to get relevant context from Redis.
- Feeds that context and your question into an Ollama LLM (`qwen2.5-coder:7b`) using `langchain-ollama`.

Run:

```bash
python main.py
```

Then:

1. Enter your question when prompted.
2. Wait for retrieval + generation.
3. Read the final answer printed under `--- FINAL ANSWER ---`.

---

## Project Structure (key files)

- `main.py` – CLI entry point that runs the full RAG flow.
- `scripts/import_script.py` – builds the Redis vector index from a PDF.
- `scripts/retrieval_script.py` – retrieves context from Redis for a question.
- `docker-compose.yml` – runs a Redis Stack instance with persistence.
- `requirements.txt` – Python dependencies.

---

## Troubleshooting

- **Redis connection errors**: Ensure Docker is running and `docker-compose up -d` has been executed.
- **No results / empty context**: Confirm `scripts/import_script.py` has been run successfully and that `rag_index` exists in Redis.
- **Ollama / model issues**: Make sure the Ollama daemon is running and the `qwen2.5-coder:7b` model is pulled and available.
