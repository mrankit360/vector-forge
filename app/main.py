
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .vector_db import VectorDB
from .document_db import DocumentDB
from .ollama_client import OllamaClient
from .chunker import chunk_text
from .demo_data import load_demo


DIMS = 16

app = FastAPI(title="VectorDB Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


def require_query_vector(value):
    try:
        values = [float(x) for x in value.split(",")]
    except (ValueError, AttributeError):
        raise HTTPException(400, "Invalid vector")

    if len(values) != DIMS:
        raise HTTPException(400, f"Need {DIMS}D vector")

    return values


def require_document_question(body):
    question = body.get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(400, "Need question")
    return question.strip()


@app.get("/")
def index():
    path = Path("index.html")
    if not path.is_file():
        raise HTTPException(404, "index.html not found")
    return FileResponse(path)


@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine",
           algo: str = "hnsw"):
    try:
        return db.search(require_query_vector(v), k, metric, algo)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/insert")
async def insert(request: Request):
    body = await request.json()

    try:
        item_id = db.insert(
            body["metadata"],
            body.get("category", ""),
            body["embedding"],
        )
        return {"id": item_id}
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.delete("/delete/{item_id}")
def delete(item_id: int):
    return {"ok": db.remove(item_id)}


@app.get("/items")
def items():
    return db.all()


@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        return db.benchmark(require_query_vector(v), k, metric)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/hnsw-info")
def hnsw_info():
    # USearch does not expose the same graph inspection interface
    # as the custom C++ HNSW class.
    items = db.all()

    return {
        "topLayer": None,
        "nodeCount": len(items),
        "nodesPerLayer": [],
        "edgesPerLayer": [],
        "nodes": [
            {
                "id": item["id"],
                "metadata": item["metadata"],
                "category": item["category"],
                "maxLyr": None,
            }
            for item in items
        ],
        "edges": [],
        "graphAvailable": False,
    }


@app.get("/stats")
def stats():
    return {
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    }


@app.post("/doc/insert")
async def doc_insert(request: Request):
    body = await request.json()
    title = body.get("title", "")
    text = body.get("text", "")

    if not isinstance(title, str) or not title.strip():
        raise HTTPException(400, "Need title and text")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Need title and text")

    chunks = chunk_text(text, 250, 30)
    ids = []

    try:
        # Embed every chunk first. Avoid partially inserting a document
        # if Ollama fails halfway through embedding.
        embedded = []

        for chunk in chunks:
            vector = ollama.embed(chunk)
            if not vector:
                raise HTTPException(503, "Ollama embedding failed")
            embedded.append(vector)

        for i, (chunk, vector) in enumerate(zip(chunks, embedded)):
            chunk_title = (
                f"{title} [{i + 1}/{len(chunks)}]"
                if len(chunks) > 1 else title
            )
            ids.append(doc_db.insert(chunk_title, chunk, vector))

    except ValueError as exc:
        # Remove chunks already committed if dimensions or input fail.
        for item_id in ids:
            doc_db.remove(item_id)
        raise HTTPException(400, str(exc))

    return {
        "ids": ids,
        "chunks": len(chunks),
        "dims": doc_db.get_dims(),
    }


@app.delete("/doc/delete/{item_id}")
def doc_delete(item_id: int):
    return {"ok": doc_db.remove(item_id)}


@app.get("/doc/list")
def doc_list():
    result = []

    for item in doc_db.all():
        text = item["text"]
        result.append({
            "id": item["id"],
            "title": item["title"],
            "preview": text[:120] + ("…" if len(text) > 120 else ""),
            "words": len(text.split()),
        })

    return result


@app.post("/doc/search")
async def doc_search(request: Request):
    body = await request.json()
    question = require_document_question(body)
    k = max(1, min(int(body.get("k", 3)), 100))

    query_embedding = ollama.embed(question)
    if not query_embedding:
        raise HTTPException(503, "Ollama embedding failed")

    try:
        hits = doc_db.search(query_embedding, k)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return {
        "contexts": [
            {
                "id": item["id"],
                "title": item["title"],
                "distance": round(distance, 4),
            }
            for distance, item in hits
        ]
    }


@app.post("/doc/ask")
async def doc_ask(request: Request):
    body = await request.json()
    question = require_document_question(body)
    k = max(1, min(int(body.get("k", 3)), 100))

    query_embedding = ollama.embed(question)
    if not query_embedding:
        raise HTTPException(503, "Ollama embedding failed")

    try:
        hits = doc_db.search(query_embedding, k)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    context = "\n\n".join(
        f"[{i}] {item['title']}:\n{item['text']}"
        for i, (distance, item) in enumerate(hits, start=1)
    )

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use relevant information from the retrieved documents. "
        "If they do not contain the answer, use your general knowledge. "
        "Do not refer to the retrieved text as 'the context'.\n\n"
        f"Retrieved documents:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    answer = ollama.generate(prompt)

    return {
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {
                "id": item["id"],
                "title": item["title"],
                "text": item["text"],
                "distance": round(distance, 4),
            }
            for distance, item in hits
        ],
        "docCount": doc_db.size(),
    }


@app.get("/status")
def status():
    return {
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    }
