import ollama
import chromadb
from chromadb.config import Settings
import uuid

# =========================
# CHROMA SETUP (PERSISTENT)
# =========================
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))

collection = chroma_client.get_or_create_collection(name="documents")

# =========================
# CHAT MEMORY
# =========================
chat_memory = []


# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# =========================
# EMBEDDING
# =========================
def embed_text(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


# =========================
# STORE DOCUMENT (MULTI FILE)
# =========================
def store_document(text, filename):
    document_id = str(uuid.uuid4())
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{document_id}_{i}"],
            metadatas=[{
                "document_id": document_id,
                "filename": filename
            }]
        )

    return document_id


# =========================
# RETRIEVE CONTEXT
# =========================
def retrieve_context(query, k=5, filename=None):
    query_embedding = embed_text(query)

    if filename:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"filename": filename}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

    if not results["documents"]:
        return "", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(docs)

    return context, metas


# =========================
# RAG STREAM WITH MEMORY
# =========================
def query_rag_stream(question, filename=None):
    global chat_memory

    context, metas = retrieve_context(question, filename=filename)

    messages = [
        {
            "role": "system",
            "content": "Answer only based on the provided context."
        }
    ]

    for msg in chat_memory:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    stream = ollama.chat(
        model="llama3",
        messages=messages,
        stream=True
    )

    return stream, metas