from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import PyPDF2
import io

from app.agent import run_agent
from app.rag import (
    store_document,
    query_rag_stream,
    chat_memory
)

from app.database import SessionLocal
from app.models import DocumentMetadata, ChatLog

app = FastAPI(
    title="Academic Agentic AI System",
    version="2.0"
)

# =====================================================
# REQUEST SCHEMA
# =====================================================

class QuestionRequest(BaseModel):
    question: str
    filename: str | None = None


class AgentRequest(BaseModel):
    query: str


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            raise ValueError("PDF tidak mengandung teks.")

        return text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca PDF: {str(e)}")


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
    return {
        "status": "Agentic AI Academic System Running 🚀"
    }


# =====================================================
# UPLOAD PDF + SAVE METADATA
# =====================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    contents = await file.read()
    text = extract_text_from_pdf(contents)

    # Simpan ke Vector DB (RAG)
    document_id = store_document(text, file.filename)

    # Hitung jumlah chunk sederhana
    chunk_count = len(text.split("\n\n"))

    # Simpan metadata ke PostgreSQL
    db = SessionLocal()
    try:
        metadata = DocumentMetadata(
            filename=file.filename,
            uploaded_by="system",  # nanti bisa diganti user login
            chunk_count=chunk_count,
            embedding_model="your-embedding-model"
        )

        db.add(metadata)
        db.commit()

    finally:
        db.close()

    return {
        "message": "Dokumen berhasil disimpan",
        "document_id": document_id,
        "filename": file.filename
    }


# =====================================================
# ASK QUESTION + SAVE CHAT LOG
# =====================================================

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    def generate():

        full_answer = ""

        stream, metas = query_rag_stream(
            question=request.question,
            filename=request.filename
        )

        for chunk in stream:
            if "message" in chunk:
                content = chunk["message"]["content"]
                full_answer += content
                yield content

        # Simpan ke memory (runtime)
        chat_memory.append({
            "role": "user",
            "content": request.question
        })

        chat_memory.append({
            "role": "assistant",
            "content": full_answer
        })

        # 🔥 SAVE TO DATABASE (Chat Log)
        db = SessionLocal()
        try:
            log = ChatLog(
                user_query=request.question,
                ai_response=full_answer,
                related_document=request.filename
            )

            db.add(log)
            db.commit()

        finally:
            db.close()

        # Tambahkan sumber
        yield "\n\n---\nSources:\n"
        for meta in metas:
            yield f"- {meta['filename']}\n"

    return StreamingResponse(generate(), media_type="text/plain")


# =====================================================
# RESET MEMORY
# =====================================================

@app.post("/reset-memory")
def reset_memory():
    chat_memory.clear()
    return {"message": "Chat memory cleared."}


# =====================================================
# AGENT ENDPOINT
# =====================================================

@app.post("/agent")
def agent_endpoint(request: AgentRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    result = run_agent(request.query)

    return {"response": result}