from app.database import SessionLocal
from app.models import ChatLog

def save_chat_log(question, answer, document_name):
    db = SessionLocal()
    log = ChatLog(
        user_query=question,
        ai_response=answer,
        related_document=document_name
    )
    db.add(log)
    db.commit()
    db.close()