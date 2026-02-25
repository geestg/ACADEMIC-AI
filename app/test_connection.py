from database import SessionLocal
from models import Document

db = SessionLocal()

try:
    documents = db.query(Document).all()
    print("Koneksi berhasil.")
    print("Jumlah data:", len(documents))
except Exception as e:
    print("Terjadi kesalahan:", e)
finally:
    db.close()