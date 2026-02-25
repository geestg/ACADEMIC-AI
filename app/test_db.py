# test_db.py
from app.database import engine

try:
    conn = engine.connect()
    print("Koneksi database berhasil")
    conn.close()
except Exception as e:
    print("Terjadi kesalahan:", e)