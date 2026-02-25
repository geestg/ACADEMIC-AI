from openai import OpenAI
from app.tools.rag_tool import retrieve_context

client = OpenAI()

def run_summarizer(vectorstore, question: str):
    context = retrieve_context(vectorstore, question)

    prompt = f"""
    Anda adalah asisten akademik.
    Ringkas materi berikut secara sistematis dan ilmiah.

    Materi:
    {context}

    Berikan:
    1. Ringkasan singkat
    2. Poin-poin penting
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content