from openai import OpenAI
from app.tools.rag_tool import retrieve_context

client = OpenAI()

def run_question_generator(vectorstore, question: str):
    context = retrieve_context(vectorstore, question)

    prompt = f"""
    Berdasarkan materi berikut, buatkan 5 soal pilihan ganda
    tingkat kesulitan sedang dan sertakan kunci jawaban.

    Materi:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content