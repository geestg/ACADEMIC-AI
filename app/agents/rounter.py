def route_question(question: str) -> str:
    q = question.lower()

    if any(keyword in q for keyword in ["ringkas", "summary", "inti"]):
        return "summarizer"

    if any(keyword in q for keyword in ["soal", "latihan", "kuis"]):
        return "question_generator"

    return "default"