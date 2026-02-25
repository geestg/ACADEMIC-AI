from langchain_ollama import ChatOllama
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from app.rag import retrieve_context


# ======================
# LLM
# ======================
llm = ChatOllama(
    model="llama3",
    temperature=0
)


# ======================
# TOOLS
# ======================
def rag_tool(query: str) -> str:
    context, _ = retrieve_context(query)
    return context


def summarizer(text: str) -> str:
    prompt = f"Ringkas materi berikut:\n{text}"
    return llm.invoke(prompt).content


def question_generator(text: str) -> str:
    prompt = f"Buatkan 5 soal pilihan ganda beserta jawaban dari materi berikut:\n{text}"
    return llm.invoke(prompt).content


tools = [
    Tool(
        name="RAG_Search",
        func=rag_tool,
        description="Cari informasi dari dokumen yang diupload."
    ),
    Tool(
        name="Summarizer",
        func=summarizer,
        description="Ringkas teks."
    ),
    Tool(
        name="Question_Generator",
        func=question_generator,
        description="Buat soal dari materi."
    ),
]


# ======================
# AGENT
# ======================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def run_agent(query: str):
    return agent.run(query)