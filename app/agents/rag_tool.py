from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve_context(vectorstore, query: str):
    retriever = get_retriever(vectorstore)
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])