import os
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from utils.config import get_config

def retrieve_context(query: str, session_id: str, k: int = 5) -> list[Document]:
    """Truy xuất context liên quan cho đúng session (Hybrid + Rerank)."""
    chroma_path = get_config("CHROMA_DB_PATH", "./vector_store")

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name="academic_chatbot",
    )

    filters = {"session_id": session_id}

    vector_docs = db.similarity_search(query, k=15, filter=filters)

    db_data = db.get(where=filters)
    all_documents = []
    if db_data and "ids" in db_data:
        for i in range(len(db_data["ids"])):
            all_documents.append(
                Document(
                    page_content=db_data["documents"][i],
                    metadata=db_data["metadatas"][i],
                    id=db_data["ids"][i],
                )
            )

    if all_documents:
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 15
        bm25_docs = bm25_retriever.invoke(query)
    else:
        bm25_docs = []

    unique_docs = []
    seen_contents = set()
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    cohere_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=k)
    reranked_docs = cohere_rerank.compress_documents(query=query, documents=unique_docs)

    filtered_docs = []
    for doc in reranked_docs:
        score = doc.metadata.get("relevance_score", 0.0)
        if score >= 0.35:
            filtered_docs.append(doc)

    return filtered_docs
