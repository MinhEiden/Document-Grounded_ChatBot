import os
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def retrieve_context(query: str, k: int = 5) -> list[Document]:
    """Truy xuất context liên quan từ ChromaDB kết hợp Hybrid Search và Reranking."""
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    
    # 1. Lấy 15 chunk cao nhất bằng Vector Search (Cosine Similarity)
    vector_docs = db.similarity_search(query, k=15)
    
    # 2. Lấy 15 chunk cao nhất bằng Keyword Search (BM25)
    db_data = db.get()
    all_documents = []
    if db_data and "ids" in db_data:
        for i in range(len(db_data['ids'])):
            all_documents.append(
                Document(
                    page_content=db_data['documents'][i], 
                    metadata=db_data['metadatas'][i], 
                    id=db_data['ids'][i]
                )
            )
            
    if all_documents:
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 15
        bm25_docs = bm25_retriever.invoke(query)
    else:
        bm25_docs = []
        
    # 3. Kết hợp và bỏ kết quả trùng lặp
    unique_docs = []
    seen_contents = set()
    
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
            
    if not unique_docs:
        return []
        
    # 4. Cohere Reranker: nhận 30 chunk và đánh giá lại độ phù hợp
    cohere_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=k)
    
    # compress_documents trả về danh sách Document nhưng có thuộc tính "relevance_score" trong metadata
    reranked_docs = cohere_rerank.compress_documents(query=query, documents=unique_docs)
    
    # Lọc những Document có relevance_score >= 0.35
    filtered_docs = []
    for doc in reranked_docs:
        score = doc.metadata.get("relevance_score", 0.0)
        if score >= 0.35:
            filtered_docs.append(doc)
            
    return filtered_docs
