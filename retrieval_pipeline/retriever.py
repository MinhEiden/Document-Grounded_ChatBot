import os
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pyvi import ViTokenizer

def load_stopwords(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

vi_stopwords_path = os.path.join(os.path.dirname(__file__), "Vietnamese_stopword.txt")
en_stopwords_path = os.path.join(os.path.dirname(__file__), "English_stopword.txt")
COMBINED_STOPWORDS = load_stopwords(vi_stopwords_path) | load_stopwords(en_stopwords_path)

def hybrid_preprocess_func(text: str) -> list[str]:
    tokens = ViTokenizer.tokenize(text.lower()).split()
    return [token for token in tokens if token not in COMBINED_STOPWORDS]

def search(query: str, k: int = 20) -> list[Document]:
    """Thực hiện Hybrid Search (Vector + BM25) để lấy danh sách context liên quan từ ChromaDB."""
    if not query or not query.strip():
        return []

    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    
    vector_docs = db.similarity_search(query, k=k)
    
    print("\n" + "="*40)
    print(f"📊 VECTOR SEARCH RESULTS (Top {len(vector_docs)} chunks)")
    for i, doc in enumerate(vector_docs):
        content_preview = doc.page_content.replace("\n", " ")[:150]
        print(f"  [{i+1}] {content_preview}...")
    

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
        bm25_retriever = BM25Retriever.from_documents(all_documents, preprocess_func=hybrid_preprocess_func)
        bm25_retriever.k = k
        
        # query_tokens = hybrid_preprocess_func(query)
        # print("\n" + "="*40)
        # # print(f"🔑 BM25 QUERY TOKENS: {query_tokens}")
        
        bm25_docs = bm25_retriever.invoke(query)
    else:
        bm25_docs = []
        
    print("\n" + "="*40)
    print(f"🔤 KEYWORD SEARCH (BM25) RESULTS (Top {len(bm25_docs)} chunks)")
    for i, doc in enumerate(bm25_docs):
        content_preview = doc.page_content.replace("\n", " ")[:150] 
        print(f"  [{i+1}] {content_preview}...")
        
    unique_docs = []
    seen_contents = set()
    
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
            
    return unique_docs

def rerank(query: str, documents: list[Document], k: int = 3) -> list[Document]:
    """Sắp xếp lại các documents bằng Cohere Rerank và lọc theo relevance_score."""
    if not documents:
        return []
        
    cohere_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=k)
    
    reranked_docs = cohere_rerank.compress_documents(query=query, documents=documents)
    filtered_docs = []
    for doc in reranked_docs:
        score = doc.metadata.get("relevance_score", 0.0)
        if score >= 0.3:
            filtered_docs.append(doc)
            
    print("\n" + "="*40)
    print(f"🔍 RERANKER RESULTS (Top {len(filtered_docs)} chunks >= 0.3)")
    print("="*40)
    for i, doc in enumerate(filtered_docs):
        score = doc.metadata.get("relevance_score", 0.0)
        content_preview = doc.page_content.replace("\n", " ")[:150] 
        print(f"[{i+1}] Score: {score:.4f}")
        print(f"    Content: {content_preview}...")
    print("="*40 + "\n")
            
    return filtered_docs
