import os
import json
import random
import uuid
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

def load_all_chunks() -> list[Document]:
    """
    Load thực tế toàn bộ chunks từ database ChromaDB.
    """
    load_dotenv()
    chroma_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    if os.getenv("CHROMA_DB_PATH"):
        chroma_path = os.getenv("CHROMA_DB_PATH")
    elif os.path.exists("./chroma_db"):
        chroma_path = "./chroma_db"

    print(f" Đang kết nối tới ChromaDB tại: {chroma_path}")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    
    db_data = db.get()
    documents = []
    if db_data and "ids" in db_data:
        for i in range(len(db_data['ids'])):
            metadata = db_data['metadatas'][i] if db_data['metadatas'][i] else {}
            # Lấy luôn ID của Chroma làm chunk_id
            metadata["chunk_id"] = db_data['ids'][i]
            documents.append(
                Document(
                    page_content=db_data['documents'][i], 
                    metadata=metadata
                )
            )
    print(f"Đã tải thành công {len(documents)} chunks từ database.")
    return documents

def generate_ground_truth():
    print("Loading chunks...")
    all_chunks = load_all_chunks()
    
    filtered_chunks = all_chunks[3:] if len(all_chunks) > 3 else all_chunks
    
    sample_size = min(40, len(filtered_chunks))
    if sample_size == 0:
        print("❌ Not enough chunks to process.")
        return
        
    sampled_chunks = random.sample(filtered_chunks, sample_size)
    print(f" Sampled {sample_size} random chunks.")

    print("Initializing LLM (ChatOllama - llama3.2)...")
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Act as a curious student reading a textbook. Read the following text chunk and generate exactly ONE short, specific question that this text perfectly answers. Do not ask meta-questions like 'What does this paragraph discuss?'. Ask about the actual concepts (e.g., 'What is abstract labor?'). Output ONLY the question, nothing else. Please output with the same language as the given text"),
        ("user","Text chunk:\n\n{text}")
    ])
    
    chain = prompt | llm
   
    eval_dir = os.path.join(os.path.dirname(__file__)) if os.path.basename(os.path.dirname(__file__)) == 'Evaluation' else os.path.join(os.getcwd(), 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    output_path = os.path.join(eval_dir, "ground_truth.json")
    
    ground_truth_data = []


    for chunk in tqdm(sampled_chunks, desc="Generating synthetic questions", unit="chunk"):
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            chunk_id = uuid.uuid4().hex
            chunk.metadata["chunk_id"] = chunk_id
            
        original_text = chunk.page_content
        
        response = chain.invoke({"text": original_text})
        
        generated_question = response.content.strip().strip("\"'").strip()
        
        ground_truth_data.append({
            "question": generated_question,
            "chunk_id": chunk_id,
            "chunk_content": original_text
        })

    # Export to JSON
    print(f" Saving generated dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully created Ground Truth dataset with {len(ground_truth_data)} question-answer pairs!")

if __name__ == "__main__":
    generate_ground_truth()
