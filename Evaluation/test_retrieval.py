import json
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# Đảm bảo Python có thể tìm thấy thư mục gốc chứa các thư mục module
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Load biến môi trường từ file .env ở thư mục gốc
load_dotenv(os.path.join(root_dir, ".env"))

# Đọc các module thật từ hệ thống
from retrieval_pipeline.query_rewriter import rewrite_query
from retrieval_pipeline.retriever import search

def evaluate_retrieval(ground_truth_path="Evaluation/ground_truth.json"):
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ File log không tồn tại: {ground_truth_path}")
        return
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    total_cases = len(test_cases)
    if total_cases == 0:
        print("❌ File ground_truth.json rỗng!")
        return
        
    hits_at_10 = 0
    
    print(f"🚀 Bắt đầu đánh giá trên {total_cases} test cases...\n")
    
    for case in tqdm(test_cases, desc="Đánh giá Retrieval", unit="case"):
        question = case.get("question", "")
        expected_chunk_id = case.get("chunk_id", "")
        expected_content = case.get("chunk_content", "")
        
        if not question:
            continue

        rewritten_q = rewrite_query(question, [])
        
        # Bước 2: Hybrid Search (Lấy ra 10 kết quả tốt nhất từ Vector + BM25 để đo Hits@10)
        searched_docs = search(rewritten_q, k=5)
        
        # Hàm kiểm tra Hit match chung
        def is_hit(docs, expected_id, expected_text):
            for doc in docs:
                doc_chunk_id = doc.metadata.get("chunk_id", "")
                
                if doc_chunk_id == expected_id:
                    return True
                if expected_text and expected_text in doc.page_content:
                    return True
            return False
            
        # Kiểm tra Hits@10 (Kiểm tra hết 10 tài liệu sau Hybrid Search)
        if is_hit(searched_docs[:10], expected_chunk_id, expected_content):
            hits_at_10 += 1

    # Tính toán Recall (% trên tổng số bộ câu hỏi)
    recall_at_10 = (hits_at_10 / total_cases) * 100

    # In ra Report chi tiết
    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRUY XUẤT (RECALL)")
    print("="*50)
    print(f"🔸 Tổng số Test Cases: {total_cases}")
    print(f"🔸 Hits@10: {hits_at_10} / {total_cases}")
    print("-" * 50)
    print(f"✅ Recall@10: {recall_at_10:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(current_dir, "ground_truth.json")
    
    evaluate_retrieval(ground_truth_path=gt_path)