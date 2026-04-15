import json
import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

load_dotenv(os.path.join(root_dir, ".env"))

from retrieval_pipeline.query_rewriter import rewrite_query
from retrieval_pipeline.retriever import search, rerank

def evaluate_rerank(ground_truth_path="Evaluation/ground_truth.json"):
    if not os.path.exists(ground_truth_path):
        print(f"❌ File log không tồn tại: {ground_truth_path}")
        return
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    total_cases = len(test_cases)
    if total_cases == 0:
        print("❌ File ground_truth.json rỗng!")
        return
        
    total_rr = 0.0
    valid_cases = 0
    
    print(f"🚀 Bắt đầu đánh giá Rerank (MRR) trên {total_cases} test cases...\n")
    
    for case in tqdm(test_cases, desc="Đánh giá Rerank", unit="case"):
        question = case.get("question", "")
        expected_chunk_id = case.get("chunk_id", "")
        expected_content = case.get("chunk_content", "")
        
        if not question:
            continue
            
        valid_cases += 1

        rewritten_q = rewrite_query(question, [])
        
    
        searched_docs = search(rewritten_q, k=10)
        
        reranked_docs = rerank(rewritten_q, searched_docs, k=3)
        
        rr = 0.0
        for rank, doc in enumerate(reranked_docs, start=1):
            doc_chunk_id = doc.metadata.get("chunk_id", "")
            
            if doc_chunk_id == expected_chunk_id or (expected_content and expected_content in doc.page_content):
                rr = 1.0 / rank
                break
                
        total_rr += rr
        time.sleep(4)

    mrr = (total_rr / valid_cases) if valid_cases > 0 else 0.0

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ RERANKER (MRR)")
    print("="*50)
    print(f"🔸 Tổng số Test Cases hợp lệ: {valid_cases}")
    print(f"🔸 Tổng điểm Reciprocal Rank (Sum RR): {total_rr:.4f}")
    print("-" * 50)
    print(f"✅ Mean Reciprocal Rank (MRR): {mrr:.4f} (Max: 1.0)")
    print("="*50 + "\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(current_dir, "ground_truth.json")
    
    evaluate_rerank(ground_truth_path=gt_path)