import json
import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

load_dotenv(os.path.join(root_dir, ".env"))

from retrieval_pipeline.query_rewriter import rewrite_query
from retrieval_pipeline.retriever import search, rerank
from retrieval_pipeline.generator import generate_answer

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class FaithfulnessOutput(BaseModel):
    reasoning: str = Field(description="Lý do chi tiết giải thích vì sao câu trả lời có chứa ảo giác, hay hoàn toàn trung thành với tài liệu.")
    score: int = Field(description="Điểm trung thành: Trả về 1 nếu câu trả lời TUYỆT ĐỐI trung thành với Context (không tự phang thêm ý ngoài). Nếu có thông tin nào không nằm trong Context, cho 0.")

class RelevanceOutput(BaseModel):
    reasoning: str = Field(description="Lý do giải thích vì sao câu trả lời có trúng trọng tâm câu hỏi hay không.")
    score: int = Field(description="Điểm liên quan: Trả về 1 nếu trả lời ĐÚNG VÀ ĐỦ ý câu hỏi. Nếu lan man, lạc đề hoặc nói 'không biết' không hợp lý, cho 0.")

def evaluate_generation(ground_truth_path="ground_truth.json", output_path="generation_results.json"):
    if not os.path.exists(ground_truth_path):
        print(f"❌ File ground truth không tồn tại: {ground_truth_path}")
        return
        
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    if not test_cases:
        print("❌ File ground truth rỗng!")
        return

    llm_judge = ChatOllama(
        model="llama3.2",
        temperature=0.0
    )
    
    judge_faithfulness = llm_judge.with_structured_output(FaithfulnessOutput)
    judge_relevance = llm_judge.with_structured_output(RelevanceOutput)

    prompt_f = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một giám khảo nghiêm khắc. Hãy đánh giá tính 'Trung thành' (Faithfulness). Đọc Context sau đó đọc Answer. Xem xét Answer có bịa đặt ra bất kỳ sự thật nào KHÔNG CÓ trong Context hay không. Cấm dùng kiến thức ngoài lề. Đánh giá 0 (có kiến thức rác/bịa đặt) hoặc 1 (tuyệt đối chỉ dùng Context). Giải thích lý do trong reasoning. Trả lời bằng tiếng Việt."),
        ("user", "Question: {question}\n\nContext (Ngữ cảnh cung cấp):\n{context}\n\nAnswer (Câu trả lời hệ thống sinh): {answer}\n\nHãy xuất ra điểm số và giải thích.")
    ])

    prompt_r = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một giám khảo khách quan. Hãy đánh giá 'Mức độ trúng đích' (Relevance) của Answer so với Question. Answer có giải quyết được ý đồ của câu hỏi không? Trả lời có trực tiếp không? Đánh giá 1 nếu trúng đích, 0 nếu lạc đề. Trả lời bằng tiếng Việt."),
        ("user", "Question: {question}\n\nAnswer (Câu trả lời hệ thống sinh): {answer}\n\nHãy xuất ra điểm số và giải thích.")
    ])

    total_faithfulness = 0
    total_relevance = 0
    valid_cases = len(test_cases)
    results_list = []

    print(f"🚀 Bắt đầu đánh giá GENERATION trên {valid_cases} test cases...\n")

    for case in tqdm(test_cases, desc="Đánh giá Generator", unit="case"):
        question = case.get("question", "")
        
        rewritten_q = rewrite_query(question, [])
        searched_docs = search(rewritten_q, k=10)
        reranked_docs = rerank(rewritten_q, searched_docs, k=5)
        
        context_str = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        answer = generate_answer(question, context_str)
        
        try:
            eval_f = judge_faithfulness.invoke(prompt_f.format(question=question, context=context_str, answer=answer))
            score_f = eval_f.score
            reason_f = eval_f.reasoning
        except Exception as e:
            print(f"\n⚠️ Lỗi khi chấm Faithfulness: {e}")
            score_f, reason_f = 0, str(e)
            
        try:
            eval_r = judge_relevance.invoke(prompt_r.format(question=question, answer=answer))
            score_r = eval_r.score
            reason_r = eval_r.reasoning
        except Exception as e:
            print(f"\n⚠️ Lỗi khi chấm Relevance: {e}")
            score_r, reason_r = 0, str(e)

        total_faithfulness += score_f
        total_relevance += score_r
        
        results_list.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": score_f,
            "faithfulness_reason_from_judge": reason_f,
            "relevance_score": score_r,
            "relevance_reason_from_judge": reason_r,
        })
        
        time.sleep(1.5)

    avg_f = (total_faithfulness / valid_cases) * 100 if valid_cases > 0 else 0
    avg_r = (total_relevance / valid_cases) * 100 if valid_cases > 0 else 0

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG SINH VĂN BẢN (GENERATION)")
    print("="*60)
    print(f"🔸 Tổng Test Cases: {valid_cases}")
    print("-" * 60)
    print(f"Faithfulness (Trung thành): {avg_f:.2f}% ({total_faithfulness}/{valid_cases})")
    print(f"Answer Relevance (Liên quan): {avg_r:.2f}% ({total_relevance}/{valid_cases})")
    print(f"Log chi tiết đã lưu: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(current_dir, "ground_truth.json")
    out_path = os.path.join(current_dir, "generation_results.json")
    
    evaluate_generation(ground_truth_path=gt_path, output_path=out_path)
