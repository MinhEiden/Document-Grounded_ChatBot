from langchain_ollama import ChatOllama


def generate_answer(question: str, context: str) -> str:
    """Sinh câu trả lời dựa trên context đã truy xuất (RAG)."""
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    if not context.strip():
        prompt = (
            "Bạn là trợ lý học thuật thông minh. Hiện tại không có tài liệu ngữ cảnh nào được tìm thấy. "
            "Hãy BẮT BUỘC bắt đầu câu trả lời của bạn bằng đúng dòng chữ sau: "
            "\"⚠️ Tôi không tìm thấy dữ liệu từ file của bạn cung cấp, dưới đây chỉ là thông tin tham khảo:\" "
            "Sau đó, hãy cố gắng trả lời câu hỏi dưới đây dựa trên kiến thức sẵn có của bạn.\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )
    else:
        prompt = (
            "Bạn là trợ lý học thuật thông minh. Dựa vào ngữ cảnh được cung cấp, "
            "hãy trả lời câu hỏi một cách chính xác và chi tiết. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )
        
    response = llm.invoke(prompt)
    return response.content.strip()
