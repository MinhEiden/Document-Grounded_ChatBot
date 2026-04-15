from langchain_ollama import ChatOllama


def generate_answer(question: str, context: str, chat_history: list = None) -> str:
    """Sinh câu trả lời dựa trên context đã truy xuất (RAG)."""
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_lines.append(f"{role}: {msg['content']}")
        history_str = "\n".join(history_lines)

    if not context.strip():
        prompt = (
            "Bạn là trợ lý học thuật thông minh. Hiện tại không có tài liệu ngữ cảnh nào được tìm thấy. "
            "Hãy BẮT BUỘC bắt đầu câu trả lời của bạn bằng đúng dòng chữ sau: "
            "\"⚠️ Tôi không tìm thấy dữ liệu từ file của bạn cung cấp, dưới đây chỉ là thông tin tham khảo:\" "
            "Sau đó, hãy cố gắng trả lời câu hỏi dưới đây dựa trên kiến thức sẵn có của bạn.\n\n"
        )
        if history_str:
            prompt += f"Lịch sử trò chuyện gần đây:\n{history_str}\n\n"
            
        prompt += (
            f"Câu hỏi hiện tại: {question}\n\n"
            "Trả lời:"
        )
    else:
        prompt = (
            "Bạn là trợ lý học thuật thông minh. Dựa vào tài liệu được cung cấp, "
            "hãy trả lời câu hỏi hiện tại một cách chính xác và chi tiết. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ.\n\n"
        )
        if history_str:
            prompt += f"Lịch sử trò chuyện gần đây:\n{history_str}\n\n"
            
        prompt += (
            f"Tài Liệu được cung cấp:\n{context}\n\n"
            f"Câu hỏi hiện tại: {question}\n\n"
            "Trả lời:"
        )
        
    response = llm.invoke(prompt)
    return response.content.strip()
