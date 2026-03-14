from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def rewrite_query(query: str, chat_history: list = None) -> str:
    """
    Viết lại câu hỏi để tối ưu cho việc truy xuất.
    - Nếu không có chat_history: Trả về nguyên câu hỏi mộc.
    - Nếu có chat_history (đã được giới hạn): Gọi LLM mượn ngữ cảnh trước đó để biến
      thành câu hỏi độc lập (standalone question).
    """
    if not chat_history:
        return query

    # Dùng llama3.2 (hoặc gemma2, qwen2) tùy model bạn đã pull trong Ollama
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Format chat history thành định dạng Langchain hiểu
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        else:
            formatted_history.append(AIMessage(content=msg["content"]))
            
    # Tạo chuỗi hội thoại
    system_prompt = SystemMessage(
        content=(
            "Bạn là trợ lý hệ thống. Dựa vào lịch sử hội thoại dưới đây và câu hỏi mới nhất, "
            "hãy viết lại câu hỏi mới nhất thành một câu hỏi độc lập, đầy đủ ngữ cảnh để có thể "
            "tìm kiếm trong cơ sở dữ liệu. ĐỪNG trả lời câu hỏi, CHỈ viết lại câu hỏi. "
            "Nếu câu hỏi mới đã đầy đủ ý nghĩa và không liên quan đến cuộc trò chuyện cũ, "
            "hãy giữ nguyên câu hỏi đó."
        )
    )
    
    # Ghép system prompt + history + câu hỏi mới hiện tại
    messages = [system_prompt] + formatted_history + [HumanMessage(content=f"Câu hỏi mới: {query}")]
    
    response = llm.invoke(messages)
    return response.content.strip()
