import os
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from utils.config import get_config


def rewrite_query(query: str, chat_history: list = None) -> str:
    """
    Viết lại câu hỏi để tối ưu cho việc truy xuất.
    - Nếu không có chat_history: Trả về nguyên câu hỏi mộc.
    - Nếu có chat_history (đã được giới hạn): Gọi LLM mượn ngữ cảnh trước đó để biến
      thành câu hỏi độc lập (standalone question).
    """
    if not chat_history:
        return query

    gemini_api_key = get_config("GEMINI_API_KEY")
    if not gemini_api_key:
        warning_msg = "[query_rewriter] Thieu GEMINI_API_KEY. Bo qua buoc rewrite va dung cau hoi goc."
        print(f"⚠️ {warning_msg}")
        warnings.warn(warning_msg, RuntimeWarning)
        return query

    llm = ChatGoogleGenerativeAI(
        model=get_config("GEMINI_MODEL", "gemini-1.5-flash"),
        temperature=0,
        google_api_key=gemini_api_key,
    )
    
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
