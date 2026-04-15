from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def rewrite_query(query: str, chat_history: list = None) -> str:
    if not chat_history:
        return query

    llm = ChatOllama(model="llama3.2", temperature=0)
    
    history_lines = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    history_str = "\n".join(history_lines)
    
    prompt_text = f"""Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question in Vietnamese which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. 

Chat History:{history_str}

Latest user question: {query}
Standalone question:"""

    messages = [HumanMessage(content=prompt_text)]
    
    response = llm.invoke(messages)
    rewritten_query = response.content.strip()
    
    if not rewritten_query:
        rewritten_query = query
        
    print(f"\n[Query Rewriter] Câu hỏi gốc: {query}")
    print(f"[Query Rewriter] Câu hỏi đã rewrite: {rewritten_query}\n")

    return rewritten_query
