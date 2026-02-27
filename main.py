from langchain_ollama import OllamaLLM
from scripts.retrieval_script import get_context_for_question

llm = OllamaLLM(model="qwen2.5-coder:7b")


def generate_local_answer(question: str, context: str) -> str:
    prompt = f"""
You are an efficient personal academic assistant. Use the provided context to answer the
user's question. If the answer cannot be provided in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:
"""

    response = llm.invoke(prompt)
    return response


if __name__ == "__main__":
    user_question = input("Enter your question: ")
    context = get_context_for_question(user_question)
    final_answer = generate_local_answer(user_question, context)
    print("\n--- FINAL ANSWER ---\n")
    print(final_answer)