from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
index = SearchIndex.from_existing("rag_index", redis_url="redis://localhost:6379")

# Function for context retrieval of the document contents.
def get_context_for_question(user_question: str, num_results: int = 3) -> str:
    """Retrieve relevant context from Redis for a given question."""
    query_vector = vectorizer.embed(user_question)
    query = VectorQuery(
        vector=query_vector,
        vector_field_name="embedding",
        num_results=num_results,
        return_fields=["content"],
        return_score=True,
    )

    # Generate vector query in the Redis vector database
    results = index.query(query)
    context_text = "\n".join([doc["content"] for doc in results])
    return context_text


if __name__ == "__main__":
    demo_question = "What are the core features of the project?"
    context_text = get_context_for_question(demo_question)
    print("--- RETRIEVED CONTEXT ---")
    print(context_text)