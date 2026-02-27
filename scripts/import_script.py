import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

# Set up the necessary files into vector database
loader = PyPDFLoader("your_file_goes_here.pdf") # Replace with your file path
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# Simple schema for RAG indexing. Uses cosine similarity score for the vector model searching
schema = {
    "index": {
        "name": "rag_index",
        "prefix": "doc"
    },
    "fields": [
        {"name": "content", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 384, # Dimensions for 'all-MiniLM-L6-v2'
                "algorithm": "hnsw",
                "distance_metric": "cosine"
            }
        }
    ]
}

index = SearchIndex.from_dict(schema, redis_url="redis://localhost:6379")
index.create(overwrite=True)

vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
data = []

for i, chunk in enumerate(chunks):
    # Convert embedding list to bytes for Redis
    embedding = np.array(
        vectorizer.embed(chunk.page_content), dtype="float32"
    ).tobytes()
    data.append({
        "id": f"doc:{i}",
        "content": chunk.page_content,
        "embedding": embedding
    })

index.load(data)
print(f"Successfully loaded {len(data)} chunks into Redis!")