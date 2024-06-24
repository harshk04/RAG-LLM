from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import ollama

# Initialize the embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant client
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="books")

# Perform a semantic search
query = "Write about the book Years of Solitude"
docs = db.similarity_search_with_score(query=query, k=5)

# Prepare the context for the Llama3 model
context = "\n".join([doc.page_content for doc, score in docs])

# Generate a response using Llama3
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': f"Using the following context, write about the book 'Years of Solitude':\n{context}",
  },
])

# Print the response from Llama3
print(response['message']['content'])
