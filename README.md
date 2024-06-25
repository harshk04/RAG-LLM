# LlamaGPT Contextual Search Engine

## Introduction

Welcome to the LlamaGPT Contextual Search Engine! This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using the Llama3 model. The application allows users to interact with a search engine that provides contextual responses based on the content of a PDF document.

## Features

- **PDF Document Ingestion:** Load and split PDF documents into chunks for semantic search.
- **Embedding Generation:** Use HuggingFace's BAAI/bge-large-en model for generating embeddings.
- **Contextual Search:** Perform semantic searches to retrieve relevant document sections.
- **Response Generation:** Use Llama3 to generate responses based on the retrieved context.
- **Streamlit Interface:** User-friendly interface with navigation options for Home, Generate Response, and Contact Us pages.

## 📌 Sneak Peek of Main Page 🙈 :

<img width="700" alt="Screenshot 2024-06-24 at 3 00 04 PM" src="https://github.com/harshk04/RAG-LLM/assets/115946158/04380d3b-fe17-4bf9-b105-db6356d61e21">

<img width="700" alt="Screenshot 2024-06-24 at 2 59 48 PM" src="https://github.com/harshk04/RAG-LLM/assets/115946158/72bcc12a-20c4-42c5-8ea6-736b1a89bd16">

<img width="700" alt="Screenshot 2024-06-24 at 3 00 16 PM" src="https://github.com/harshk04/RAG-LLM/assets/115946158/22d9b677-f665-48cd-a8a7-2d910eea1089">


## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Qdrant
- HuggingFace Transformers
- Ollama
- Streamlit Option Menu

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/harshk04/RAG-LLM
   cd RAG-LLM

2. Install the required packages:
   
  `pip install -r requirements.txt`

3. Ensure Qdrant is running locally:
   
`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

## Usage
### Ingesting Data
1. Place your PDF document in the project directory and name it `Data.pdf`.
2. Run the `ingest.py` script to load the PDF, split it into chunks, and store the embeddings in Qdrant:

`python ingest.py`

### Running the Streamlit App
1. Run the Streamlit app:
   `streamlit run app.py`
2. Open your browser and go to `http://localhost:8501` to access the application.

## Project Structure

- **ingest.py**: Script to load PDF, split text, generate embeddings, and store them in Qdrant.
- **app.py**: Main application script with Streamlit interface for interacting with the search engine.
- **requirements.txt**: List of required Python packages.
- **Untitled.pdf**: Sample PDF document (replace with your own document).

## App Navigation

- **Home**: Overview of the application with an image and welcome message.
- **Generate Response**: Interactive chat interface to generate responses based on user input and document context.
- **Contact Us**: Form to contact the developer.

### License

This project is licensed under the `MIT License`.


### 📬 Contact


If you want to contact me, you can reach me through below handles.

&nbsp;&nbsp;<a href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="30"></img></a>

© 2024 Harsh

