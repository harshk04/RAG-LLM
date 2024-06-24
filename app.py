import streamlit as st
from streamlit_option_menu import option_menu  # Make sure to install this package
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

# Streamlit app
st.title("LlamaGPT Contextual Search Engine")
# st.image("img.jpeg", caption="RAG with LLM")  # Replace with a valid image URL

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    page = option_menu(
        "Navigation", 
        ["Home", "Generate Response", "Contact Us"],
        icons=["house", "search", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

if page == "Home":
    

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("img.jpeg", width=600, caption="RAG with LLM", use_column_width=True)  # Replace with a valid image URL
    st.subheader("Home")
    st.success("Retrieval-Augmented Generation (RAG) with Llama3")
    st.write("Welcome to the RAG with Llama3 application. Select 'Generate Response' from the menu to get started.")
elif page == "Generate Response":
    st.subheader("Generate a Response")
    query = st.text_area("Enter your query:", height=200)  # Textarea for user input

    if st.button("Generate"):
        with st.spinner("Generating response..."):
            # Perform a semantic search
            docs = db.similarity_search_with_score(query=query, k=5)

            # Prepare the context for the Llama3 model
            context = "\n".join([doc.page_content for doc, score in docs])

            # Generate a response using Llama3
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': f"Using the following context, write about{query}:\n{context}",
                },
            ])

            # Display the response from Llama3
            st.success("Response generated!")
            st.write(response['message']['content'])

elif page == "Contact Us":
    st.markdown("***")

    st.header("Contact Me")
    st.write("Please fill out the form below to get in touch with me.")

    # Input fields for user's name, email, and message
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message", height=150)

    # Submit button
    if st.button("Submit"):
        if name.strip() == "" or email.strip() == "" or message.strip() == "":
            st.warning("Please fill out all the fields.")
        else:
            send_email_to = 'kumawatharsh2004@gmail.com'
            st.success("Your message has been sent successfully!")

st.sidebar.success("This app demonstrates Retrieval-Augmented Generation (RAG) using the Llama3 model.")
st.sidebar.warning("Developed by [Harsh Kumawat](https://www.linkedin.com/in/harsh-k04/)")