import os
import dotenv
import streamlit as st
import openai
from qdrant_client import QdrantClient

# Import your pipeline modules
from retrival.hybrid_document_retrival import initialize_retrievers
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline

dotenv.load_dotenv()

# Build the layout and appearance
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖")
st.title("🤖 RAG PDF Chatbot")

# Streamlit caches this so that it doesn't re-embed or re-ingest documents on every chat interaction
@st.cache_resource
def init_rag_pipeline():
    # Retrieve Qdrant URL dynamically
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    
    # Ingest pipeline
    vector_store, documents = ingest_pipeline(client)
    
    # Initialization of retriever
    hybrid_retriever = initialize_retrievers(vector_store, documents, 20)
    
    return client, vector_store, documents, hybrid_retriever

# Show a loading spinner during heavy startup operations
with st.spinner("Initializing the RAG Pipeline..."):
    client, vector_store, documents, hybrid_retriever = init_rag_pipeline()


# Session state is used to keep track of the conversation across app reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input from the user
prompt = st.chat_input("Enter your query ➡️")

if prompt:
    # 1. Display user's question and add to history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get and display the bot's streamed response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Ask the generator for chunks of the response
            response_generator = query_pipeline(
                vector_store, 
                prompt, 
                documents, 
                client, 
                hybrid_retriever
            )
            
            # Display incoming chunks in an animated fashion
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Print the final result without the cursor
            message_placeholder.markdown(full_response)
            
        except openai.APITimeoutError:
            full_response = "❌ Timeout Error: OpenAI took too long to respond."
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"❌ Error generating response: {e}"
            message_placeholder.error(full_response)

    # 3. Add bot answer to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
