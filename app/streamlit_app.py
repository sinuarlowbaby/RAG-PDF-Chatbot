import os
import dotenv

from retrieval.hybrid_document_retrieval import initialize_retrievers
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline
import streamlit as st
import openai

dotenv.load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖")
st.title("🤖 RAG PDF Chatbot")
st.subheader("Powered by Qdrant and GPT-4o")

@st.cache_resource
def init_rag_pipeline():
    try:
        client = QdrantClient(url="http://localhost:6333")
        vector_store,documents = ingest_pipeline(client)
        hybrid_retriver = initialize_retrievers(vector_store,documents,20)
        return client, vector_store, documents, hybrid_retriver
    except Exception as e:
        st.error(f"❌ Error initializing RAG pipeline: {e}")
        st.stop()

# Build the layout and appearance   
try:
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Model:** GPT-4o")
        st.markdown("**Vector DB:** Qdrant")
        st.markdown("**Search:** Hybrid (MMR + BM25)")
        st.divider()
        st.caption("Built with LangChain + OpenAI")


    with st.spinner("Loading documents into vector store..."):
        client, vector_store, documents, hybrid_retriver = init_rag_pipeline()

    st.success("✅ Documents loaded successfully!")


    user_query = st.text_input("☁️ Enter your question here : ")

    if st.button("Ask") and user_query:
        with st.spinner("Generating response..."):
            full_response = ''
            placeholder = st.empty()
            try:
                response_generator = query_pipeline(vector_store, user_query, documents, client, hybrid_retriver)
                st.markdown("### 🤖 ###")
        
                for chunk in response_generator:
                    full_response += chunk
                    placeholder.markdown(full_response +" ▌")
                placeholder.markdown(full_response)

            except openai.APITimeoutError:
                print("\n❌ [Timeout Error]: OpenAI took too long to respond. Please try again.")
            except openai.APIStatusError as e:
                print(f"\n❌ [API Error]: OpenAI returned an error status: {e.status_code}")
            except Exception as e:
                print(f"\n❌ [Unexpected Error]: Something broke during the stream: {e}")
            
except Exception as e:
    st.error(f"❌ Error: {e}")

