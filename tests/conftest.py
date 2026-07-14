import sys
import os
from unittest.mock import MagicMock, patch

# Set mock env variables first so pydantic-settings loads them
os.environ["OPENAI_API_KEY"] = "mock-openai-key"
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["LANGFUSE_SECRET_KEY"] = "mock-langfuse-secret"
os.environ["LANGFUSE_PUBLIC_KEY"] = "mock-langfuse-public"

# Mock sentence_transformers, qdrant_client, redis, langchain_openai, groq, and document_loaders BEFORE app is imported
import sentence_transformers
import qdrant_client
import redis
import langchain_openai
import groq
import langchain_community.document_loaders
from langchain_core.embeddings import Embeddings

class MockCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        import numpy as np
        return np.array([1.0])

sentence_transformers.CrossEncoder = MockCrossEncoder

class MockQdrantClient:
    def __init__(self, *args, **kwargs):
        pass
    def get_collections(self):
        mock_collections = MagicMock()
        mock_collections.collections = []
        return mock_collections
    def get_collection(self, collection_name):
        mock_info = MagicMock()
        # Mock the dense vector config validation
        mock_info.config.params.vectors = MagicMock()
        mock_info.config.params.vectors.size = 1536
        mock_info.config.params.vectors.distance = qdrant_client.models.Distance.COSINE
        return mock_info
    def create_collection(self, *args, **kwargs):
        pass
    def upsert(self, collection_name, points, **kwargs):
        pass
    def query_points(self, *args, **kwargs):
        mock_result = MagicMock()
        mock_result.points = []
        return mock_result
    def scroll(self, *args, **kwargs):
        class MockRecord:
            def __init__(self, payload):
                self.payload = payload
        return ([MockRecord(payload={"page_content": "This is test page content", "metadata": {"file_name": "test.pdf"}})], None)
    def delete(self, *args, **kwargs):
        pass
    def close(self):
        pass

qdrant_client.QdrantClient = MockQdrantClient

class MockRedis:
    def __init__(self, *args, **kwargs):
        pass
    def ping(self):
        return True
    def exists(self, *args, **kwargs):
        return True
    def delete(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass
    def setex(self, *args, **kwargs):
        pass
    def keys(self, *args, **kwargs):
        return []
    def scan_iter(self, match=None, count=None):
        return []

redis.Redis = MockRedis

class MockOpenAIEmbeddings(Embeddings):
    def __init__(self, *args, **kwargs):
        pass
    def embed_query(self, *args, **kwargs):
        return [0.1] * 1536
    def embed_documents(self, *args, **kwargs):
        return [[0.1] * 1536]

langchain_openai.OpenAIEmbeddings = MockOpenAIEmbeddings

class MockPyPDFLoader:
    def __init__(self, file_path, **kwargs):
        pass
    def load(self):
        from langchain_core.documents import Document
        return [Document(page_content="This is mocked document content of at least fifty characters long for test ingestion.", metadata={"source": "test.pdf"})]

langchain_community.document_loaders.PyPDFLoader = MockPyPDFLoader

class MockGroq:
    def __init__(self, *args, **kwargs):
        self.chat = MagicMock()
        
        # Configure chat.completions.create mock
        def mock_create(*args, **kwargs):
            if kwargs.get("stream") is True:
                # Mock chunk stream for _llm_stream
                chunk1 = MagicMock()
                chunk1.choices = [MagicMock()]
                chunk1.choices[0].delta.content = "Answer "
                
                chunk2 = MagicMock()
                chunk2.choices = [MagicMock()]
                chunk2.choices[0].delta.content = "content"
                
                return [chunk1, chunk2]
            else:
                # Mock JSON response for multi-query generate_queries
                response = MagicMock()
                response.choices = [MagicMock()]
                response.choices[0].message.content = '["expanded query 1", "expanded query 2"]'
                return response
        
        self.chat.completions.create = mock_create

groq.Groq = MockGroq

# Add app directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))
