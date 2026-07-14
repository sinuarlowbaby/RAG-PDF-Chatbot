import sys
import os
import pytest
from langchain_core.documents import Document

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from ingest import clean_text, doc_chunker
from retrieval.deduplication import deduplication
from retrieval.build_context import build_context
from config import Settings

def test_clean_text():
    # Test collapse whitespace
    assert clean_text("hello   world") == "hello world"
    # Test dot leaders removal
    assert clean_text("Chapter 1. . . . . . 5") == "Chapter 15"
    # Test strip whitespace
    assert clean_text("   clean me   ") == "clean me"

def test_doc_chunker():
    # Test skipped documents when character count < 50
    raw_docs = [
        Document(page_content="short", metadata={"source": "test.pdf"}),
        Document(page_content="This is a long enough document content that should definitely exceed fifty characters and get chunked properly.", metadata={"source": "test.pdf"})
    ]
    chunks = doc_chunker(raw_docs, session_id="test-session")
    
    # The first document has only 5 chars (< 50) and should be skipped.
    # The second document has ~110 chars and should be chunked.
    assert len(chunks) > 0
    assert chunks[0].metadata["session_id"] == "test-session"
    assert chunks[0].metadata["file_name"] == "test.pdf"

def test_config_computed_properties():
    # Test compute properties for Redis URL
    settings = Settings(redis_url="redis://my-redis-host:1234")
    assert settings.redis_host == "my-redis-host"
    assert settings.redis_port == 1234

    settings_default = Settings(redis_url="redis://localhost:6379")
    assert settings_default.redis_host == "localhost"
    assert settings_default.redis_port == 6379

def test_deduplication():
    docs = [
        Document(page_content="Text A", metadata={"source": "doc1.pdf"}),
        Document(page_content="Text a", metadata={"source": "doc1.pdf"}),  # duplicate under normalized comparison
        Document(page_content="Text B", metadata={"source": "doc2.pdf"})
    ]
    unique = deduplication(docs, k=5)
    assert len(unique) == 2
    assert unique[0].page_content == "Text A"
    assert unique[1].page_content == "Text B"

def test_build_context():
    docs = [
        (Document(page_content="Text A", metadata={}), 0.9),
        (Document(page_content="Text B", metadata={}), 0.8)
    ]
    context = build_context(docs, token_limit=1000)
    assert "Text A" in context
    assert "Text B" in context
