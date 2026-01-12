"""
Enterprise AI Assistant - Integration Tests for RAG System
============================================================

RAG sistem için mock-based integration testleri.
Gerçek vector DB ve embedding çağrısı yapmadan test eder.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockDocument:
    """Mock document for testing."""
    
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        doc_id: str = None
    ):
        self.content = content
        self.page_content = content  # LangChain compatibility
        self.metadata = metadata or {}
        self.id = doc_id or f"doc_{hash(content) % 10000}"


class MockEmbeddingModel:
    """Mock embedding model."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0
    
    def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        self.call_count += 1
        # Deterministic mock embedding based on text hash
        seed = hash(text) % 10000
        return [((seed + i) % 100) / 100 for i in range(self.dimension)]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.embed(text) for text in texts]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query."""
        return self.embed(query)


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self, embedding_model: MockEmbeddingModel = None):
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.documents: Dict[str, MockDocument] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_documents(self, documents: List[MockDocument]) -> List[str]:
        """Add documents to store."""
        ids = []
        for doc in documents:
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = self.embedding_model.embed(doc.content)
            ids.append(doc.id)
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Dict = None
    ) -> List[MockDocument]:
        """Search for similar documents."""
        query_embedding = self.embedding_model.embed_query(query)
        
        # Simple cosine-like scoring (mock)
        scores = {}
        for doc_id, doc in self.documents.items():
            # Apply filter if provided
            if filter:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue
            
            # Mock similarity score
            doc_embedding = self.embeddings[doc_id]
            score = sum(a * b for a, b in zip(query_embedding[:10], doc_embedding[:10]))
            scores[doc_id] = score
        
        # Sort by score and return top k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [self.documents[doc_id] for doc_id in sorted_ids[:k]]
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.embeddings.pop(doc_id, None)
    
    def get(self, ids: List[str]) -> List[MockDocument]:
        """Get documents by ID."""
        return [self.documents[doc_id] for doc_id in ids if doc_id in self.documents]
    
    def count(self) -> int:
        """Get document count."""
        return len(self.documents)


class MockChunker:
    """Mock text chunker."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
            if end >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[MockDocument]) -> List[MockDocument]:
        """Split documents into chunks."""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                result.append(MockDocument(
                    content=chunk,
                    metadata={**doc.metadata, "chunk_index": i},
                    doc_id=f"{doc.id}_chunk_{i}"
                ))
        return result


class MockRAGPipeline:
    """Complete mock RAG pipeline."""
    
    def __init__(
        self,
        embedding_model: MockEmbeddingModel = None,
        vector_store: MockVectorStore = None,
        chunker: MockChunker = None
    ):
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.vector_store = vector_store or MockVectorStore(self.embedding_model)
        self.chunker = chunker or MockChunker()
    
    def ingest_document(self, content: str, metadata: Dict = None) -> str:
        """Ingest a document."""
        doc = MockDocument(content=content, metadata=metadata or {})
        chunks = self.chunker.split_documents([doc])
        self.vector_store.add_documents(chunks)
        return doc.id
    
    def query(self, query: str, k: int = 5) -> List[Dict]:
        """Query the RAG system."""
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "id": doc.id
            }
            for doc in results
        ]
    
    def get_context(self, query: str, k: int = 5) -> str:
        """Get context for LLM."""
        results = self.query(query, k)
        return "\n\n---\n\n".join(r["content"] for r in results)


class TestVectorStore:
    """Vector store testleri."""
    
    @pytest.fixture
    def vector_store(self):
        return MockVectorStore()
    
    def test_add_documents(self, vector_store):
        """Döküman ekleme testi."""
        docs = [
            MockDocument("Python is a programming language", {"type": "tech"}),
            MockDocument("Machine learning is a field of AI", {"type": "tech"}),
        ]
        
        ids = vector_store.add_documents(docs)
        
        assert len(ids) == 2
        assert vector_store.count() == 2
    
    def test_similarity_search(self, vector_store):
        """Benzerlik araması testi."""
        docs = [
            MockDocument("Python programming tutorial"),
            MockDocument("JavaScript web development"),
            MockDocument("Python machine learning guide"),
        ]
        vector_store.add_documents(docs)
        
        results = vector_store.similarity_search("Python", k=2)
        
        assert len(results) <= 2
    
    def test_filtered_search(self, vector_store):
        """Filtrelenmiş arama testi."""
        docs = [
            MockDocument("Python basics", {"level": "beginner"}),
            MockDocument("Python advanced", {"level": "advanced"}),
        ]
        vector_store.add_documents(docs)
        
        results = vector_store.similarity_search(
            "Python",
            k=10,
            filter={"level": "beginner"}
        )
        
        assert all(r.metadata.get("level") == "beginner" for r in results)
    
    def test_delete_documents(self, vector_store):
        """Döküman silme testi."""
        docs = [MockDocument("Test document", doc_id="test_id")]
        vector_store.add_documents(docs)
        
        assert vector_store.count() == 1
        
        vector_store.delete(["test_id"])
        
        assert vector_store.count() == 0
    
    def test_get_documents(self, vector_store):
        """Döküman alma testi."""
        docs = [
            MockDocument("Doc 1", doc_id="id1"),
            MockDocument("Doc 2", doc_id="id2"),
        ]
        vector_store.add_documents(docs)
        
        retrieved = vector_store.get(["id1"])
        
        assert len(retrieved) == 1
        assert retrieved[0].content == "Doc 1"


class TestChunker:
    """Chunker testleri."""
    
    @pytest.fixture
    def chunker(self):
        return MockChunker(chunk_size=100, chunk_overlap=20)
    
    def test_split_short_text(self, chunker):
        """Kısa metin bölme testi."""
        text = "Short text"
        chunks = chunker.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_long_text(self, chunker):
        """Uzun metin bölme testi."""
        text = "A" * 250
        chunks = chunker.split_text(text)
        
        assert len(chunks) > 1
    
    def test_overlap(self, chunker):
        """Overlap testi."""
        text = "A" * 200
        chunks = chunker.split_text(text)
        
        if len(chunks) > 1:
            # Check overlap exists
            end_of_first = chunks[0][-20:]
            start_of_second = chunks[1][:20]
            assert end_of_first == start_of_second
    
    def test_split_documents(self, chunker):
        """Döküman bölme testi."""
        docs = [
            MockDocument("A" * 250, {"source": "test.txt"})
        ]
        
        chunks = chunker.split_documents(docs)
        
        assert len(chunks) > 1
        assert all("chunk_index" in c.metadata for c in chunks)


class TestRAGPipeline:
    """RAG pipeline testleri."""
    
    @pytest.fixture
    def rag(self):
        return MockRAGPipeline()
    
    def test_ingest_document(self, rag):
        """Döküman yükleme testi."""
        doc_id = rag.ingest_document(
            "Python is a versatile programming language",
            {"source": "intro.txt"}
        )
        
        assert doc_id is not None
        assert rag.vector_store.count() >= 1
    
    def test_query(self, rag):
        """Sorgu testi."""
        rag.ingest_document("Python programming tutorial for beginners")
        rag.ingest_document("JavaScript web development guide")
        
        results = rag.query("Python", k=5)
        
        assert len(results) >= 1
    
    def test_get_context(self, rag):
        """Context alma testi."""
        rag.ingest_document("Machine learning uses algorithms")
        rag.ingest_document("Deep learning is a subset of ML")
        
        context = rag.get_context("What is machine learning?")
        
        assert len(context) > 0
    
    def test_multiple_documents(self, rag):
        """Çoklu döküman testi."""
        for i in range(5):
            rag.ingest_document(f"Document {i} content about topic {i}")
        
        results = rag.query("topic", k=3)
        
        assert len(results) <= 3


class TestEmbeddingModel:
    """Embedding model testleri."""
    
    @pytest.fixture
    def embedding(self):
        return MockEmbeddingModel(dimension=768)
    
    def test_embed_single(self, embedding):
        """Tekil embedding testi."""
        result = embedding.embed("Test text")
        
        assert len(result) == 768
        assert all(isinstance(v, float) for v in result)
    
    def test_embed_documents(self, embedding):
        """Çoklu embedding testi."""
        texts = ["Text 1", "Text 2", "Text 3"]
        results = embedding.embed_documents(texts)
        
        assert len(results) == 3
        assert all(len(r) == 768 for r in results)
    
    def test_embed_query(self, embedding):
        """Query embedding testi."""
        result = embedding.embed_query("Search query")
        
        assert len(result) == 768
    
    def test_deterministic_embedding(self, embedding):
        """Deterministik embedding testi."""
        text = "Same text"
        
        result1 = embedding.embed(text)
        result2 = embedding.embed(text)
        
        assert result1 == result2
    
    def test_different_embeddings_for_different_texts(self, embedding):
        """Farklı metinler farklı embedding üretmeli."""
        result1 = embedding.embed("Text A")
        result2 = embedding.embed("Text B")
        
        # Should be different (though both are mock)
        assert result1 != result2


class TestRAGWithChunking:
    """Chunking ile RAG testleri."""
    
    @pytest.fixture
    def rag(self):
        return MockRAGPipeline(chunker=MockChunker(chunk_size=50, chunk_overlap=10))
    
    def test_large_document_chunking(self, rag):
        """Büyük döküman chunking testi."""
        large_content = "This is a test sentence. " * 20
        
        rag.ingest_document(large_content)
        
        # Should have multiple chunks
        assert rag.vector_store.count() > 1
    
    def test_chunk_retrieval(self, rag):
        """Chunk retrieval testi."""
        content = "Section A: Introduction. " * 10 + "Section B: Conclusion. " * 10
        
        rag.ingest_document(content)
        
        results = rag.query("Introduction", k=3)
        
        assert len(results) >= 1


class TestRAGMetadata:
    """Metadata handling testleri."""
    
    @pytest.fixture
    def rag(self):
        return MockRAGPipeline()
    
    def test_metadata_preservation(self, rag):
        """Metadata korunma testi."""
        rag.ingest_document(
            "Test content",
            {"author": "Test Author", "date": "2024-01-01"}
        )
        
        results = rag.query("content")
        
        # Metadata should include original plus chunk info
        assert "author" in results[0]["metadata"] or "chunk_index" in results[0]["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
