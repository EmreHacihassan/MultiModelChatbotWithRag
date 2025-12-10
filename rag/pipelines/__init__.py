"""
RAG (Retrieval Augmented Generation) Pipeline
==============================================
Bu modül, chatbot'un kendi dökümanlarını anlayıp yanıt vermesini sağlar.

Özellikler:
- PDF, DOCX, TXT dosya desteği
- ChromaDB vector database
- Semantic search
- Chunk-based retrieval
- Multi-language embedding support

Kullanım:
    from rag.pipelines import RAGPipeline
    
    rag = RAGPipeline()
    rag.add_document("document.pdf")
    results = rag.search("sorum nedir?")
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import asyncio

logger = logging.getLogger('rag.pipelines')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """RAG Pipeline konfigürasyonu."""
    # Dizinler
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"
    INDEX_DIR: Path = BASE_DIR / "index"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # ChromaDB
    COLLECTION_NAME: str = "mychatbot_docs"
    PERSIST_DIRECTORY: str = str(BASE_DIR / "chroma_db")
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Search
    TOP_K_RESULTS: int = 5
    MIN_RELEVANCE_SCORE: float = 0.3


CONFIG = RAGConfig()

# Dizinleri oluştur
for dir_path in [CONFIG.EMBEDDINGS_DIR, CONFIG.INDEX_DIR, CONFIG.UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================

class DocumentProcessor:
    """Döküman işleme sınıfı."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.json', '.csv'}
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Dosya hash'i oluştur (duplicate kontrolü için)."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """PDF'den metin çıkar."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            logger.warning("pypdf yüklü değil. pip install pypdf")
            return ""
        except Exception as e:
            logger.error(f"PDF okuma hatası: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """DOCX'ten metin çıkar."""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text.strip()
        except ImportError:
            logger.warning("python-docx yüklü değil. pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"DOCX okuma hatası: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """TXT/MD'den metin çıkar."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"TXT okuma hatası: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Dosya türüne göre metin çıkar."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif ext in {'.txt', '.md'}:
            return cls.extract_text_from_txt(file_path)
        elif ext == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
            except:
                return ""
        else:
            logger.warning(f"Desteklenmeyen dosya türü: {ext}")
            return ""


# =============================================================================
# TEXT CHUNKER
# =============================================================================

class TextChunker:
    """Metni parçalara ayırır."""
    
    def __init__(self, chunk_size: int = CONFIG.CHUNK_SIZE, 
                 chunk_overlap: int = CONFIG.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Metni overlapping chunk'lara böl."""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Cümle sonunda kesmeye çalış
            if end < len(text):
                # Son nokta, soru işareti veya ünlem işaretini bul
                for punct in ['. ', '? ', '! ', '\n\n', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    'id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'metadata': metadata or {}
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            start = end - self.chunk_overlap
            if start < 0:
                start = end
        
        return chunks


# =============================================================================
# EMBEDDING MANAGER
# =============================================================================

class EmbeddingManager:
    """Embedding yönetimi."""
    
    def __init__(self, model_name: str = CONFIG.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy loading for embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model yüklendi: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers yüklü değil. pip install sentence-transformers")
                raise
        return self._model
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Metin listesi için embedding oluştur."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Tek sorgu için embedding oluştur."""
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()


# =============================================================================
# VECTOR STORE (ChromaDB)
# =============================================================================

class VectorStore:
    """ChromaDB vector database wrapper."""
    
    def __init__(self):
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy loading for ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=CONFIG.PERSIST_DIRECTORY,
                    anonymized_telemetry=False
                ))
                logger.info("ChromaDB client oluşturuldu")
            except ImportError:
                logger.error("chromadb yüklü değil. pip install chromadb")
                raise
            except Exception as e:
                # Fallback: in-memory client
                import chromadb
                self._client = chromadb.Client()
                logger.warning(f"ChromaDB persist hatası, in-memory kullanılıyor: {e}")
        return self._client
    
    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=CONFIG.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """Dökümanları vector store'a ekle."""
        if not chunks or not embeddings:
            return 0
        
        ids = [f"doc_{chunk['metadata'].get('file_hash', 'unknown')}_{chunk['id']}" 
               for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search(self, query_embedding: List[float], top_k: int = CONFIG.TOP_K_RESULTS) -> List[Dict]:
        """Semantic search yap."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Cosine distance -> similarity score (1 - distance)
            score = 1 - distance
            
            if score >= CONFIG.MIN_RELEVANCE_SCORE:
                search_results.append({
                    'rank': i + 1,
                    'text': doc,
                    'metadata': metadata,
                    'score': round(score, 4)
                })
        
        return search_results
    
    def delete_by_file(self, file_hash: str):
        """Dosyaya ait tüm dökümanları sil."""
        # ChromaDB where filter
        self.collection.delete(
            where={"file_hash": file_hash}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Collection istatistikleri."""
        return {
            'total_documents': self.collection.count(),
            'collection_name': CONFIG.COLLECTION_NAME
        }


# =============================================================================
# RAG PIPELINE (Main Class)
# =============================================================================

class RAGPipeline:
    """
    Ana RAG Pipeline sınıfı.
    
    Kullanım:
        rag = RAGPipeline()
        rag.add_document("/path/to/document.pdf")
        results = rag.search("Sorum nedir?")
        context = rag.get_context_for_query("Sorum nedir?")
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.chunker = TextChunker()
        self.embedding_manager = None  # Lazy init
        self.vector_store = None  # Lazy init
        self._initialized = False
        
        # Index metadata
        self.index_file = CONFIG.INDEX_DIR / "document_index.json"
        self.document_index = self._load_index()
    
    def _ensure_initialized(self):
        """Lazy initialization."""
        if not self._initialized:
            try:
                self.embedding_manager = EmbeddingManager()
                self.vector_store = VectorStore()
                self._initialized = True
            except Exception as e:
                logger.error(f"RAG initialization hatası: {e}")
                raise
    
    def _load_index(self) -> Dict[str, Any]:
        """Döküman index'ini yükle."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'documents': {}, 'last_updated': None}
    
    def _save_index(self):
        """Döküman index'ini kaydet."""
        self.document_index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, ensure_ascii=False, indent=2)
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Döküman ekle.
        
        Args:
            file_path: Dosya yolu
            metadata: Ek metadata (opsiyonel)
        
        Returns:
            İşlem sonucu
        """
        self._ensure_initialized()
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'success': False, 'error': 'Dosya bulunamadı'}
        
        if file_path.suffix.lower() not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            return {'success': False, 'error': f'Desteklenmeyen dosya türü: {file_path.suffix}'}
        
        # Duplicate kontrolü
        file_hash = self.document_processor.get_file_hash(str(file_path))
        
        if file_hash in self.document_index['documents']:
            return {'success': False, 'error': 'Bu döküman zaten eklenmiş'}
        
        # Metin çıkar
        text = self.document_processor.extract_text(str(file_path))
        
        if not text:
            return {'success': False, 'error': 'Dökümandan metin çıkarılamadı'}
        
        # Chunk'lara böl
        doc_metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'file_type': file_path.suffix.lower(),
            'added_at': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        chunks = self.chunker.chunk_text(text, doc_metadata)
        
        if not chunks:
            return {'success': False, 'error': 'Chunk oluşturulamadı'}
        
        # Embedding oluştur
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.embedding_manager.embed_texts(chunk_texts)
        
        # Vector store'a ekle
        added_count = self.vector_store.add_documents(chunks, embeddings)
        
        # Index'e kaydet
        self.document_index['documents'][file_hash] = {
            'file_name': file_path.name,
            'chunk_count': added_count,
            'char_count': len(text),
            'added_at': doc_metadata['added_at']
        }
        self._save_index()
        
        logger.info(f"Döküman eklendi: {file_path.name} ({added_count} chunk)")
        
        return {
            'success': True,
            'file_name': file_path.name,
            'chunks_added': added_count,
            'total_characters': len(text)
        }
    
    def add_text(self, text: str, source_name: str = "manual_input", 
                 metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Direkt metin ekle (dosya olmadan)."""
        self._ensure_initialized()
        
        if not text or len(text.strip()) < 10:
            return {'success': False, 'error': 'Metin çok kısa'}
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        doc_metadata = {
            'source': source_name,
            'file_hash': text_hash,
            'file_type': 'text',
            'added_at': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        chunks = self.chunker.chunk_text(text, doc_metadata)
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.embedding_manager.embed_texts(chunk_texts)
        
        added_count = self.vector_store.add_documents(chunks, embeddings)
        
        return {
            'success': True,
            'source': source_name,
            'chunks_added': added_count
        }
    
    def search(self, query: str, top_k: int = CONFIG.TOP_K_RESULTS) -> List[Dict]:
        """
        Semantic search yap.
        
        Args:
            query: Arama sorgusu
            top_k: Döndürülecek sonuç sayısı
        
        Returns:
            Sonuç listesi
        """
        self._ensure_initialized()
        
        if not query or len(query.strip()) < 2:
            return []
        
        query_embedding = self.embedding_manager.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> str:
        """
        Sorgu için context string oluştur (LLM'e verilecek).
        
        Args:
            query: Kullanıcı sorusu
            max_context_length: Maksimum context uzunluğu
        
        Returns:
            Formatlanmış context string
        """
        results = self.search(query)
        
        if not results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for result in results:
            text = result['text']
            source = result['metadata'].get('file_name', 'Bilinmeyen kaynak')
            score = result['score']
            
            part = f"[Kaynak: {source} (Relevance: {score:.2f})]\n{text}\n"
            
            if total_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            total_length += len(part)
        
        return "\n---\n".join(context_parts)
    
    def delete_document(self, file_hash: str) -> bool:
        """Dökümanı sil."""
        self._ensure_initialized()
        
        if file_hash not in self.document_index['documents']:
            return False
        
        self.vector_store.delete_by_file(file_hash)
        del self.document_index['documents'][file_hash]
        self._save_index()
        
        return True
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Eklenen dökümanları listele."""
        return [
            {'hash': h, **info}
            for h, info in self.document_index['documents'].items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Pipeline istatistikleri."""
        try:
            self._ensure_initialized()
            vs_stats = self.vector_store.get_stats()
        except:
            vs_stats = {'total_documents': 0}
        
        return {
            'indexed_files': len(self.document_index['documents']),
            'total_chunks': vs_stats.get('total_documents', 0),
            'last_updated': self.document_index.get('last_updated'),
            'embedding_model': CONFIG.EMBEDDING_MODEL
        }


# =============================================================================
# ASYNC WRAPPER (WebSocket/API için)
# =============================================================================

class AsyncRAGPipeline:
    """RAGPipeline'ın async wrapper'ı."""
    
    def __init__(self):
        self._pipeline = None
    
    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = RAGPipeline()
        return self._pipeline
    
    async def add_document(self, file_path: str, metadata: Dict = None) -> Dict:
        """Async döküman ekleme."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pipeline.add_document, file_path, metadata
        )
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Async arama."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pipeline.search, query, top_k
        )
    
    async def get_context(self, query: str) -> str:
        """Async context alma."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pipeline.get_context_for_query, query
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_rag_instance: Optional[RAGPipeline] = None
_async_rag_instance: Optional[AsyncRAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Singleton RAG pipeline instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


def get_async_rag_pipeline() -> AsyncRAGPipeline:
    """Singleton async RAG pipeline instance."""
    global _async_rag_instance
    if _async_rag_instance is None:
        _async_rag_instance = AsyncRAGPipeline()
    return _async_rag_instance


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RAGPipeline',
    'AsyncRAGPipeline',
    'RAGConfig',
    'DocumentProcessor',
    'TextChunker',
    'EmbeddingManager',
    'VectorStore',
    'get_rag_pipeline',
    'get_async_rag_pipeline',
    'CONFIG',
]
