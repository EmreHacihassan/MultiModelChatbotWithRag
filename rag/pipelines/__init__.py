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
    
    # Embedding - Hafif ve hızlı model
    # all-MiniLM-L6-v2: 80MB, iyi kalite
    # paraphrase-MiniLM-L3-v2: 50MB, hızlı
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    
    # Search
    TOP_K_RESULTS: int = 5
    MIN_RELEVANCE_SCORE: float = 0.1  # Düşük tutarak daha fazla sonuç al


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
        """PDF'den metin çıkar (sayfa bilgisi ile)."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_num = i + 1
                page_text = page.extract_text() or ""
                # Her sayfanın başına ve sonuna işaretleyici ekle
                text += f"\n\n[SAYFA {page_num}/{total_pages} BAŞLANGIÇ]\n"
                text += page_text.strip()
                text += f"\n[SAYFA {page_num}/{total_pages} BİTİŞ]\n"
            return text.strip()
        except ImportError:
            logger.warning("pypdf yüklü değil. pip install pypdf")
            return ""
        except Exception as e:
            logger.error(f"PDF okuma hatası: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_pdf_by_page(file_path: str) -> List[Dict[str, Any]]:
        """PDF'den sayfa sayfa metin çıkar."""
        pages = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_num = i + 1
                page_text = page.extract_text() or ""
                pages.append({
                    'page_number': page_num,
                    'total_pages': total_pages,
                    'text': page_text.strip(),
                    'char_count': len(page_text)
                })
        except Exception as e:
            logger.error(f"PDF sayfa okuma hatası: {e}")
        return pages
    
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
    """Metni parçalara ayırır (sayfa bilgisi koruyarak)."""
    
    def __init__(self, chunk_size: int = CONFIG.CHUNK_SIZE, 
                 chunk_overlap: int = CONFIG.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _extract_page_info(self, text: str, position: int) -> Dict[str, Any]:
        """Verilen pozisyondaki sayfa bilgisini çıkar."""
        import re
        # [SAYFA X/Y BAŞLANGIÇ] pattern'ini bul
        page_pattern = r'\[SAYFA (\d+)/(\d+) BAŞLANGIÇ\]'
        
        # Position'dan önce en yakın sayfa işaretini bul
        text_before = text[:position]
        matches = list(re.finditer(page_pattern, text_before))
        
        if matches:
            last_match = matches[-1]
            page_num = int(last_match.group(1))
            total_pages = int(last_match.group(2))
            return {'page_number': page_num, 'total_pages': total_pages}
        
        return {'page_number': 1, 'total_pages': 1}
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Metni overlapping chunk'lara böl (sayfa bilgisi ile)."""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        text_len = len(text)
        max_iterations = (text_len // (self.chunk_size - self.chunk_overlap)) + 100  # Güvenlik sınırı
        iterations = 0
        
        while start < text_len and iterations < max_iterations:
            iterations += 1
            end = min(start + self.chunk_size, text_len)
            
            # Cümle sonunda kesmeye çalış (sadece text'in ortasındaysa)
            if end < text_len:
                best_end = end
                for punct in ['. ', '? ', '! ', '\n\n', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:  # start'tan büyük olmalı
                        best_end = last_punct + len(punct)
                        break
                end = best_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Sayfa bilgisini çıkar
                page_info = self._extract_page_info(text, start)
                
                # Metadata'ya sayfa bilgisini ekle
                chunk_metadata = {**(metadata or {}), **page_info}
                
                chunk_data = {
                    'id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'metadata': chunk_metadata
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Sonraki başlangıç noktası - overlap ile geri git ama asla geri gitme
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end  # İlerleme garantisi
            start = next_start
        
        if iterations >= max_iterations:
            logger.warning(f"Chunk iteration limit reached! text_len={text_len}, chunks={len(chunks)}")
        
        return chunks


# =============================================================================
# EMBEDDING MANAGER (Singleton)
# =============================================================================

# Global singleton instances
_embedding_model_instance = None
_chroma_client_instance = None

def get_embedding_model():
    """Global singleton embedding model."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Embedding model yükleniyor: {CONFIG.EMBEDDING_MODEL}")
        _embedding_model_instance = SentenceTransformer(CONFIG.EMBEDDING_MODEL)
        logger.info("Embedding model hazır!")
    return _embedding_model_instance

def get_chroma_client():
    """Global singleton ChromaDB client."""
    global _chroma_client_instance
    if _chroma_client_instance is None:
        import chromadb
        from chromadb.config import Settings
        # Telemetry'yi kapat ve client oluştur
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        _chroma_client_instance = chromadb.PersistentClient(
            path=CONFIG.PERSIST_DIRECTORY,
            settings=settings
        )
        logger.info(f"ChromaDB client oluşturuldu: {CONFIG.PERSIST_DIRECTORY}")
    return _chroma_client_instance

class EmbeddingManager:
    """Embedding yönetimi - Singleton model kullanır."""
    
    def __init__(self, model_name: str = CONFIG.EMBEDDING_MODEL):
        self.model_name = model_name
    
    @property
    def model(self):
        """Global singleton model döndür."""
        return get_embedding_model()
    
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
    """ChromaDB vector database wrapper - Singleton client kullanır."""
    
    def __init__(self):
        self._collection = None
    
    @property
    def client(self):
        """Global singleton client döndür."""
        return get_chroma_client()
    
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
        if not results['documents'] or not results['documents'][0]:
            logger.debug("Arama sonucu boş")
            return search_results
            
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # ChromaDB cosine space: distance range [0, 2]
            # 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            score = max(0, 1 - (distance / 2))
            
            logger.debug(f"Sonuç {i+1}: distance={distance:.4f}, score={score:.4f}")
            
            # Minimum skoru geçenleri ekle
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
        """Döküman index'ini yükle (UTF-8 BOM destekli)."""
        if self.index_file.exists():
            try:
                # UTF-8 BOM'lu dosyaları da oku (utf-8-sig)
                with open(self.index_file, 'r', encoding='utf-8-sig') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Index dosyası bozuk, sıfırlanıyor: {e}")
                return {'documents': {}, 'pending': {}, 'last_updated': None}
        return {'documents': {}, 'pending': {}, 'last_updated': None}
    
    def _save_index(self):
        """Döküman index'ini kaydet (UTF-8, BOM'suz)."""
        self.document_index['last_updated'] = datetime.now().isoformat()
        # BOM olmadan kaydet
        with open(self.index_file, 'w', encoding='utf-8', newline='') as f:
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
        import time
        start_time = time.time()
        
        logger.info(f"[RAG] Döküman ekleme başlıyor: {file_path}")
        
        self._ensure_initialized()
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"[RAG] Dosya bulunamadı: {file_path}")
            return {'success': False, 'error': 'Dosya bulunamadı'}
        
        if file_path.suffix.lower() not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            logger.error(f"[RAG] Desteklenmeyen dosya türü: {file_path.suffix}")
            return {'success': False, 'error': f'Desteklenmeyen dosya türü: {file_path.suffix}'}
        
        # Duplicate kontrolü
        logger.info(f"[RAG] Hash hesaplanıyor...")
        t1 = time.time()
        file_hash = self.document_processor.get_file_hash(str(file_path))
        logger.info(f"[RAG] Hash: {file_hash[:8]}... ({time.time()-t1:.2f}s)")
        
        if file_hash in self.document_index['documents']:
            logger.warning(f"[RAG] Döküman zaten mevcut: {file_path.name}")
            return {'success': False, 'error': 'Bu döküman zaten eklenmiş'}
        
        # Metin çıkar
        logger.info(f"[RAG] Metin çıkarılıyor...")
        t1 = time.time()
        text = self.document_processor.extract_text(str(file_path))
        logger.info(f"[RAG] Metin çıkarıldı: {len(text) if text else 0} karakter ({time.time()-t1:.2f}s)")
        
        if not text:
            logger.error(f"[RAG] Metin çıkarılamadı!")
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
        
        logger.info(f"[RAG] Chunk'lara bölünüyor...")
        t1 = time.time()
        chunks = self.chunker.chunk_text(text, doc_metadata)
        logger.info(f"[RAG] {len(chunks) if chunks else 0} chunk oluşturuldu ({time.time()-t1:.2f}s)")
        
        if not chunks:
            logger.error(f"[RAG] Chunk oluşturulamadı!")
            return {'success': False, 'error': 'Chunk oluşturulamadı'}
        
        # Embedding oluştur
        logger.info(f"[RAG] Embedding oluşturuluyor ({len(chunks)} chunk)...")
        t1 = time.time()
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.embedding_manager.embed_texts(chunk_texts)
        logger.info(f"[RAG] Embedding tamamlandı ({time.time()-t1:.2f}s)")
        
        # Vector store'a ekle
        logger.info(f"[RAG] ChromaDB'ye kaydediliyor...")
        t1 = time.time()
        added_count = self.vector_store.add_documents(chunks, embeddings)
        logger.info(f"[RAG] ChromaDB'ye kaydedildi: {added_count} chunk ({time.time()-t1:.2f}s)")
        
        # Index'e kaydet
        self.document_index['documents'][file_hash] = {
            'file_name': file_path.name,
            'chunk_count': added_count,
            'char_count': len(text),
            'added_at': doc_metadata['added_at']
        }
        
        # Pending'den kaldır (varsa)
        if 'pending' in self.document_index and file_hash in self.document_index['pending']:
            del self.document_index['pending'][file_hash]
        
        self._save_index()
        
        total_time = time.time() - start_time
        logger.info(f"[RAG] ✅ Döküman eklendi: {file_path.name} ({added_count} chunk, toplam {total_time:.2f}s)")
        
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
        """Eklenen dökümanları listele (pending dahil)."""
        result = []
        
        # Tamamlanmış dökümanlar
        for h, info in self.document_index.get('documents', {}).items():
            result.append({
                'hash': h,
                'file_name': info.get('file_name', 'unknown'),
                'chunk_count': info.get('chunk_count', 0),
                'added_at': info.get('added_at'),
                'status': 'ready',
                **info
            })
        
        # Pending dökümanlar (henüz işlenmemiş)
        for h, info in self.document_index.get('pending', {}).items():
            if h not in self.document_index.get('documents', {}):
                result.append({
                    'hash': h,
                    'file_name': info.get('file_name', 'unknown'),
                    'chunk_count': 0,
                    'added_at': info.get('added_at'),
                    'status': info.get('status', 'processing'),
                })
        
        return result
    
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
    
    def clear(self) -> bool:
        """Tüm dökümanları ve vektörleri temizle."""
        try:
            self._ensure_initialized()
            # Vector store'u temizle - collection'ı sil ve yeniden oluştur
            if self.vector_store:
                try:
                    client = get_chroma_client()
                    # Collection'ı tamamen sil
                    try:
                        client.delete_collection(CONFIG.COLLECTION_NAME)
                        logger.info(f"Collection '{CONFIG.COLLECTION_NAME}' silindi")
                    except Exception:
                        pass
                    # Yeni collection oluştur
                    self.vector_store._collection = None
                    _ = self.vector_store.collection  # Yeniden oluştur
                except Exception as e:
                    logger.warning(f"Collection temizleme hatası: {e}")
            # Index'i temizle
            self.document_index = {
                'documents': {},
                'pending': {},
                'last_updated': datetime.now().isoformat(),
            }
            self._save_index()
            logger.info("RAG Pipeline temizlendi")
            return True
        except Exception as e:
            logger.error(f"Clear hatası: {e}")
            return False


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
    
    async def search_async(self, query: str, top_k: int = 5) -> List[Dict]:
        """Async arama (alias for search)."""
        return await self.search(query, top_k)
    
    async def get_context(self, query: str) -> str:
        """Async context alma."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.pipeline.get_context_for_query, query
        )
    
    async def get_context_async(self, query: str) -> str:
        """Async context alma (alias)."""
        return await self.get_context(query)
    
    def list_documents(self) -> List[Dict]:
        """Dökümanları listele."""
        return self.pipeline.list_documents()
    
    def get_stats(self) -> Dict:
        """İstatistikleri al."""
        return self.pipeline.get_stats()


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


def preload_embedding_model():
    """
    Embedding modelini önceden yükle.
    Sunucu başlangıcında çağrılırsa, kullanıcı bekleme yaşamaz.
    """
    try:
        logger.info("Embedding modeli ön yükleniyor...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(CONFIG.EMBEDDING_MODEL)
        # Test embedding
        _ = model.encode(["test"], convert_to_numpy=True)
        logger.info(f"Embedding modeli hazır: {CONFIG.EMBEDDING_MODEL}")
        return True
    except Exception as e:
        logger.warning(f"Embedding modeli ön yüklenemedi: {e}")
        return False


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
    'preload_embedding_model',
    'CONFIG',
]
