"""
WebSocket Chat Consumer - Gerçek Zamanlı Streaming.

Bu consumer adapter'lardan gelen her token'ı anında client'a gönderir.
Python 3.11+ native asyncio.timeout kullanır.

Özellikler:
- ANLIK streaming (buffering yok)
- Otomatik adapter seçimi (Gemini, HuggingFace, Ollama)
- AI Agents desteği (tool calling, reasoning)
- RAG desteği (doküman arama)
- Rate limiting
- Keepalive ping/pong
- Graceful shutdown
- Detaylı istatistikler
- Stop komutu desteği

Protocol:
    Client -> Server:
        {"modelId": "gemini-flash", "messages": [...]}
        {"modelId": "...", "messages": [...], "useAgent": true}  - Agent modu
        {"modelId": "...", "messages": [...], "useRag": true, "ragQuery": "..."}  - RAG modu
        "__STOP__"  - Streaming'i durdur
        "__PING__"  - Manuel ping
    
    Server -> Client:
        {"type": "connected", "ts": ...}  - Bağlantı onayı
        {"delta": "token"}                - Her token anında
        {"type": "thought", "content": "..."}  - Agent düşüncesi
        {"type": "tool_call", "tool": "...", "input": {...}}  - Tool çağrısı
        {"type": "tool_result", "tool": "...", "result": "..."}  - Tool sonucu
        {"type": "rag_context", "docs": [...]}  - RAG dokümanları
        {"done": true, "stats": {...}}    - Tamamlandı
        {"error": "...", "detail": "..."}  - Hata
        {"stopped": true}                  - Durduruldu
        {"type": "ping", "ts": ...}       - Keepalive
"""

import json
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from channels.generic.websocket import AsyncWebsocketConsumer

from backend.adapters import gemini as gem, huggingface as hf, ollama as ol

# Agent ve RAG imports (lazy loading)
_agent_executor = None
_rag_pipeline = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger('websockets.consumers')
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Keepalive ping interval (saniye)
PING_INTERVAL: int = 25

# Streaming timeout (saniye) - maksimum yanıt süresi
STREAM_TIMEOUT: int = 180

# Rate limiting
RATE_LIMIT_WINDOW: int = 5      # Saniye cinsinden pencere
RATE_LIMIT_MAX: int = 10        # Pencere içinde maksimum istek

# Mesaj boyutu limitleri
MAX_MESSAGE_SIZE: int = 100000  # 100KB
MAX_MESSAGES_COUNT: int = 500   # Tek istekte maksimum mesaj sayısı


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

ADAPTERS: Dict[str, Any] = {
    'gemini': gem,
    'hf': hf,
    'ollama': ol,
}

# Model prefix -> Adapter mapping
MODEL_PREFIXES: Dict[str, str] = {
    'gemini': 'gemini',
    'hf': 'hf',
    'ollama': 'ollama',
}


def get_adapter(model_id: str):
    """
    Model ID'ye göre uygun adapter'ı seç.
    
    Args:
        model_id: Model tanımlayıcısı (örn: "gemini-flash", "hf-mistral", "ollama:qwen")
    
    Returns:
        Adapter modülü
    """
    if not model_id:
        logger.debug("No model_id provided, defaulting to Gemini")
        return gem
    
    model_lower = model_id.lower().strip()
    
    # Prefix-based seçim
    for prefix, adapter_key in MODEL_PREFIXES.items():
        if model_lower.startswith(prefix):
            adapter = ADAPTERS.get(adapter_key, gem)
            logger.debug(f"Selected adapter '{adapter_key}' for model '{model_id}'")
            return adapter
    
    # Varsayılan: Gemini
    logger.debug(f"Unknown model prefix in '{model_id}', defaulting to Gemini")
    return gem


def get_agent_executor():
    """Lazy load agent executor."""
    global _agent_executor
    if _agent_executor is None:
        try:
            from backend.agents import AgentExecutor
            _agent_executor = AgentExecutor()
            logger.info("AgentExecutor loaded successfully")
        except ImportError as e:
            logger.warning(f"AgentExecutor not available: {e}")
            _agent_executor = False  # Mark as unavailable
    return _agent_executor if _agent_executor else None


def get_rag_pipeline():
    """Lazy load RAG pipeline - singleton embedding model kullanır."""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            # Önce embedding modelini preload et (singleton)
            from rag.pipelines import get_embedding_model
            get_embedding_model()  # Singleton'u initialize et
            
            from rag.pipelines import AsyncRAGPipeline
            _rag_pipeline = AsyncRAGPipeline()
            logger.info("RAGPipeline loaded successfully")
        except ImportError as e:
            logger.warning(f"RAGPipeline not available: {e}")
            _rag_pipeline = False  # Mark as unavailable
    return _rag_pipeline if _rag_pipeline else None


# =============================================================================
# CHAT CONSUMER
# =============================================================================

class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket Chat Consumer - ANLIK STREAMING.
    
    Her client bağlantısı için ayrı bir instance oluşturulur.
    Streaming sırasında her token anında client'a gönderilir.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Bağlantı durumu
        self._connected: bool = False
        self._stop_flag: bool = False
        self._streaming: bool = False
        
        # Task referansları
        self._ping_task: Optional[asyncio.Task] = None
        self._stream_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self._request_times: List[float] = []
        
        # Client bilgileri
        self._client_id: str = ''
        self._connect_time: float = 0
        self._total_requests: int = 0
        self._total_tokens: int = 0
    
    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================
    
    async def connect(self):
        """
        WebSocket bağlantısını kabul et ve başlat.
        """
        # Client ID oluştur
        client_info = self.scope.get('client', ['unknown', 0])
        self._client_id = f"{client_info[0]}:{client_info[1]}" if len(client_info) >= 2 else str(client_info[0])
        
        # Durumu güncelle
        self._connected = True
        self._stop_flag = False
        self._connect_time = time.time()
        
        # Bağlantıyı kabul et
        await self.accept()
        
        # Keepalive task başlat
        self._ping_task = asyncio.create_task(self._keepalive_loop())
        
        logger.info(f"WebSocket connected: {self._client_id}")
        
        # Hoş geldin mesajı gönder
        await self._send_json({
            'type': 'connected',
            'ts': int(time.time() * 1000),
            'client_id': self._client_id,
        })
    
    async def disconnect(self, code: int):
        """
        WebSocket bağlantısını kapat ve temizle.
        
        Args:
            code: WebSocket kapatma kodu
        """
        self._connected = False
        self._stop_flag = True
        
        # Task'ları iptal et
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # İstatistikleri logla
        session_duration = time.time() - self._connect_time
        logger.info(
            f"WebSocket disconnected: {self._client_id}, "
            f"code={code}, duration={session_duration:.1f}s, "
            f"requests={self._total_requests}, tokens={self._total_tokens}"
        )
    
    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================
    
    async def receive(self, text_data: Optional[str] = None, bytes_data: Optional[bytes] = None):
        """
        Client'tan gelen mesajları işle.
        
        Args:
            text_data: Text formatında mesaj
            bytes_data: Binary formatında mesaj (desteklenmiyor)
        """
        if not text_data:
            return
        
        # Mesaj boyutu kontrolü
        if len(text_data) > MAX_MESSAGE_SIZE:
            await self._send_error('message_too_large', f'Maksimum mesaj boyutu: {MAX_MESSAGE_SIZE} byte')
            return
        
        text = text_data.strip()
        
        # === ÖZEL KOMUTLAR ===
        
        # Stop komutu
        if text == '__STOP__':
            await self._handle_stop()
            return
        
        # Ping komutu
        if text == '__PING__':
            await self._send_json({
                'type': 'pong',
                'ts': int(time.time() * 1000)
            })
            return
        
        # === JSON MESAJ ===
        
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            await self._send_error('invalid_json', 'Geçersiz JSON formatı')
            return
        
        # Rate limiting kontrolü
        if not self._check_rate_limit():
            await self._send_json({
                'error': 'rate_limited',
                'detail': f'Çok fazla istek. {RATE_LIMIT_WINDOW} saniye bekleyin.',
                'retry_after': RATE_LIMIT_WINDOW
            })
            return
        
        # Chat isteğini işle
        await self._handle_chat(payload)
    
    async def _handle_stop(self):
        """Stop komutunu işle."""
        logger.debug(f"Stop command received: {self._client_id}")
        
        self._stop_flag = True
        
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        
        await self._send_json({'stopped': True})
    
    async def _handle_chat(self, payload: Dict[str, Any]):
        """
        Chat isteğini işle ve streaming başlat.
        
        Args:
            payload: Chat isteği payload'ı
        """
        # Model ve mesajları al
        model_id = payload.get('modelId') or payload.get('model') or 'gemini-flash'
        messages = payload.get('messages') or []
        
        # Agent ve RAG modları
        use_agent = payload.get('useAgent', False)
        use_rag = payload.get('useRag', False)
        rag_query = payload.get('ragQuery', '')
        
        # Validasyon
        if not messages:
            await self._send_error('empty_messages', 'Mesaj listesi boş olamaz')
            return
        
        if len(messages) > MAX_MESSAGES_COUNT:
            await self._send_error('too_many_messages', f'Maksimum mesaj sayısı: {MAX_MESSAGES_COUNT}')
            return
        
        # Önceki stream'i iptal et
        if self._stream_task and not self._stream_task.done():
            logger.debug(f"Cancelling previous stream: {self._client_id}")
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        # Yeni stream başlat
        self._stop_flag = False
        self._streaming = True
        self._total_requests += 1
        
        # Hangi modu kullanacağımızı belirle
        if use_agent:
            self._stream_task = asyncio.create_task(
                self._agent_response(model_id, messages)
            )
        elif use_rag:
            self._stream_task = asyncio.create_task(
                self._rag_response(model_id, messages, rag_query)
            )
        else:
            self._stream_task = asyncio.create_task(
                self._stream_response(model_id, messages)
            )
    
    # =========================================================================
    # STREAMING - ANLIK
    # =========================================================================
    
    async def _stream_response(self, model_id: str, messages: List[Dict[str, Any]]):
        """
        Adapter'dan streaming yanıt al ve ANLIK gönder.
        
        Her token geldiğinde hemen client'a iletilir.
        Buffering YOKTUR.
        
        Args:
            model_id: Kullanılacak model ID
            messages: Mesaj listesi
        """
        adapter = get_adapter(model_id)
        start_time = time.time()
        chunk_count = 0
        total_chars = 0
        
        logger.info(f"Streaming started: client={self._client_id}, model={model_id}, messages={len(messages)}")
        
        try:
            # ✅ Python 3.11+ Native asyncio.timeout
            async with asyncio.timeout(STREAM_TIMEOUT):
                async for delta in adapter.stream(messages, model_id):
                    # Stop veya disconnect kontrolü
                    if self._stop_flag or not self._connected:
                        logger.debug(f"Streaming stopped: {self._client_id}")
                        await self._send_json({'stopped': True})
                        return
                    
                    # Delta varsa ANLIK gönder
                    if delta:
                        chunk_count += 1
                        total_chars += len(delta)
                        
                        # === ANLIK GÖNDER - BUFFERING YOK ===
                        await self._send_json(
                            {'delta': delta},
                            ensure_ascii=False
                        )
            
            # Streaming başarıyla tamamlandı
            elapsed = time.time() - start_time
            self._total_tokens += chunk_count
            
            # Tamamlandı mesajı gönder
            await self._send_json({
                'done': True,
                'stats': {
                    'chunks': chunk_count,
                    'chars': total_chars,
                    'duration_ms': int(elapsed * 1000),
                    'model': model_id,
                    'tokens_per_second': round(chunk_count / elapsed, 1) if elapsed > 0 else 0,
                }
            })
            
            logger.info(
                f"Streaming completed: client={self._client_id}, "
                f"chunks={chunk_count}, chars={total_chars}, "
                f"duration={elapsed:.2f}s"
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Streaming timeout: {self._client_id}, model={model_id}")
            await self._send_json({
                'error': 'timeout',
                'detail': f'Yanıt {STREAM_TIMEOUT} saniye içinde tamamlanamadı'
            })
            
        except asyncio.CancelledError:
            logger.debug(f"Streaming cancelled: {self._client_id}")
            # Sessizce çık, client muhtemelen stop gönderdi
            
        except Exception as e:
            logger.exception(f"Streaming error: {self._client_id}, error={e}")
            await self._send_json({
                'error': 'stream_failed',
                'detail': str(e)[:300]
            })
            
        finally:
            self._streaming = False
    
    # =========================================================================
    # KEEPALIVE
    # =========================================================================
    
    async def _keepalive_loop(self):
        """
        Periyodik ping gönder - bağlantıyı canlı tut.
        
        WebSocket bağlantıları proxy'ler tarafından kapatılabilir,
        bu ping'ler bağlantının aktif olduğunu gösterir.
        """
        try:
            while self._connected:
                await asyncio.sleep(PING_INTERVAL)
                
                if self._connected:
                    await self._send_json({
                        'type': 'ping',
                        'ts': int(time.time() * 1000)
                    })
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Keepalive error: {self._client_id}, error={e}")
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    def _check_rate_limit(self) -> bool:
        """
        Rate limiting kontrolü yap.
        
        Returns:
            True: İstek izinli
            False: Rate limit aşıldı
        """
        now = time.time()
        
        # Eski istekleri temizle
        self._request_times = [
            t for t in self._request_times 
            if now - t < RATE_LIMIT_WINDOW
        ]
        
        # Limit kontrolü
        if len(self._request_times) >= RATE_LIMIT_MAX:
            logger.warning(f"Rate limit exceeded: {self._client_id}")
            return False
        
        # Yeni isteği kaydet
        self._request_times.append(now)
        return True
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def _send_json(self, data: Dict[str, Any], ensure_ascii: bool = True):
        """
        JSON formatında mesaj gönder.
        
        Args:
            data: Gönderilecek dict
            ensure_ascii: ASCII olmayan karakterleri escape et
        """
        if not self._connected:
            return
        
        try:
            text = json.dumps(data, ensure_ascii=ensure_ascii)
            await self.send(text_data=text)
        except Exception as e:
            logger.debug(f"Send error: {self._client_id}, error={e}")
    
    async def _send_error(self, error_code: str, detail: str):
        """
        Hata mesajı gönder.
        
        Args:
            error_code: Hata kodu
            detail: Hata detayı
        """
        await self._send_json({
            'error': error_code,
            'detail': detail,
            'ts': int(time.time() * 1000)
        })
    
    # =========================================================================
    # AGENT MODE - AI ile Tool Calling
    # =========================================================================
    
    async def _agent_response(self, model_id: str, messages: List[Dict[str, Any]]):
        """
        Agent modu - AI araçları kullanarak yanıt üretir.
        
        ReAct pattern: Düşün -> Araç Kullan -> Gözlemle -> Tekrarla
        
        Args:
            model_id: Kullanılacak model ID
            messages: Mesaj listesi
        """
        agent = get_agent_executor()
        if not agent:
            await self._send_error('agent_unavailable', 'Agent sistemi yüklenemedi')
            return
        
        adapter = get_adapter(model_id)
        start_time = time.time()
        
        # Son kullanıcı mesajını al
        user_message = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        if not user_message:
            await self._send_error('no_user_message', 'Kullanıcı mesajı bulunamadı')
            return
        
        logger.info(f"Agent started: client={self._client_id}, model={model_id}")
        
        try:
            async with asyncio.timeout(STREAM_TIMEOUT):
                # Agent'ı çalıştır (streaming callback ile)
                async def on_thought(thought: str):
                    if self._connected and not self._stop_flag:
                        await self._send_json({
                            'type': 'thought',
                            'content': thought
                        })
                
                async def on_tool_call(tool_name: str, tool_input: Dict[str, Any]):
                    if self._connected and not self._stop_flag:
                        await self._send_json({
                            'type': 'tool_call',
                            'tool': tool_name,
                            'input': tool_input
                        })
                
                async def on_tool_result(tool_name: str, result: str):
                    if self._connected and not self._stop_flag:
                        await self._send_json({
                            'type': 'tool_result',
                            'tool': tool_name,
                            'result': result[:1000]  # Truncate long results
                        })
                
                # Agent'ı çalıştır
                result = await agent.run_async(
                    query=user_message,
                    model_id=model_id,
                    adapter=adapter,
                    on_thought=on_thought,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result
                )
                
                # Final yanıtı stream et
                if result and self._connected and not self._stop_flag:
                    # Yanıtı parçalara böl ve stream et
                    chunk_size = 20
                    for i in range(0, len(result), chunk_size):
                        if self._stop_flag or not self._connected:
                            break
                        chunk = result[i:i + chunk_size]
                        await self._send_json({'delta': chunk}, ensure_ascii=False)
                        await asyncio.sleep(0.01)  # Smooth streaming
                
                elapsed = time.time() - start_time
                
                await self._send_json({
                    'done': True,
                    'stats': {
                        'mode': 'agent',
                        'duration_ms': int(elapsed * 1000),
                        'model': model_id,
                    }
                })
                
                logger.info(f"Agent completed: client={self._client_id}, duration={elapsed:.2f}s")
                
        except asyncio.TimeoutError:
            await self._send_json({
                'error': 'timeout',
                'detail': f'Agent {STREAM_TIMEOUT} saniye içinde tamamlanamadı'
            })
        except asyncio.CancelledError:
            logger.debug(f"Agent cancelled: {self._client_id}")
        except Exception as e:
            logger.exception(f"Agent error: {self._client_id}, error={e}")
            await self._send_json({
                'error': 'agent_failed',
                'detail': str(e)[:300]
            })
        finally:
            self._streaming = False
    
    # =========================================================================
    # RAG MODE - Doküman Tabanlı Yanıt
    # =========================================================================
    
    def _extract_page_numbers(self, text: str) -> List[int]:
        """
        Metinden sayfa numaralarını çıkarır.
        
        Desteklenen formatlar:
        - "9. sayfayı anlat" -> [9]
        - "sayfa 12'de ne var" -> [12]
        - "5 ve 10. sayfalarda" -> [5, 10]
        - "3-7 arasındaki sayfalar" -> [3, 4, 5, 6, 7]
        - "sayfa 1,2,5" -> [1, 2, 5]
        - "9 ile 12 arası" -> [9, 10, 11, 12]
        
        Returns:
            Bulunan sayfa numaralarının listesi
        """
        import re
        
        page_numbers = set()
        text_lower = text.lower()
        
        # Pattern 1: "X. sayfa" veya "X. sayfayı" veya "X. sayfada" veya "X. sayfasında"
        pattern1 = r'(\d+)\.\s*sayfa'
        for match in re.finditer(pattern1, text_lower):
            page_numbers.add(int(match.group(1)))
        
        # Pattern 2: "sayfa X" veya "sayfa X'de" veya "sayfa X'da"
        pattern2 = r'sayfa\s*(\d+)'
        for match in re.finditer(pattern2, text_lower):
            page_numbers.add(int(match.group(1)))
        
        # Pattern 3: "X ile Y arası" veya "X-Y arası" (aralık)
        pattern3 = r'(\d+)\s*(?:ile|-)\s*(\d+)\s*(?:aras[ıi]|sayfalar[ıi]?)'
        for match in re.finditer(pattern3, text_lower):
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end <= start + 50:  # Max 50 sayfa aralık
                page_numbers.update(range(start, end + 1))
        
        # Pattern 4: "X-Y. sayfa" (aralık)
        pattern4 = r'(\d+)\s*-\s*(\d+)\.\s*sayfa'
        for match in re.finditer(pattern4, text_lower):
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end <= start + 50:
                page_numbers.update(range(start, end + 1))
        
        # Pattern 5: "X, Y ve Z. sayfalar" (virgüllü liste)
        pattern5 = r'(\d+(?:\s*,\s*\d+)+)\s*(?:ve|\.|,)?\s*(?:\d+)?\s*\.?\s*sayfa'
        for match in re.finditer(pattern5, text_lower):
            nums = re.findall(r'\d+', match.group(0))
            for n in nums:
                page_numbers.add(int(n))
        
        # Pattern 6: "X ve Y. sayfa" (iki sayfa)
        pattern6 = r'(\d+)\s+ve\s+(\d+)\s*\.?\s*sayfa'
        for match in re.finditer(pattern6, text_lower):
            page_numbers.add(int(match.group(1)))
            page_numbers.add(int(match.group(2)))
        
        # Pattern 7: "X. ve Y. sayfalar" (noktali format)
        pattern7 = r'(\d+)\.\s+ve\s+(\d+)\.\s*sayfa'
        for match in re.finditer(pattern7, text_lower):
            page_numbers.add(int(match.group(1)))
            page_numbers.add(int(match.group(2)))
        
        # Sayfa numaralarını sırala ve döndür
        return sorted(list(page_numbers))
    
    async def _rag_response(self, model_id: str, messages: List[Dict[str, Any]], query: str):
        """
        RAG modu - Dokümanlardan context alarak yanıt üretir.
        
        Sayfa numarası içeren sorgularda (örn: "9. sayfayı anlat") doğrudan
        o sayfanın içeriğini getirir. Diğer sorgularda semantik arama yapar.
        
        Args:
            model_id: Kullanılacak model ID
            messages: Mesaj listesi
            query: RAG sorgusu (boşsa son mesaj kullanılır)
        """
        rag = get_rag_pipeline()
        if not rag:
            await self._send_error('rag_unavailable', 'RAG sistemi yüklenemedi')
            return
        
        adapter = get_adapter(model_id)
        start_time = time.time()
        
        # Query'yi belirle
        if not query:
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    query = msg.get('content', '')
                    break
        
        if not query:
            await self._send_error('no_query', 'RAG sorgusu bulunamadı')
            return
        
        logger.info(f"RAG started: client={self._client_id}, model={model_id}, query_len={len(query)}")
        
        try:
            async with asyncio.timeout(STREAM_TIMEOUT):
                # Yüklü dökümanların listesini al (documents + pending)
                all_docs = rag.list_documents()
                
                # Ready ve processing dökümanları ayır
                ready_docs = [d for d in all_docs if d.get('status') == 'ready']
                pending_docs = [d for d in all_docs if d.get('status') != 'ready']
                
                doc_list_text = ""
                if all_docs:
                    doc_names = []
                    for doc in ready_docs:
                        doc_names.append(f"✅ {doc.get('file_name', 'unknown')}")
                    for doc in pending_docs:
                        doc_names.append(f"⏳ {doc.get('file_name', 'unknown')} (işleniyor...)")
                    doc_list_text = f"\n\nYüklü dökümanlar ({len(all_docs)} adet):\n" + "\n".join([f"  {name}" for name in doc_names])
                
                # ===== SAYFA BAZLI ARAMA DESTEĞİ =====
                # Sorgudan sayfa numaralarını çıkar
                requested_pages = self._extract_page_numbers(query)
                page_based_search = len(requested_pages) > 0
                
                docs = []
                search_mode = "semantic"  # veya "page-based"
                
                if page_based_search:
                    # Sayfa numarası belirtilmiş - doğrudan o sayfaları getir
                    logger.info(f"Page-based search: pages={requested_pages}")
                    search_mode = "page-based"
                    
                    page_docs = await rag.get_pages_by_number(requested_pages)
                    
                    if page_docs:
                        docs = page_docs
                        logger.info(f"Found {len(docs)} chunks for pages {requested_pages}")
                    else:
                        # Sayfa bulunamadı, semantik aramaya geri dön
                        logger.info(f"No chunks found for pages {requested_pages}, falling back to semantic search")
                        docs = await rag.search_async(query, top_k=5)
                        search_mode = "semantic-fallback"
                else:
                    # Normal semantik arama
                    docs = await rag.search_async(query, top_k=5)
                
                # Dokümanları client'a gönder
                if docs:
                    await self._send_json({
                        'type': 'rag_context',
                        'search_mode': search_mode,
                        'requested_pages': requested_pages if page_based_search else None,
                        'docs': [
                            {
                                'content': doc.get('text', '')[:500],
                                'source': doc.get('metadata', {}).get('file_name', 'unknown'),
                                'page': doc.get('metadata', {}).get('page_number', doc.get('page_number')),
                                'score': doc.get('score', 0)
                            }
                            for doc in docs
                        ]
                    })
                
                # Context'i mesajlara ekle (sayfa bilgisi ile)
                context_text = ""
                if docs:
                    context_parts = []
                    for doc in docs:
                        meta = doc.get('metadata', {})
                        file_name = meta.get('file_name', 'unknown')
                        page_num = meta.get('page_number', '')
                        total_pages = meta.get('total_pages', '')
                        page_info = f", Sayfa {page_num}/{total_pages}" if page_num else ""
                        
                        context_parts.append(
                            f"[Kaynak: {file_name}{page_info}]\n{doc.get('text', '')}"
                        )
                    context_text = "\n\n---\n\n".join(context_parts)
                
                # System mesajı varsa güncelle, yoksa ekle
                system_msg = None
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        system_msg = msg
                        break
                
                # Döküman listesini oluştur
                doc_list_items = []
                for doc in ready_docs:
                    doc_list_items.append(f"✅ {doc.get('file_name', 'unknown')} (hazır)")
                for doc in pending_docs:
                    doc_list_items.append(f"⏳ {doc.get('file_name', 'unknown')} (işleniyor...)")
                
                doc_list_str = chr(10).join(doc_list_items) if doc_list_items else "(Henüz döküman yüklenmemiş)"
                
                # Sayfa bazlı arama bilgisi
                page_search_info = ""
                if page_based_search:
                    page_search_info = f"""
🔍 SAYFA BAZLI ARAMA YAPILDI
İstenen sayfalar: {requested_pages}
Bulunan chunk sayısı: {len(docs)}
Arama modu: {search_mode}
"""
                
                rag_instruction = f"""Sen bir RAG (Retrieval Augmented Generation) asistanısın.
Kullanıcının yüklediği dökümanları kullanarak sorularını yanıtlıyorsun.

=== YÜKLÜ DÖKÜMANLAR ({len(all_docs)} adet) ===
{doc_list_str}
=== YÜKLÜ DÖKÜMANLAR SONU ===
{page_search_info}
ÖNEMLİ: "⏳ işleniyor" olan dökümanlar henüz aranabilir değil. Kullanıcı bu dökümanlar hakkında soru sorarsa, "Bu döküman henüz işleniyor, lütfen bekleyin" de.

Kullanıcı sana dökümanlar hakkında sorular sorabilir:
- "Hangi dökümanları yükledim?" → Yukarıdaki döküman listesini AYNEN göster
- "X. sayfayı anlat" → Aşağıda o sayfanın içeriği var, onu kullanarak anlat
- "Y sayfasında ne yazıyor?" → O sayfadaki içeriği bul ve yanıtla
- "Bu konuda ne biliyorsun?" → Dökümanlardan ilgili bilgileri bul

{"📖 Aşağıda, kullanıcının istediği " + (f"SAYFA {requested_pages} içeriği var:" if page_based_search else "sorusuyla ilgili bulunan döküman parçaları var:") if docs else "Bu sorguyla ilgili döküman bulunamadı (belki dökümanlar henüz işleniyor)."}

=== İLGİLİ DÖKÜMAN İÇERİĞİ ===
{context_text if context_text else "(Eşleşen içerik bulunamadı)"}
=== DÖKÜMAN İÇERİĞİ SONU ===

Kurallar:
1. Kullanıcı "hangi dökümanları yükledim" derse, YÜKLÜ DÖKÜMANLAR listesini AYNEN göster
2. Yanıtlarını SADECE yüklenen ve hazır olan (✅) dokümanlara dayandır
3. Her yanıtta kaynak döküman adını ve sayfa numarasını belirt (varsa)
4. Eğer sorulan bilgi dokümanlarda yoksa veya dökümanlar işleniyorsa, bunu açıkça söyle
5. Kullanıcı sayfa numarası sorduğunda, yukarıda o sayfanın içeriği verilmişse, onu detaylıca anlat
6. Sayfa içeriği yoksa "Bu sayfa numarasına ait içerik bulunamadı" de
"""
                
                if system_msg:
                    system_msg['content'] = system_msg['content'] + "\n\n" + rag_instruction
                else:
                    messages.insert(0, {'role': 'system', 'content': rag_instruction})
                
                # Normal streaming yap
                chunk_count = 0
                total_chars = 0
                
                async for delta in adapter.stream(messages, model_id):
                    if self._stop_flag or not self._connected:
                        await self._send_json({'stopped': True})
                        return
                    
                    if delta:
                        chunk_count += 1
                        total_chars += len(delta)
                        await self._send_json({'delta': delta}, ensure_ascii=False)
                
                elapsed = time.time() - start_time
                
                await self._send_json({
                    'done': True,
                    'stats': {
                        'mode': 'rag',
                        'search_mode': search_mode,
                        'requested_pages': requested_pages if page_based_search else None,
                        'chunks': chunk_count,
                        'chars': total_chars,
                        'docs_found': len(docs) if docs else 0,
                        'duration_ms': int(elapsed * 1000),
                        'model': model_id,
                    }
                })
                
                logger.info(
                    f"RAG completed: client={self._client_id}, "
                    f"docs={len(docs) if docs else 0}, duration={elapsed:.2f}s"
                )
                
        except asyncio.TimeoutError:
            await self._send_json({
                'error': 'timeout',
                'detail': f'RAG {STREAM_TIMEOUT} saniye içinde tamamlanamadı'
            })
        except asyncio.CancelledError:
            logger.debug(f"RAG cancelled: {self._client_id}")
        except Exception as e:
            logger.exception(f"RAG error: {self._client_id}, error={e}")
            await self._send_json({
                'error': 'rag_failed',
                'detail': str(e)[:300]
            })
        finally:
            self._streaming = False


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = ['ChatConsumer']