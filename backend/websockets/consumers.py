"""
WebSocket Chat Consumer - Gerçek Zamanlı Streaming.

Bu consumer adapter'lardan gelen her token'ı anında client'a gönderir.
"""

import json
import asyncio
import time
import logging
from typing import Optional, Dict, Any
from channels.generic.websocket import AsyncWebsocketConsumer

from backend.adapters import gemini as gem, huggingface as hf, ollama as ol

# =============================================================================
# CONFIGURATION
# =============================================================================

PING_INTERVAL = 25  # saniye
STREAM_TIMEOUT = 180  # saniye
RATE_LIMIT_WINDOW = 5  # saniye
RATE_LIMIT_MAX = 10  # maksimum istek

logger = logging.getLogger('websockets.consumers')

# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

ADAPTERS = {'gemini': gem, 'hf': hf, 'ollama': ol}

def get_adapter(model_id: str):
    """Model ID'ye göre adapter seç."""
    if not model_id:
        return gem
    mid = model_id.lower()
    if mid.startswith('gemini'):
        return gem
    elif mid.startswith('hf'):
        return hf
    elif mid.startswith('ollama'):
        return ol
    return gem


# =============================================================================
# CHAT CONSUMER
# =============================================================================

class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket Chat Consumer - ANLIK STREAMING.
    
    Client -> Server:
        {"modelId": "gemini-flash", "messages": [...]}
        "__STOP__"
    
    Server -> Client:
        {"delta": "token"}  - Her token anında
        {"done": true, "stats": {...}}
        {"error": "..."}
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_flag = False
        self._connected = False
        self._streaming = False
        self._ping_task: Optional[asyncio.Task] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._request_times = []
        self._client_id = ''
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def connect(self):
        """Bağlantı kabul."""
        self._client_id = str(self.scope.get('client', ['unknown'])[0])
        self._connected = True
        self._stop_flag = False
        
        await self.accept()
        
        # Ping task başlat
        self._ping_task = asyncio.create_task(self._keepalive_loop())
        
        logger.info(f"WS connected: {self._client_id}")
        
        await self.send(text_data=json.dumps({
            'type': 'connected',
            'ts': int(time.time() * 1000)
        }))
    
    async def disconnect(self, code):
        """Bağlantı kapat."""
        self._connected = False
        self._stop_flag = True
        
        # Task'ları temizle
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        
        logger.info(f"WS disconnected: {self._client_id}, code={code}")
    
    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================
    
    async def receive(self, text_data=None, bytes_data=None):
        """Mesaj al ve işle."""
        if not text_data:
            return
        
        text = text_data.strip()
        
        # Stop komutu
        if text == '__STOP__':
            self._stop_flag = True
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
            await self.send(text_data=json.dumps({'stopped': True}))
            return
        
        # Ping-pong
        if text == '__PING__':
            await self.send(text_data=json.dumps({'type': 'pong', 'ts': int(time.time() * 1000)}))
            return
        
        # JSON parse
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({'error': 'invalid_json'}))
            return
        
        # Rate limit
        if not self._check_rate_limit():
            await self.send(text_data=json.dumps({
                'error': 'rate_limited',
                'retry_after': RATE_LIMIT_WINDOW
            }))
            return
        
        # Chat isteği
        await self._handle_chat(payload)
    
    async def _handle_chat(self, payload: Dict[str, Any]):
        """Chat isteğini işle."""
        model_id = payload.get('modelId') or payload.get('model') or 'gemini-flash'
        messages = payload.get('messages') or []
        
        if not messages:
            await self.send(text_data=json.dumps({'error': 'empty_messages'}))
            return
        
        # Önceki stream'i iptal et
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        # Yeni stream başlat
        self._stop_flag = False
        self._streaming = True
        self._stream_task = asyncio.create_task(
            self._stream_response(model_id, messages)
        )
    
    # =========================================================================
    # STREAMING - ANLIK
    # =========================================================================
    
    async def _stream_response(self, model_id: str, messages: list):
        """
        Adapter'dan streaming yanıt al ve ANLIK gönder.
        
        Her token geldiğinde hemen client'a iletilir.
        """
        adapter = get_adapter(model_id)
        start_time = time.time()
        chunk_count = 0
        total_chars = 0
        
        logger.info(f"Streaming başladı: model={model_id}")
        
        try:
            # ✅ Python 3.10 uyumlu timeout - asyncio.wait_for kullan
            async def stream_with_timeout():
                nonlocal chunk_count, total_chars
                async for delta in adapter.stream(messages, model_id):
                    # Stop kontrolü
                    if self._stop_flag or not self._connected:
                        await self.send(text_data=json.dumps({'stopped': True}))
                        return False
                    
                    # Delta varsa ANLIK gönder
                    if delta:
                        chunk_count += 1
                        total_chars += len(delta)
                        
                        # === ANLIK GÖNDER ===
                        await self.send(text_data=json.dumps(
                            {'delta': delta},
                            ensure_ascii=False
                        ))
                return True
            
            # Timeout ile çalıştır
            completed = await asyncio.wait_for(
                stream_with_timeout(),
                timeout=STREAM_TIMEOUT
            )
            
            if not completed:
                return
            
            # Tamamlandı
            elapsed = time.time() - start_time
            await self.send(text_data=json.dumps({
                'done': True,
                'stats': {
                    'chunks': chunk_count,
                    'chars': total_chars,
                    'duration_ms': int(elapsed * 1000),
                    'model': model_id,
                }
            }))
            
            logger.info(f"Streaming tamamlandı: {chunk_count} chunks, {total_chars} chars, {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            await self.send(text_data=json.dumps({
                'error': 'timeout',
                'detail': f'{STREAM_TIMEOUT}s zaman aşımı'
            }))
            
        except asyncio.CancelledError:
            logger.debug("Streaming iptal edildi")
            
        except Exception as e:
            logger.exception(f"Streaming hatası: {e}")
            await self.send(text_data=json.dumps({
                'error': 'stream_failed',
                'detail': str(e)[:300]
            }))
            
        finally:
            self._streaming = False
    
    # =========================================================================
    # KEEPALIVE
    # =========================================================================
    
    async def _keepalive_loop(self):
        """Ping gönder."""
        try:
            while self._connected:
                await asyncio.sleep(PING_INTERVAL)
                if self._connected:
                    await self.send(text_data=json.dumps({
                        'type': 'ping',
                        'ts': int(time.time() * 1000)
                    }))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    def _check_rate_limit(self) -> bool:
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < RATE_LIMIT_WINDOW]
        if len(self._request_times) >= RATE_LIMIT_MAX:
            return False
        self._request_times.append(now)
        return True