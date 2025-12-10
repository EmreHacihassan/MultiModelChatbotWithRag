"""
Core Routers - REST API Endpoint'leri.

Tüm HTTP endpoint'leri bu modülde tanımlanır:
- /health/ - Sağlık kontrolü
- /models - Model listesi (dinamik)
- /sessions/* - Oturum yönetimi
- /chat - Non-streaming chat
- /chat/stream - SSE streaming chat

Aralık 2024 - Güncellenmiş versiyon.
"""

import json
import os
import time
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Generator
from functools import wraps

from django.http import JsonResponse, StreamingHttpResponse, HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.urls import path

from backend.adapters import gemini as gem, huggingface as hf, ollama as ol

# =============================================================================
# CONFIG & LOGGING
# =============================================================================

logger = logging.getLogger('core.routers')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'sessions')
os.makedirs(DATA_DIR, exist_ok=True)

SSE_RETRY_MS = 3000
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60
_rate_limit_store: Dict[str, List[float]] = {}

ADAPTERS = {'gemini': gem, 'hf': hf, 'ollama': ol}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_adapter(model_id: str) -> Tuple[str, Any]:
    """Model ID'ye göre adapter seç."""
    if not model_id:
        return ('gemini', gem)
    mid = model_id.lower().strip()
    if mid.startswith('gemini'):
        return ('gemini', gem)
    elif mid.startswith('hf'):
        return ('hf', hf)
    elif mid.startswith('ollama'):
        return ('ollama', ol)
    return ('gemini', gem)


def _session_path(sid: str) -> str:
    safe_sid = ''.join(c for c in sid if c.isalnum() or c in '-_')
    return os.path.join(DATA_DIR, f'{safe_sid}.json')


def _load_json(filepath: str, default: Any = None) -> Any:
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"JSON load error: {e}")
        return default


def _save_json(filepath: str, data: Any) -> bool:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"JSON save error: {e}")
        return False


def _get_client_ip(request: HttpRequest) -> str:
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []
    _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


def _parse_json_body(request: HttpRequest) -> Dict[str, Any]:
    try:
        if not request.body:
            return {}
        return json.loads(request.body)
    except Exception:
        return {}


def rate_limited(view_func):
    @wraps(view_func)
    def wrapper(request: HttpRequest, *args, **kwargs):
        if not _check_rate_limit(_get_client_ip(request)):
            return JsonResponse({'error': 'rate_limited', 'retry_after': RATE_LIMIT_WINDOW}, status=429)
        return view_func(request, *args, **kwargs)
    return wrapper


def _get_preview(messages: list) -> str:
    if not messages:
        return ''
    content = str(messages[-1].get('content', ''))
    return content[:100] + '...' if len(content) > 100 else content


# =============================================================================
# MODELS ROUTER - DİNAMİK MODEL LİSTESİ
# =============================================================================

def models_router() -> List[path]:
    """
    Model listesi endpoint'i.
    
    Tüm adaptörlerden (Gemini, HuggingFace, Ollama) model listesini
    dinamik olarak toplar ve döndürür.
    """
    
    def list_models(request: HttpRequest) -> JsonResponse:
        all_models = []
        
        # =====================================================================
        # GEMINI MODELLERİ
        # =====================================================================
        gemini_models = [
            {
                'id': 'gemini-3-pro',
                'name': 'Gemini 3 Pro',
                'provider': 'gemini',
                'streaming': True,
                'description': 'Google Gemini 3 Pro - En güçlü model (2025)',
                'context_window': 2097152,
                'tier': 1,
            },
            {
                'id': 'gemini-flash',
                'name': 'Gemini 2.5 Flash',
                'provider': 'gemini',
                'streaming': True,
                'description': 'Google Gemini 2.5 Flash - Hızlı ve güçlü',
                'context_window': 1048576,
                'tier': 1,
            },
            {
                'id': 'gemini-pro',
                'name': 'Gemini 1.5 Pro',
                'provider': 'gemini',
                'streaming': True,
                'description': 'Google Gemini 1.5 Pro - Detaylı yanıtlar',
                'context_window': 2097152,
                'tier': 2,
            },
        ]
        all_models.extend(gemini_models)
        
        # =====================================================================
        # HUGGINGFACE MODELLERİ (Adaptörden dinamik)
        # =====================================================================
        try:
            # HuggingFace adaptöründen model listesini al
            if hasattr(hf, 'get_available_models'):
                hf_models = hf.get_available_models()
            elif hasattr(hf, 'MODELS'):
                # MODELS dict'inden manuel oluştur
                hf_models = []
                for model_id, config in hf.MODELS.items():
                    hf_models.append({
                        'id': model_id,
                        'name': config.get('name', model_id),
                        'provider': 'hf',
                        'streaming': True,
                        'description': config.get('description', 'HuggingFace model'),
                        'tier': config.get('tier', 5),
                    })
            else:
                hf_models = []
            
            all_models.extend(hf_models)
            logger.debug(f"Loaded {len(hf_models)} HuggingFace models")
            
        except Exception as e:
            logger.error(f"HuggingFace model listesi alınamadı: {e}")
            # Fallback: HuggingFace Inference Providers (2025 - Together & Hyperbolic)
            fallback_hf = [
                # Together Provider - Tier 1
                {'id': 'hf-llama-3.2-3b', 'name': 'Llama 3.2 3B', 'provider': 'hf', 'streaming': True, 'tier': 1, 'description': 'Meta - Hızlı ve yetenekli'},
                {'id': 'hf-llama-3.1-8b', 'name': 'Llama 3.1 8B', 'provider': 'hf', 'streaming': True, 'tier': 1, 'description': 'Meta - Güçlü ve dengeli'},
                {'id': 'hf-llama-3.1-70b', 'name': 'Llama 3.1 70B', 'provider': 'hf', 'streaming': True, 'tier': 1, 'description': 'Meta - En güçlü açık kaynak'},
                {'id': 'hf-qwen-2.5-72b', 'name': 'Qwen 2.5 72B', 'provider': 'hf', 'streaming': True, 'tier': 1, 'description': 'Alibaba - En güçlü Qwen'},
                # Together Provider - Tier 2
                {'id': 'hf-qwen-2.5-7b', 'name': 'Qwen 2.5 7B', 'provider': 'hf', 'streaming': True, 'tier': 2, 'description': 'Alibaba - Çok dilli'},
                # Hyperbolic Provider
                {'id': 'hf-llama-3.2-3b-hyp', 'name': 'Llama 3.2 3B (Hyp)', 'provider': 'hf', 'streaming': True, 'tier': 2, 'description': 'Meta - Hyperbolic'},
                {'id': 'hf-qwen-2.5-72b-hyp', 'name': 'Qwen 2.5 72B (Hyp)', 'provider': 'hf', 'streaming': True, 'tier': 1, 'description': 'Alibaba - Hyperbolic'},
            ]
            all_models.extend(fallback_hf)
        
        # =====================================================================
        # OLLAMA MODELLERİ
        # =====================================================================
        ollama_models = [
            {
                'id': 'ollama:qwen2.5',
                'name': 'Qwen 2.5 (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Alibaba Qwen 2.5 - Yerel',
                'context_window': 32768,
            },
            {
                'id': 'ollama:llama3.1',
                'name': 'Llama 3.1 (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Meta Llama 3.1 - Yerel',
                'context_window': 8192,
            },
            {
                'id': 'ollama:mistral',
                'name': 'Mistral (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Mistral AI - Yerel',
                'context_window': 8192,
            },
            {
                'id': 'ollama:phi3',
                'name': 'Phi-3 (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Microsoft Phi-3 - Yerel',
                'context_window': 4096,
            },
            {
                'id': 'ollama:gemma2',
                'name': 'Gemma 2 (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Google Gemma 2 - Yerel',
                'context_window': 8192,
            },
            {
                'id': 'ollama:codellama',
                'name': 'CodeLlama (Ollama)',
                'provider': 'ollama',
                'streaming': True,
                'description': 'Meta CodeLlama - Kod yazımı',
                'context_window': 16384,
            },
        ]
        all_models.extend(ollama_models)
        
        # =====================================================================
        # SONUÇ
        # =====================================================================
        logger.info(f"Toplam {len(all_models)} model döndürülüyor")
        return JsonResponse(all_models, safe=False)
    
    return [path('models', list_models)]


# =============================================================================
# SESSIONS ROUTER
# =============================================================================

def sessions_router() -> List[path]:
    
    @csrf_exempt
    @require_http_methods(['GET'])
    def list_sessions(request: HttpRequest) -> JsonResponse:
        items = []
        try:
            for fn in os.listdir(DATA_DIR):
                if not fn.endswith('.json'):
                    continue
                sid = fn[:-5]
                data = _load_json(_session_path(sid), {})
                items.append({
                    'id': sid,
                    'title': data.get('title', 'Sohbet'),
                    'updatedAt': data.get('updatedAt'),
                    'count': len(data.get('messages', [])),
                    'preview': _get_preview(data.get('messages', [])),
                })
        except Exception as e:
            logger.error(f"List sessions error: {e}")
        items.sort(key=lambda x: x.get('updatedAt') or 0, reverse=True)
        return JsonResponse(items, safe=False)
    
    @csrf_exempt
    @require_http_methods(['POST'])
    def create_session(request: HttpRequest) -> JsonResponse:
        payload = _parse_json_body(request)
        title = (payload.get('title') or 'Yeni Sohbet').strip()[:100]
        sid = f"{int(time.time() * 1000)}"
        data = {
            'id': sid, 'title': title, 'messages': [],
            'createdAt': int(time.time()), 'updatedAt': int(time.time()), 'autoTitled': False,
        }
        if _save_json(_session_path(sid), data):
            return JsonResponse({'id': sid, 'title': title})
        return JsonResponse({'error': 'save_failed'}, status=500)
    
    @require_http_methods(['GET'])
    def get_session(request: HttpRequest, sid: str) -> JsonResponse:
        data = _load_json(_session_path(sid))
        if not data:
            return JsonResponse({'error': 'not_found'}, status=404)
        return JsonResponse(data)
    
    @csrf_exempt
    @require_http_methods(['POST'])
    def rename_session(request: HttpRequest, sid: str) -> JsonResponse:
        data = _load_json(_session_path(sid))
        if not data:
            return JsonResponse({'error': 'not_found'}, status=404)
        payload = _parse_json_body(request)
        data['title'] = (payload.get('title') or 'Sohbet').strip()[:100]
        data['updatedAt'] = int(time.time())
        if _save_json(_session_path(sid), data):
            return JsonResponse({'ok': True, 'title': data['title']})
        return JsonResponse({'error': 'save_failed'}, status=500)
    
    @csrf_exempt
    @require_http_methods(['POST', 'DELETE'])
    def delete_session(request: HttpRequest, sid: str) -> JsonResponse:
        filepath = _session_path(sid)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return JsonResponse({'ok': True})
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return JsonResponse({'error': 'delete_failed'}, status=500)
    
    @csrf_exempt
    @require_http_methods(['POST'])
    def append_message(request: HttpRequest, sid: str) -> JsonResponse:
        """
        Oturuma mesaj ekle.
        """
        data = _load_json(_session_path(sid))
        if not data:
            return JsonResponse({'error': 'not_found'}, status=404)
        
        msg = _parse_json_body(request)
        if 'role' not in msg or 'content' not in msg:
            return JsonResponse({'error': 'invalid_message'}, status=400)
        
        clean_msg = {
            'role': msg['role'],
            'content': str(msg['content'])[:10000],
            'modelId': msg.get('modelId'),
            'modelName': msg.get('modelName'),
            'timestamp': msg.get('timestamp') or int(time.time() * 1000),
        }
        
        if 'messages' not in data:
            data['messages'] = []
        data['messages'].append(clean_msg)
        
        # İlk kullanıcı mesajından otomatik başlık
        if not data.get('autoTitled') and msg.get('role') == 'user':
            data['title'] = str(msg.get('content', ''))[:50] or 'Sohbet'
            data['autoTitled'] = True
        
        data['updatedAt'] = int(time.time())
        
        if _save_json(_session_path(sid), data):
            return JsonResponse({'ok': True, 'count': len(data['messages'])})
        return JsonResponse({'error': 'save_failed'}, status=500)
    
    return [
        path('sessions', list_sessions),
        path('sessions/create', create_session),
        path('sessions/<str:sid>', get_session),
        path('sessions/<str:sid>/rename', rename_session),
        path('sessions/<str:sid>/delete', delete_session),
        path('sessions/<str:sid>/append', append_message),
    ]


# =============================================================================
# CHAT ROUTER (Non-streaming)
# =============================================================================

def chat_router() -> List[path]:
    
    @csrf_exempt
    @require_http_methods(['POST'])
    @rate_limited
    def chat(request: HttpRequest) -> JsonResponse:
        payload = _parse_json_body(request)
        model_id = payload.get('modelId') or 'gemini-flash'
        messages = payload.get('messages') or []
        
        if not messages:
            return JsonResponse({'error': 'empty_messages'}, status=400)
        
        _, adapter = get_adapter(model_id)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(adapter.generate(messages, model_id))
            finally:
                loop.close()
            
            return JsonResponse({
                'text': result,
                'model': model_id,
                'timestamp': int(time.time() * 1000)
            })
        except Exception as e:
            logger.exception(f"Chat error: {e}")
            return JsonResponse({'error': 'generation_failed', 'detail': str(e)[:500]}, status=500)
    
    return [path('chat', chat)]


# =============================================================================
# SSE ROUTER - STREAMING
# =============================================================================

def sse_router() -> List[path]:
    
    @csrf_exempt
    @require_http_methods(['POST'])
    def chat_stream(request: HttpRequest) -> HttpResponse:
        client_ip = _get_client_ip(request)
        if not _check_rate_limit(client_ip):
            return JsonResponse({'error': 'rate_limited'}, status=429)
        
        payload = _parse_json_body(request)
        model_id = payload.get('modelId') or payload.get('model') or 'gemini-flash'
        messages = payload.get('messages') or []
        
        logger.info(f"SSE stream: model={model_id}, messages={len(messages)}")
        
        adapter_name, adapter = get_adapter(model_id)
        
        def generate_sse_events() -> Generator[str, None, None]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                yield f"retry: {SSE_RETRY_MS}\n\n"
                
                start_time = time.time()
                chunk_count = 0
                total_chars = 0
                
                stream_gen = adapter.stream(messages, model_id)
                
                while True:
                    try:
                        delta = loop.run_until_complete(
                            asyncio.wait_for(stream_gen.__anext__(), timeout=120)
                        )
                        
                        if delta:
                            chunk_count += 1
                            total_chars += len(delta)
                            event_data = json.dumps({'delta': delta}, ensure_ascii=False)
                            yield f"data: {event_data}\n\n"
                        
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                        break
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"
                        break
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                done_data = json.dumps({
                    'done': True,
                    'stats': {
                        'chunks': chunk_count,
                        'chars': total_chars,
                        'duration_ms': elapsed_ms,
                        'model': model_id,
                    }
                }, ensure_ascii=False)
                yield f"data: {done_data}\n\n"
                
            except Exception as e:
                logger.exception(f"SSE error: {e}")
                yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"
            finally:
                loop.close()
        
        response = StreamingHttpResponse(
            generate_sse_events(),
            content_type='text/event-stream; charset=utf-8'
        )
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['X-Accel-Buffering'] = 'no'
        response['Connection'] = 'keep-alive'
        
        return response
    
    return [path('chat/stream', chat_stream)]


# =============================================================================
# HEALTH CHECK ROUTER
# =============================================================================

def health_router() -> List[path]:
    
    def health_simple(request: HttpRequest) -> JsonResponse:
        return JsonResponse({'status': 'ok', 'timestamp': int(time.time() * 1000)})
    
    def health_detailed(request: HttpRequest) -> JsonResponse:
        adapters_status = {}
        for name, adapter in ADAPTERS.items():
            try:
                has_key = hasattr(adapter, 'API_KEY') and bool(getattr(adapter, 'API_KEY', None))
                if name == 'hf':
                    has_key = hasattr(adapter, '_get_api_key') or hasattr(adapter, 'MODELS')
                adapters_status[name] = {'available': True, 'has_key': has_key}
            except Exception:
                adapters_status[name] = {'available': False}
        
        session_count = 0
        try:
            session_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
        except Exception:
            pass
        
        return JsonResponse({
            'status': 'ok',
            'timestamp': int(time.time() * 1000),
            'adapters': adapters_status,
            'sessions': session_count,
        })
    
    return [
        path('health/', health_simple),
        path('health/detailed/', health_detailed),
    ]


# =============================================================================
# DEBUG ROUTER (Development only)
# =============================================================================

def debug_router() -> List[path]:
    """Debug endpoint'leri - sadece development için."""
    
    @csrf_exempt
    def debug_models(request: HttpRequest) -> JsonResponse:
        """Tüm adaptörlerin model listelerini göster."""
        result = {
            'gemini': [],
            'hf': [],
            'ollama': [],
        }
        
        # Gemini
        if hasattr(gem, 'AVAILABLE_MODELS'):
            result['gemini'] = gem.AVAILABLE_MODELS
        
        # HuggingFace
        if hasattr(hf, 'MODELS'):
            result['hf'] = list(hf.MODELS.keys())
        
        # Ollama
        if hasattr(ol, 'MODEL_ALIASES'):
            result['ollama'] = list(ol.MODEL_ALIASES.keys())
        
        return JsonResponse(result)
    
    @csrf_exempt
    def debug_hf_health(request: HttpRequest) -> JsonResponse:
        """HuggingFace adaptör durumu."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                health = loop.run_until_complete(hf.health_check())
            finally:
                loop.close()
            return JsonResponse(health)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return [
        path('debug/models', debug_models),
        path('debug/hf-health', debug_hf_health),
    ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'models_router',
    'sessions_router',
    'chat_router',
    'sse_router',
    'health_router',
    'debug_router',
    'get_adapter',
]
