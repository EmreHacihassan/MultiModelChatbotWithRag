"""
Ollama Adapter - Yerel LLM desteği için async adapter.

Özellikler:
- Gerçek streaming desteği (Ollama native)
- Async uyumlu (time.sleep yerine asyncio.sleep)
- Hata yönetimi ve retry logic
- Model alias desteği
- Health check
"""

import os
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ollama sunucu adresi
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Timeout ayarları
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120  # Streaming için uzun timeout
WRITE_TIMEOUT = 30

# Model alias mapping
MODEL_ALIASES = {
    'ollama:qwen': 'qwen2.5:latest',
    'ollama:qwen2': 'qwen2.5:latest',
    'ollama:llama': 'llama3.2:latest',
    'ollama:llama3': 'llama3.2:latest',
    'ollama:mistral': 'mistral:latest',
    'ollama:codellama': 'codellama:latest',
    'ollama:phi': 'phi3:latest',
}

# Varsayılan model
DEFAULT_MODEL = 'qwen2.5:latest'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _resolve_model(model_id: str) -> str:
    """
    Model ID'yi Ollama model adına çevir.
    
    Örnekler:
        'ollama:qwen' -> 'qwen2.5:latest'
        'qwen' -> 'qwen'
        'ollama:llama' -> 'llama3.2:latest'
    """
    if not model_id:
        return DEFAULT_MODEL
    
    # Alias kontrolü
    if model_id.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[model_id.lower()]
    
    # 'ollama:' prefix'i varsa ayır
    if ':' in model_id and model_id.lower().startswith('ollama:'):
        name = model_id.split(':', 1)[1]
        return name if name else DEFAULT_MODEL
    
    return model_id


def _extract_user_message(messages: List[Dict[str, Any]]) -> str:
    """Mesaj listesinden son kullanıcı mesajını çıkar."""
    if not messages:
        return ''
    
    for m in reversed(messages):
        if isinstance(m, dict) and m.get('role') == 'user':
            return str(m.get('content', ''))
    
    return ''


def _build_prompt(messages: List[Dict[str, Any]], include_history: bool = True) -> str:
    """
    Mesaj listesinden Ollama prompt'u oluştur.
    
    Args:
        messages: Mesaj listesi
        include_history: Tüm geçmişi dahil et (True) veya sadece son mesaj (False)
    
    Returns:
        Formatlanmış prompt string
    """
    if not messages:
        return 'Merhaba'
    
    if not include_history:
        return _extract_user_message(messages)
    
    # Tüm mesajları prompt'a çevir
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        
        role = m.get('role', 'user')
        content = str(m.get('content', '')).strip()
        
        if not content:
            continue
        
        if role == 'system':
            parts.append(f"[System]: {content}")
        elif role == 'user':
            parts.append(f"[User]: {content}")
        elif role == 'assistant':
            parts.append(f"[Assistant]: {content}")
    
    if not parts:
        return 'Merhaba'
    
    return '\n\n'.join(parts) + '\n\n[Assistant]:'


def _get_httpx_timeout() -> httpx.Timeout:
    """Configured timeout object."""
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=CONNECT_TIMEOUT,
    )


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

async def generate(
    messages: List[Dict[str, Any]],
    model_id: str = 'ollama:qwen',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    include_history: bool = False,
) -> str:
    """
    Ollama ile tek seferlik yanıt üret (non-streaming).
    
    Args:
        messages: Mesaj listesi [{"role": "user", "content": "..."}]
        model_id: Model ID (ollama:qwen, ollama:llama, vb.)
        temperature: Yaratıcılık (0.0-2.0)
        max_tokens: Maksimum token
        include_history: Mesaj geçmişini dahil et
    
    Returns:
        Üretilen metin yanıtı
    """
    model_name = _resolve_model(model_id)
    prompt = _build_prompt(messages, include_history)
    
    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }
    
    try:
        async with httpx.AsyncClient(timeout=_get_httpx_timeout()) as client:
            response = await client.post(
                f'{OLLAMA_HOST}/api/generate',
                json=payload,
            )
            
            if response.status_code == 404:
                return f'[Hata] Model bulunamadı: {model_name}. "ollama pull {model_name}" ile indirin.'
            
            response.raise_for_status()
            data = response.json()
            
            return (data.get('response') or '')[:4000]
            
    except httpx.ConnectError:
        return f'[Hata] Ollama sunucusuna bağlanılamadı ({OLLAMA_HOST}). Ollama çalışıyor mu?'
    except httpx.TimeoutException:
        return '[Hata] İstek zaman aşımına uğradı. Model yükleniyor olabilir, tekrar deneyin.'
    except httpx.HTTPStatusError as e:
        return f'[HTTP Hatası] {e.response.status_code}: {str(e)[:200]}'
    except Exception as e:
        return f'[Beklenmeyen Hata] {str(e)[:300]}'


async def stream(
    messages: List[Dict[str, Any]],
    model_id: str = 'ollama:qwen',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    include_history: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Ollama ile streaming yanıt üret.
    
    Ollama gerçek streaming destekler - her token anında gönderilir.
    
    Yields:
        Metin token'ları
    """
    model_name = _resolve_model(model_id)
    prompt = _build_prompt(messages, include_history)
    
    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': True,  # ✅ Gerçek streaming
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }
    
    try:
        async with httpx.AsyncClient(timeout=_get_httpx_timeout()) as client:
            async with client.stream(
                'POST',
                f'{OLLAMA_HOST}/api/generate',
                json=payload,
            ) as response:
                
                if response.status_code == 404:
                    yield f'[Hata] Model bulunamadı: {model_name}'
                    return
                
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        delta = data.get('response', '')
                        
                        if delta:
                            yield delta
                        
                        # Ollama done sinyali
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        # Geçersiz JSON satırını atla
                        continue
                    
                    # ✅ Async uyumlu küçük gecikme (backpressure)
                    await asyncio.sleep(0.005)
                    
    except httpx.ConnectError:
        yield f'[Hata] Ollama sunucusuna bağlanılamadı ({OLLAMA_HOST})'
    except httpx.TimeoutException:
        yield '[Hata] İstek zaman aşımına uğradı'
    except httpx.HTTPStatusError as e:
        yield f'[HTTP Hatası] {e.response.status_code}'
    except Exception as e:
        yield f'[Hata] {str(e)[:200]}'


# =============================================================================
# BONUS: CHAT API (Multi-turn conversation)
# =============================================================================

async def chat(
    messages: List[Dict[str, Any]],
    model_id: str = 'ollama:qwen',
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Ollama /api/chat endpoint'i ile multi-turn sohbet.
    
    Bu endpoint mesaj geçmişini daha iyi işler.
    """
    model_name = _resolve_model(model_id)
    
    # Mesajları Ollama chat formatına çevir
    ollama_messages = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get('role', 'user')
        content = str(m.get('content', ''))
        if content:
            ollama_messages.append({'role': role, 'content': content})
    
    if not ollama_messages:
        ollama_messages = [{'role': 'user', 'content': 'Merhaba'}]
    
    payload = {
        'model': model_name,
        'messages': ollama_messages,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }
    
    try:
        async with httpx.AsyncClient(timeout=_get_httpx_timeout()) as client:
            response = await client.post(
                f'{OLLAMA_HOST}/api/chat',
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get('message', {}).get('content', '')[:4000]
            
    except Exception as e:
        return f'[Hata] {str(e)[:300]}'


async def chat_stream(
    messages: List[Dict[str, Any]],
    model_id: str = 'ollama:qwen',
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    Ollama /api/chat endpoint'i ile streaming multi-turn sohbet.
    """
    model_name = _resolve_model(model_id)
    
    ollama_messages = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get('role', 'user')
        content = str(m.get('content', ''))
        if content:
            ollama_messages.append({'role': role, 'content': content})
    
    if not ollama_messages:
        ollama_messages = [{'role': 'user', 'content': 'Merhaba'}]
    
    payload = {
        'model': model_name,
        'messages': ollama_messages,
        'stream': True,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }
    
    try:
        async with httpx.AsyncClient(timeout=_get_httpx_timeout()) as client:
            async with client.stream(
                'POST',
                f'{OLLAMA_HOST}/api/chat',
                json=payload,
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        delta = data.get('message', {}).get('content', '')
                        
                        if delta:
                            yield delta
                        
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    
                    await asyncio.sleep(0.005)
                    
    except Exception as e:
        yield f'[Hata] {str(e)[:200]}'


# =============================================================================
# BONUS: UTILITY FUNCTIONS
# =============================================================================

async def list_models() -> List[Dict[str, Any]]:
    """Ollama'da yüklü modelleri listele."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f'{OLLAMA_HOST}/api/tags')
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    'id': f"ollama:{m['name']}",
                    'name': m['name'],
                    'size': m.get('size', 0),
                    'modified': m.get('modified_at', ''),
                }
                for m in data.get('models', [])
            ]
    except Exception:
        return []


async def health_check() -> Dict[str, Any]:
    """Ollama sunucu durumunu kontrol et."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f'{OLLAMA_HOST}/api/tags')
            
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                return {
                    'ok': True,
                    'host': OLLAMA_HOST,
                    'models': models,
                    'model_count': len(models),
                }
            
            return {'ok': False, 'error': f'Status {response.status_code}'}
            
    except httpx.ConnectError:
        return {'ok': False, 'error': 'Connection refused', 'host': OLLAMA_HOST}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


async def pull_model(model_name: str) -> AsyncGenerator[str, None]:
    """
    Model indir (streaming progress).
    
    Kullanım:
        async for status in pull_model('llama3.2'):
            print(status)
    """
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                'POST',
                f'{OLLAMA_HOST}/api/pull',
                json={'name': model_name},
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        status = data.get('status', '')
                        
                        if 'completed' in data and 'total' in data:
                            pct = int(data['completed'] / data['total'] * 100)
                            yield f"{status}: {pct}%"
                        else:
                            yield status
                            
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield f"[Hata] {str(e)}"