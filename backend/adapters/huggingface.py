"""
HuggingFace Inference API Adapter - Google Gemma.
Gerçek streaming desteği.
"""

import os
import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    'hf-gemma-7b': {
        'endpoint': 'https://api-inference.huggingface.co/models/google/gemma-7b-it',
        'name': 'Gemma 7B IT',
    },
    'hf-gemma-2b': {
        'endpoint': 'https://api-inference.huggingface.co/models/google/gemma-2b-it',
        'name': 'Gemma 2B IT',
    },
}

DEFAULT_MODEL = 'hf-gemma-7b'
DEFAULT_ENDPOINT = MODELS[DEFAULT_MODEL]['endpoint']
MAX_RETRIES = 3
RETRY_DELAY = 2
TIMEOUT_SECONDS = 120

logger = logging.getLogger('adapters.huggingface')


def _get_api_key() -> Optional[str]:
    """API anahtarını al."""
    key = os.getenv('HF_API_KEY')
    if key and key.strip():
        return key.strip()
    
    try:
        from django.conf import settings
        key = getattr(settings, 'HF_API_KEY', None)
        if key and key.strip():
            return key.strip()
    except Exception:
        pass
    
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]
        env_path = project_root / 'configs' / 'env' / '.env'
        
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or '=' not in line:
                        continue
                    if line.startswith('HF_API_KEY='):
                        value = line.split('=', 1)[1].strip()
                        for quote in ['"', "'"]:
                            if value.startswith(quote) and value.endswith(quote):
                                value = value[1:-1]
                        if value:
                            return value
    except Exception:
        pass
    
    return None


def _get_endpoint(model_id: str) -> str:
    """Model endpoint al."""
    if model_id in MODELS:
        return MODELS[model_id]['endpoint']
    return DEFAULT_ENDPOINT


def _build_prompt(messages: List[Dict[str, Any]]) -> str:
    """Gemma formatında prompt oluştur."""
    if not messages:
        return ''
    
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', 'user')
        content = str(m.get('content', '')).strip()
        if not content:
            continue
        
        if role == 'user':
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == 'assistant':
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    
    if parts:
        parts.append("<start_of_turn>model\n")
    
    return '\n'.join(parts)


def _clean_response(text: str) -> str:
    """Yanıtı temizle."""
    if '<start_of_turn>model' in text:
        parts = text.split('<start_of_turn>model')
        text = parts[-1]
    
    text = text.replace('<end_of_turn>', '').strip()
    text = text.replace('<start_of_turn>', '').strip()
    return text


async def generate(
    messages: List[Dict[str, Any]],
    model_id: str = 'hf-gemma-7b',
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> str:
    """HuggingFace Gemma ile yanıt üret (non-streaming)."""
    api_key = _get_api_key()
    
    if not api_key:
        return '[Hata] HF_API_KEY eksik.'
    
    if model_id not in MODELS:
        model_id = DEFAULT_MODEL
    
    endpoint = _get_endpoint(model_id)
    prompt = _build_prompt(messages)
    
    if not prompt:
        return '[Hata] Boş mesaj.'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': max_new_tokens,
            'temperature': max(0.01, temperature),
            'top_p': 0.95,
            'do_sample': True,
            'return_full_text': False,
        },
        'options': {
            'wait_for_model': True,
            'use_cache': True,
        },
    }
    
    fallback_models = ['hf-gemma-7b', 'hf-gemma-2b']
    
    for try_model in fallback_models:
        current_endpoint = MODELS.get(try_model, {}).get('endpoint', endpoint)
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                    response = await client.post(current_endpoint, headers=headers, json=payload)
                    
                    if response.status_code == 404:
                        break
                    if response.status_code == 410:
                        break
                    if response.status_code == 429:
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        return '[Hata] Rate limit.'
                    if response.status_code == 503:
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(10)
                            continue
                        return '[Hata] Model yükleniyor.'
                    if response.status_code == 401:
                        return '[Hata] Geçersiz API anahtarı.'
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if isinstance(data, list) and data:
                        text = data[0].get('generated_text', '')
                        return _clean_response(text)
                    
                    return str(data)[:1000]
                    
            except httpx.TimeoutException:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return '[Hata] Zaman aşımı.'
            except Exception as e:
                break
    
    return '[Hata] Tüm modeller başarısız.'


async def stream(
    messages: List[Dict[str, Any]],
    model_id: str = 'hf-gemma-7b',
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """
    GERÇEK Streaming yanıt - HuggingFace Text Generation API.
    """
    api_key = _get_api_key()
    
    if not api_key:
        yield '[Hata] HF_API_KEY eksik.'
        return
    
    if model_id not in MODELS:
        model_id = DEFAULT_MODEL
    
    endpoint = _get_endpoint(model_id)
    prompt = _build_prompt(messages)
    
    if not prompt:
        yield '[Hata] Boş mesaj.'
        return
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    # Streaming için stream: true parametresi
    payload = {
        'inputs': prompt,
        'parameters': {
            'max_new_tokens': max_new_tokens,
            'temperature': max(0.01, temperature),
            'top_p': 0.95,
            'do_sample': True,
            'return_full_text': False,
        },
        'options': {
            'wait_for_model': True,
        },
        'stream': True,  # STREAMING AKTİF
    }
    
    fallback_models = ['hf-gemma-7b', 'hf-gemma-2b']
    
    for try_model in fallback_models:
        current_endpoint = MODELS.get(try_model, {}).get('endpoint', endpoint)
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10)) as client:
                async with client.stream('POST', current_endpoint, headers=headers, json=payload) as response:
                    
                    if response.status_code in (404, 410):
                        continue
                    if response.status_code == 429:
                        yield '[Hata] Rate limit.'
                        return
                    if response.status_code == 401:
                        yield '[Hata] Geçersiz API anahtarı.'
                        return
                    if response.status_code == 503:
                        continue
                    
                    if response.status_code != 200:
                        continue
                    
                    # SSE formatında streaming oku
                    buffer = ''
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            # "data:" prefix
                            if line.startswith('data:'):
                                json_str = line[5:].strip()
                                
                                if not json_str:
                                    continue
                                
                                try:
                                    data = json.loads(json_str)
                                    # Token formatı
                                    token = data.get('token', {}).get('text', '')
                                    if token:
                                        # Özel token'ları filtrele
                                        if token not in ['<end_of_turn>', '<start_of_turn>', 'model', 'user']:
                                            yield token
                                except json.JSONDecodeError:
                                    pass
                    
                    return  # Başarılı
                    
        except httpx.TimeoutException:
            continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            continue
    
    # Fallback: non-streaming
    result = await generate(messages, model_id, temperature, max_new_tokens)
    yield result


async def health_check() -> Dict[str, Any]:
    """Health check."""
    api_key = _get_api_key()
    if not api_key:
        return {'ok': False, 'error': 'API key missing'}
    
    try:
        result = await generate([{'role': 'user', 'content': 'Merhaba'}], max_new_tokens=10)
        return {'ok': not result.startswith('[Hata]'), 'model': 'Google Gemma'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}