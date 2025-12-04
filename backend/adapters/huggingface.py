"""
HuggingFace Inference API Adapter - Güncel Modeller.
410 Gone hatası almayan endpoint'ler.
"""

import os
import asyncio
import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path

import httpx

# =============================================================================
# CONFIGURATION - GÜNCEL MODELLER (410 Gone almayan)
# =============================================================================

MODELS = {
    # Gemma 1.1 modelleri (güncel, çalışıyor)
    'hf-gemma-7b': {
        'endpoint': 'https://api-inference.huggingface.co/models/google/gemma-1.1-7b-it',
        'name': 'Gemma 1.1 7B IT',
    },
    'hf-gemma-2b': {
        'endpoint': 'https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it',
        'name': 'Gemma 1.1 2B IT',
    },
    # Alternatif modeller (daha güvenilir)
    'hf-mistral': {
        'endpoint': 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3',
        'name': 'Mistral 7B Instruct v0.3',
    },
    'hf-zephyr': {
        'endpoint': 'https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta',
        'name': 'Zephyr 7B Beta',
    },
    'hf-phi': {
        'endpoint': 'https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct',
        'name': 'Phi-3 Mini 4K',
    },
}

DEFAULT_MODEL = 'hf-mistral'  # Mistral daha güvenilir
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
    return MODELS[DEFAULT_MODEL]['endpoint']


def _build_prompt(messages: List[Dict[str, Any]], model_id: str = '') -> str:
    """Model'e göre prompt formatla."""
    if not messages:
        return ''
    
    # Mistral/Zephyr formatı
    if 'mistral' in model_id.lower() or 'zephyr' in model_id.lower():
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get('role', 'user')
            content = str(m.get('content', '')).strip()
            if not content:
                continue
            
            if role == 'user':
                parts.append(f"[INST] {content} [/INST]")
            elif role == 'assistant':
                parts.append(content)
            elif role == 'system':
                parts.insert(0, f"[INST] <<SYS>>\n{content}\n<</SYS>>\n[/INST]")
        
        return '\n'.join(parts)
    
    # Gemma formatı
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
        elif role == 'system':
            parts.append(f"<start_of_turn>user\n[System]: {content}<end_of_turn>")
    
    if parts:
        parts.append("<start_of_turn>model\n")
    
    return '\n'.join(parts)


def _clean_response(text: str, model_id: str = '') -> str:
    """Yanıtı temizle."""
    # Gemma token'ları
    text = text.replace('<end_of_turn>', '').strip()
    text = text.replace('<start_of_turn>', '').strip()
    
    # Mistral token'ları
    text = text.replace('[INST]', '').replace('[/INST]', '').strip()
    text = text.replace('<<SYS>>', '').replace('<</SYS>>', '').strip()
    
    # Model prefix'ini kaldır
    if text.startswith('model\n'):
        text = text[6:]
    
    return text.strip()


async def generate(
    messages: List[Dict[str, Any]],
    model_id: str = 'hf-mistral',
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> str:
    """HuggingFace ile yanıt üret (non-streaming)."""
    api_key = _get_api_key()
    
    if not api_key:
        return '[Hata] HF_API_KEY eksik.'
    
    # Model ID normalize
    if model_id not in MODELS:
        # Eski model ID'leri yenilere map'le
        if model_id in ['hf-gemma-7b', 'hf-gemma-2b']:
            pass  # Aynı kalabilir, endpoint güncellendi
        else:
            model_id = DEFAULT_MODEL
    
    prompt = _build_prompt(messages, model_id)
    
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
    
    # Denenecek modeller - en güvenilirden başla
    models_to_try = [model_id] + [m for m in ['hf-mistral', 'hf-zephyr', 'hf-phi'] if m != model_id]
    
    for try_model in models_to_try:
        if try_model not in MODELS:
            continue
            
        current_endpoint = MODELS[try_model]['endpoint']
        logger.info(f"HF deneniyor: {try_model}")
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                    response = await client.post(current_endpoint, headers=headers, json=payload)
                    
                    status = response.status_code
                    logger.debug(f"HF Response: {status} for {try_model}")
                    
                    if status in (404, 410):
                        logger.warning(f"Model unavailable: {try_model} ({status})")
                        break  # Sonraki modele geç
                    
                    if status == 429:
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        return '[Hata] Rate limit.'
                    
                    if status == 503:
                        if attempt < MAX_RETRIES:
                            logger.info(f"Model loading: {try_model}")
                            await asyncio.sleep(10)
                            continue
                        continue  # Sonraki modele geç
                    
                    if status == 401:
                        return '[Hata] Geçersiz HuggingFace API anahtarı.'
                    
                    if status == 200:
                        data = response.json()
                        
                        if isinstance(data, list) and data:
                            text = data[0].get('generated_text', '')
                            return _clean_response(text, try_model)
                        
                        return str(data)[:1000]
                    
                    # Diğer hatalar
                    logger.warning(f"HF unexpected status {status}")
                    break
                    
            except httpx.TimeoutException:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return '[Hata] Zaman aşımı.'
            except Exception as e:
                logger.error(f"HF error: {e}")
                break
    
    return '[Hata] HuggingFace modelleri şu an kullanılamıyor. Gemini kullanmayı deneyin.'


async def stream(
    messages: List[Dict[str, Any]],
    model_id: str = 'hf-mistral',
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """HuggingFace streaming - SSE desteği."""
    api_key = _get_api_key()
    
    if not api_key:
        yield '[Hata] HF_API_KEY eksik.'
        return
    
    if model_id not in MODELS:
        model_id = DEFAULT_MODEL
    
    prompt = _build_prompt(messages, model_id)
    
    if not prompt:
        yield '[Hata] Boş mesaj.'
        return
    
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
        },
        'stream': True,
    }
    
    models_to_try = [model_id] + [m for m in ['hf-mistral', 'hf-zephyr'] if m != model_id]
    
    for try_model in models_to_try:
        if try_model not in MODELS:
            continue
            
        current_endpoint = MODELS[try_model]['endpoint']
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10)) as client:
                async with client.stream('POST', current_endpoint, headers=headers, json=payload) as response:
                    
                    status = response.status_code
                    
                    if status in (404, 410, 503):
                        continue
                    
                    if status == 429:
                        yield '[Hata] Rate limit.'
                        return
                    
                    if status == 401:
                        yield '[Hata] Geçersiz API anahtarı.'
                        return
                    
                    if status != 200:
                        continue
                    
                    # SSE streaming
                    buffer = ''
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line or not line.startswith('data:'):
                                continue
                            
                            json_str = line[5:].strip()
                            if not json_str:
                                continue
                            
                            try:
                                data = json.loads(json_str)
                                token = data.get('token', {}).get('text', '')
                                if token:
                                    # Özel token'ları filtrele
                                    skip_tokens = ['<end_of_turn>', '<start_of_turn>', 
                                                   '[INST]', '[/INST]', 'model', 'user',
                                                   '<<SYS>>', '<</SYS>>']
                                    if token not in skip_tokens:
                                        yield token
                            except json.JSONDecodeError:
                                pass
                    
                    return
                    
        except Exception as e:
            logger.error(f"HF Stream error: {e}")
            continue
    
    # Fallback: non-streaming
    logger.info("HF streaming failed, using non-streaming")
    result = await generate(messages, model_id, temperature, max_new_tokens)
    yield result


async def health_check() -> Dict[str, Any]:
    """Health check."""
    api_key = _get_api_key()
    if not api_key:
        return {'ok': False, 'error': 'API key missing'}
    
    return {
        'ok': True, 
        'model': 'HuggingFace (Mistral/Gemma/Zephyr)',
        'note': 'Key configured',
        'available_models': list(MODELS.keys())
    }