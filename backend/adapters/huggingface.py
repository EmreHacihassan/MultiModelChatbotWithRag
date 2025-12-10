"""
HuggingFace Inference Providers API Adapter - 2025 Version.
============================================================
HuggingFace Aralık 2024'te eski api-inference endpoint'ini kapattı.
Yeni sistem: router.huggingface.co üzerinden Inference Providers.

Çalışan Providers (Aralık 2025):
- together: Together AI (Llama modelleri)
- hyperbolic: Hyperbolic (Llama modelleri)

Dokümantasyon: https://huggingface.co/docs/inference-providers/index
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta

import httpx

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger('adapters.huggingface')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


# =============================================================================
# BLACKLIST (Geçici olarak çalışmayan modelleri engelle)
# =============================================================================

_blacklisted_until: Dict[str, datetime] = {}
BLACKLIST_DURATION = timedelta(minutes=10)

def _is_blacklisted(model_id: str) -> bool:
    if model_id not in _blacklisted_until:
        return False
    if datetime.now() >= _blacklisted_until[model_id]:
        del _blacklisted_until[model_id]
        return False
    return True

def _blacklist_model(model_id: str, duration: timedelta = BLACKLIST_DURATION):
    _blacklisted_until[model_id] = datetime.now() + duration
    logger.warning(f"Blacklist: {model_id} ({duration.total_seconds():.0f}s)")

def clear_blacklist():
    global _blacklisted_until
    _blacklisted_until = {}
    logger.info("Blacklist temizlendi")


# =============================================================================
# MODEL LİSTESİ (2025 - INFERENCE PROVIDERS İLE ÇALIŞAN MODELLER)
# =============================================================================
# HuggingFace Router üzerinden Together ve Hyperbolic provider'ları ile çalışan modeller.

MODELS = {
    # === TOGETHER PROVIDER (Ücretsiz) ===
    'hf-llama-3.2-3b': {
        'id': 'meta-llama/Llama-3.2-3B-Instruct-Turbo', 
        'provider': 'together',
        'name': 'Llama 3.2 3B', 
        'tier': 1,
        'description': 'Meta - Hızlı ve yetenekli'
    },
    'hf-llama-3.1-8b': {
        'id': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 
        'provider': 'together',
        'name': 'Llama 3.1 8B', 
        'tier': 1,
        'description': 'Meta - Güçlü ve dengeli'
    },
    'hf-llama-3.1-70b': {
        'id': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 
        'provider': 'together',
        'name': 'Llama 3.1 70B', 
        'tier': 1,
        'description': 'Meta - En güçlü açık kaynak'
    },
    'hf-qwen-2.5-7b': {
        'id': 'Qwen/Qwen2.5-7B-Instruct-Turbo', 
        'provider': 'together',
        'name': 'Qwen 2.5 7B', 
        'tier': 2,
        'description': 'Alibaba - Çok dilli'
    },
    'hf-qwen-2.5-72b': {
        'id': 'Qwen/Qwen2.5-72B-Instruct-Turbo', 
        'provider': 'together',
        'name': 'Qwen 2.5 72B', 
        'tier': 1,
        'description': 'Alibaba - En güçlü Qwen'
    },
    
    # === HYPERBOLIC PROVIDER (Ücretsiz) ===
    'hf-llama-3.2-3b-hyp': {
        'id': 'meta-llama/Llama-3.2-3B-Instruct', 
        'provider': 'hyperbolic',
        'name': 'Llama 3.2 3B (Hyp)', 
        'tier': 2,
        'description': 'Meta - Hyperbolic üzerinde'
    },
    'hf-qwen-2.5-72b-hyp': {
        'id': 'Qwen/Qwen2.5-72B-Instruct', 
        'provider': 'hyperbolic',
        'name': 'Qwen 2.5 72B (Hyp)', 
        'tier': 1,
        'description': 'Alibaba - Hyperbolic üzerinde'
    },
}

DEFAULT_MODEL = 'hf-llama-3.2-3b'

# HuggingFace Router base URL
HF_ROUTER_URL = "https://router.huggingface.co"


# =============================================================================
# API KEY
# =============================================================================

def _get_api_key() -> Optional[str]:
    """API anahtarını environment veya Django settings'ten al."""
    # 1. Environment Variables
    for env_var in ['HF_API_KEY', 'HF_TOKEN', 'HUGGINGFACE_API_KEY']:
        key = os.getenv(env_var)
        if key and key.strip(): 
            return key.strip()
    
    # 2. Django Settings
    try:
        from django.conf import settings
        if hasattr(settings, 'HF_API_KEY') and settings.HF_API_KEY:
            return settings.HF_API_KEY.strip()
    except: 
        pass
    
    return None


# =============================================================================
# MESAJ FORMATLAMA
# =============================================================================

def _format_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Mesajları OpenAI uyumlu chat format'ına dönüştür."""
    formatted = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        # Sadece user/assistant/system kabul ediliyor
        if role not in ['user', 'assistant', 'system']:
            role = 'user'
            
        formatted.append({'role': role, 'content': content})
    
    return formatted


# =============================================================================
# GENERATE - HUGGINGFACE INFERENCE PROVIDERS
# =============================================================================

async def generate(
    messages: List[Dict[str, Any]],
    model_id: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_new_tokens: int = 1024,
    fallback: bool = True,
) -> str:
    """
    HuggingFace Inference Providers API ile metin üret.
    
    Yeni API: router.huggingface.co/{provider}/v1/chat/completions
    """
    
    api_key = _get_api_key()
    if not api_key:
        return '[Hata] HF_API_KEY bulunamadı. Lütfen configs/env/.env dosyasına ekleyin.'

    # Model listesini hazırla
    if not fallback:
        available = [model_id] if model_id in MODELS else list(MODELS.keys())[:1]
    else:
        available = [m for m in MODELS.keys() if not _is_blacklisted(m)]
        if not available:
            clear_blacklist()
            available = list(MODELS.keys())
        
        # İstenen modeli öne al
        if model_id in available:
            available.remove(model_id)
            available.insert(0, model_id)

    # Mesajları formatla
    formatted_messages = _format_messages(messages)
    
    last_error = None

    for idx, try_model_key in enumerate(available, 1):
        if try_model_key not in MODELS:
            continue
        
        model_config = MODELS[try_model_key]
        provider = model_config['provider']
        model_name = model_config['id']
        
        logger.info(f"[{idx}/{len(available)}] Deneniyor: {model_config['name']} ({provider}/{model_name})")

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Provider endpoint'i
                url = f"{HF_ROUTER_URL}/{provider}/v1/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": model_name,
                    "messages": formatted_messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                
                logger.debug(f"İstek URL: {url}")
                
                response = await client.post(url, headers=headers, json=payload)
                
                # Başarılı yanıt
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # OpenAI uyumlu format
                        if 'choices' in data and len(data['choices']) > 0:
                            text = data['choices'][0].get('message', {}).get('content', '')
                            if text:
                                logger.info(f"✓ BAŞARILI: {model_config['name']}")
                                return text.strip()
                        
                        logger.warning(f"Boş yanıt: {data}")
                        last_error = "Boş yanıt"
                            
                    except Exception as e:
                        logger.error(f"JSON parse hatası: {e}")
                        last_error = f"Parse hatası: {e}"
                
                # Hata durumları
                elif response.status_code == 401:
                    return "[Hata] Geçersiz API anahtarı. Lütfen HF_API_KEY değerini kontrol edin."
                
                elif response.status_code == 400:
                    error_text = response.text[:200]
                    logger.warning(f"Bad Request (400): {error_text}")
                    last_error = f"Geçersiz istek: {error_text}"
                    if fallback:
                        _blacklist_model(try_model_key, timedelta(hours=1))
                
                elif response.status_code == 404:
                    logger.warning(f"Model bulunamadı (404): {model_name}")
                    last_error = "Model bulunamadı"
                    if fallback:
                        _blacklist_model(try_model_key, timedelta(hours=24))
                
                elif response.status_code == 429:
                    logger.warning(f"Rate limit aşıldı (429): {model_name}")
                    last_error = "Rate limit aşıldı"
                    await asyncio.sleep(2)
                
                elif response.status_code == 503:
                    logger.info(f"Servis kullanılamıyor (503): {model_name}")
                    last_error = "Servis geçici olarak kullanılamıyor"
                    await asyncio.sleep(5)
                
                else:
                    error_text = response.text[:300]
                    logger.warning(f"HTTP {response.status_code}: {error_text}")
                    last_error = f"HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            logger.warning(f"Timeout: {model_name}")
            last_error = "Zaman aşımı"
            
        except Exception as e:
            logger.error(f"Beklenmeyen hata ({try_model_key}): {e}")
            last_error = str(e)[:100]

    return f"[Hata] Tüm HuggingFace modelleri başarısız. Son hata: {last_error}"


# =============================================================================
# STREAM - Streaming yanıt
# =============================================================================

async def stream(
    messages: List[Dict[str, Any]], 
    model_id: str = DEFAULT_MODEL, 
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Streaming metin üretimi.
    HuggingFace Inference Providers SSE streaming destekliyor.
    """
    api_key = _get_api_key()
    if not api_key:
        yield '[Hata] HF_API_KEY bulunamadı.'
        return
    
    # Model seç
    if model_id not in MODELS:
        model_id = DEFAULT_MODEL
    
    model_config = MODELS[model_id]
    provider = model_config['provider']
    model_name = model_config['id']
    formatted_messages = _format_messages(messages)
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{HF_ROUTER_URL}/{provider}/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "max_tokens": kwargs.get('max_new_tokens', 1024),
                "temperature": kwargs.get('temperature', 0.7),
                "stream": True,
            }
            
            async with client.stream('POST', url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    yield f"[Hata] HTTP {response.status_code}: {error.decode()[:100]}"
                    return
                
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except:
                            pass
                            
    except Exception as e:
        logger.error(f"Stream hatası: {e}")
        # Fallback: normal generate kullan
        text = await generate(messages, model_id, **kwargs)
        words = text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)


# =============================================================================
# HEALTH CHECK
# =============================================================================

async def health_check() -> Dict[str, Any]:
    """Adaptör sağlık kontrolü."""
    api_key = _get_api_key()
    
    return {
        'ok': bool(api_key),
        'adapter': 'huggingface_providers_2025',
        'api_key_set': bool(api_key),
        'models_count': len(MODELS),
        'blacklisted_count': len(_blacklisted_until),
        'default_model': DEFAULT_MODEL,
        'providers': ['together', 'hyperbolic'],
    }


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def get_available_models() -> List[Dict[str, Any]]:
    """Kullanılabilir model listesini döndür."""
    return [
        {
            'id': key,
            'name': config['name'],
            'provider': 'hf',
            'streaming': True,
            'description': config.get('description', ''),
            'tier': config.get('tier', 5),
        }
        for key, config in MODELS.items()
        if not _is_blacklisted(key)
    ]