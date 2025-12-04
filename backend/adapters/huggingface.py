"""
HuggingFace Inference API Adapter - Hybrid Version.
Tries official client first, falls back to direct HTTP for maximum reliability.
"""

import os
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import httpx
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

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
# BLACKLIST
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
# MODEL LISTESI (2025 Safe List)
# =============================================================================

MODELS = {
    'hf-gpt2': {'id': 'openai-community/gpt2', 'name': 'GPT-2', 'tier': 1},
    'hf-flan-t5-base': {'id': 'google/flan-t5-base', 'name': 'Flan-T5 Base', 'tier': 1},
    'hf-tinyllama': {'id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'name': 'TinyLlama 1.1B', 'tier': 2},
    'hf-phi-2': {'id': 'microsoft/phi-2', 'name': 'Phi-2', 'tier': 2},
    'hf-bloom-560m': {'id': 'bigscience/bloom-560m', 'name': 'BLOOM 560M', 'tier': 3},
    'hf-gpt-neo-125m': {'id': 'EleutherAI/gpt-neo-125M', 'name': 'GPT-Neo 125M', 'tier': 3},
    'hf-qwen-05b': {'id': 'Qwen/Qwen1.5-0.5B-Chat', 'name': 'Qwen 1.5 0.5B', 'tier': 3},
    'hf-opt-350m': {'id': 'facebook/opt-350m', 'name': 'OPT 350M', 'tier': 3},
}

DEFAULT_MODEL = 'hf-gpt2'


# =============================================================================
# API KEY
# =============================================================================

def _get_api_key() -> Optional[str]:
    # 1. Environment Variables
    for env_var in ['HF_API_KEY', 'HF_TOKEN', 'HUGGINGFACE_API_KEY']:
        key = os.getenv(env_var)
        if key and key.strip(): return key.strip()
    
    # 2. Django Settings
    try:
        from django.conf import settings
        if hasattr(settings, 'HF_API_KEY') and settings.HF_API_KEY:
            return settings.HF_API_KEY.strip()
    except: pass
    
    return None


# =============================================================================
# GENERATE
# =============================================================================

async def generate(
    messages: List[Dict[str, Any]],
    model_id: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_new_tokens: int = 200,
    fallback: bool = True,
) -> str:
    
    api_key = _get_api_key()
    if not api_key:
        return '[Hata] HF_API_KEY bulunamadı.'

    # Model listesini hazırla
    if not fallback:
        available = [model_id]
    else:
        available = [m for m in MODELS.keys() if not _is_blacklisted(m)]
        if not available:
            clear_blacklist()
            available = list(MODELS.keys())
        if model_id in available:
            available.remove(model_id)
            available.insert(0, model_id)
        available.sort(key=lambda x: MODELS[x].get('tier', 99))

    # Prompt hazırla
    prompt = ""
    for m in messages:
        prompt += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
    prompt += "assistant:"

    last_error = None
    tried_count = 0

    for try_model_key in available:
        tried_count += 1
        if try_model_key not in MODELS: continue
        
        model_config = MODELS[try_model_key]
        hf_model_id = model_config['id']
        logger.info(f"[{tried_count}/{len(available)}] Deneniyor: {model_config['name']} ({hf_model_id})")

        # --- YÖNTEM 1: Resmi Client ---
        try:
            client = AsyncInferenceClient(token=api_key)
            response = await client.text_generation(
                prompt, model=hf_model_id, max_new_tokens=max_new_tokens, temperature=temperature
            )
            if response:
                logger.info(f"✓ BAŞARILI (Client): {model_config['name']}")
                return response.strip()
        except Exception as e:
            logger.debug(f"Client hatası ({try_model_key}): {e}")
            # Hata devam ederse HTTP fallback'e geç

        # --- YÖNTEM 2: Doğrudan HTTP (Fallback) ---
        try:
            async with httpx.AsyncClient(timeout=30) as http_client:
                resp = await http_client.post(
                    f"https://api-inference.huggingface.co/models/{hf_model_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}}
                )
                
                if resp.status_code == 200:
                    try:
                        # Yanıt formatı bazen liste bazen dict olabilir
                        data = resp.json()
                        text = ""
                        if isinstance(data, list) and len(data) > 0:
                            text = data[0].get('generated_text', '')
                        elif isinstance(data, dict):
                            text = data.get('generated_text', '')
                        
                        # Prompt'u temizle
                        if text.startswith(prompt):
                            text = text[len(prompt):]
                            
                        if text.strip():
                            logger.info(f"✓ BAŞARILI (HTTP): {model_config['name']}")
                            return text.strip()
                    except:
                        pass
                
                # Hata analizi
                error_text = resp.text
                logger.warning(f"HTTP Hatası ({try_model_key}): {resp.status_code} - {error_text[:200]}")
                
                if resp.status_code in [401, 403]:
                    return "[Hata] API Yetkilendirme Hatası (403). Lütfen Hugging Face Token izinlerini kontrol edin (Inference API yetkisi gerekli)."
                if resp.status_code == 503:
                    logger.info("Model yükleniyor...")
                    await asyncio.sleep(5)
                
                if fallback: _blacklist_model(try_model_key, timedelta(minutes=30))
                last_error = f"HTTP {resp.status_code}"

        except Exception as e:
            logger.error(f"Beklenmeyen hata ({try_model_key}): {e}")
            last_error = str(e)

    return f"[Hata] Hiçbir model yanıt vermedi. Son hata: {last_error}"

async def stream(messages, model_id=DEFAULT_MODEL, **kwargs):
    text = await generate(messages, model_id, **kwargs)
    for word in text.split():
        yield word + " "
        await asyncio.sleep(0.05)

async def health_check():
    return {'ok': True, 'adapter': 'huggingface_hybrid'}