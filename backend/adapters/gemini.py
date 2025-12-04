"""
Gemini API Adapter - Google AI Studio Desteği.

Bu modül Google Gemini API ile iletişim kurar ve gerçek zamanlı
streaming yanıt desteği sağlar. Her token anında yield edilir.

Desteklenen Modeller:
- gemini-2.5-flash-preview-05-20 (varsayılan, en yeni)
- gemini-2.0-flash
- gemini-1.5-flash-latest
- gemini-1.5-pro-latest

Özellikler:
- Gerçek SSE streaming (streamGenerateContent API)
- Otomatik model fallback
- Retry mekanizması
- Detaylı hata yönetimi
- API key çoklu kaynak desteği
"""

import os
import json
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path

import httpx

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger('adapters.gemini')
logger.setLevel(logging.DEBUG)

# Console handler ekle (eğer yoksa)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# =============================================================================
# CONFIGURATION - MODEL MAPPING
# =============================================================================

# Kullanıcı dostu ID -> Gerçek Gemini model adı
MODEL_MAP: Dict[str, str] = {
    # Gemini 2.5 modelleri (en yeni, varsayılan)
    'gemini-flash': 'gemini-2.5-flash-preview-05-20',
    'gemini-2.5-flash': 'gemini-2.5-flash-preview-05-20',
    'gemini-2-flash': 'gemini-2.5-flash-preview-05-20',
    
    # Gemini 2.0 modelleri
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'gemini-2.0': 'gemini-2.0-flash',
    
    # Gemini 1.5 modelleri (yedek)
    'gemini-1.5-flash': 'gemini-1.5-flash-latest',
    'gemini-1.5-flash-latest': 'gemini-1.5-flash-latest',
    'gemini-pro': 'gemini-1.5-pro-latest',
    'gemini-1.5-pro': 'gemini-1.5-pro-latest',
    'gemini-1.5-pro-latest': 'gemini-1.5-pro-latest',
}

# Varsayılan model
DEFAULT_MODEL: str = 'gemini-2.5-flash-preview-05-20'

# Fallback model sırası (birincil başarısız olursa sırayla dene)
FALLBACK_MODELS: List[str] = [
    'gemini-2.5-flash-preview-05-20',
    'gemini-2.0-flash',
    'gemini-1.5-flash-latest',
]

# API endpoint
API_BASE_URL: str = 'https://generativelanguage.googleapis.com/v1beta/models'

# Timeout ayarları (saniye)
TIMEOUT_TOTAL: int = 180  # Toplam timeout
TIMEOUT_CONNECT: int = 15  # Bağlantı timeout
TIMEOUT_READ: int = 120   # Okuma timeout

# Retry ayarları
MAX_RETRIES: int = 3
RETRY_DELAY_BASE: float = 1.5  # Exponential backoff base

# Streaming ayarları
STREAM_CHUNK_SIZE: int = 1024  # Byte cinsinden chunk boyutu


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

def _get_api_key() -> Optional[str]:
    """
    Gemini API anahtarını çoklu kaynaktan al.
    
    Öncelik sırası:
    1. GEMINI_API_KEY environment variable
    2. Django settings.GEMINI_API_KEY
    3. .env dosyasından manuel okuma
    
    Returns:
        API key string veya None
    """
    # Kaynak 1: Environment variable
    key = os.getenv('GEMINI_API_KEY')
    if key and key.strip():
        logger.debug("API key loaded from environment variable")
        return key.strip()
    
    # Kaynak 2: Django settings
    try:
        from django.conf import settings
        key = getattr(settings, 'GEMINI_API_KEY', None)
        if key and key.strip():
            logger.debug("API key loaded from Django settings")
            return key.strip()
    except Exception as e:
        logger.debug(f"Could not load from Django settings: {e}")
    
    # Kaynak 3: .env dosyasını manuel oku
    try:
        # Bu dosyanın konumu: MyChatbot/backend/adapters/gemini.py
        current_file = Path(__file__).resolve()
        # Project root: MyChatbot/
        project_root = current_file.parents[2]
        # .env dosyası: MyChatbot/configs/env/.env
        env_path = project_root / 'configs' / 'env' / '.env'
        
        logger.debug(f"Looking for .env at: {env_path}")
        
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Boş satır veya yorum satırını atla
                    if not line or line.startswith('#'):
                        continue
                    
                    # Key=Value formatını parse et
                    if '=' not in line:
                        continue
                    
                    # GEMINI_API_KEY satırını bul
                    if line.startswith('GEMINI_API_KEY='):
                        value = line.split('=', 1)[1].strip()
                        
                        # Tırnak işaretlerini kaldır
                        for quote_char in ['"', "'"]:
                            if value.startswith(quote_char) and value.endswith(quote_char):
                                value = value[1:-1]
                                break
                        
                        if value:
                            logger.debug("API key loaded from .env file")
                            return value
    except Exception as e:
        logger.warning(f"Could not read .env file: {e}")
    
    logger.error("No Gemini API key found in any source")
    return None


# =============================================================================
# MODEL RESOLUTION
# =============================================================================

def _resolve_model(model_id: str) -> str:
    """
    Kullanıcı dostu model ID'yi gerçek Gemini model adına çevir.
    
    Args:
        model_id: Kullanıcı dostu model ID (örn: "gemini-flash")
    
    Returns:
        Gerçek Gemini model adı (örn: "gemini-2.5-flash-preview-05-20")
    """
    if not model_id:
        logger.debug(f"No model_id provided, using default: {DEFAULT_MODEL}")
        return DEFAULT_MODEL
    
    # Lowercase ile eşleştir
    normalized = model_id.lower().strip()
    
    # Direkt eşleşme kontrolü
    if normalized in MODEL_MAP:
        resolved = MODEL_MAP[normalized]
        logger.debug(f"Model resolved: {model_id} -> {resolved}")
        return resolved
    
    # Eğer zaten gerçek model adıysa olduğu gibi kullan
    if normalized.startswith('gemini-') and ('flash' in normalized or 'pro' in normalized):
        logger.debug(f"Model appears to be actual name: {model_id}")
        return model_id
    
    # Bilinmeyen model, varsayılanı kullan
    logger.warning(f"Unknown model '{model_id}', falling back to {DEFAULT_MODEL}")
    return DEFAULT_MODEL


# =============================================================================
# MESSAGE FORMATTING
# =============================================================================

def _build_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mesaj listesini Gemini API formatına çevir.
    
    Gemini API formatı:
    {
        "contents": [
            {"role": "user", "parts": [{"text": "..."}]},
            {"role": "model", "parts": [{"text": "..."}]},
        ]
    }
    
    Args:
        messages: OpenAI formatında mesaj listesi
                  [{"role": "user", "content": "..."}]
    
    Returns:
        Gemini formatında contents listesi
    """
    contents: List[Dict[str, Any]] = []
    
    # Role mapping: OpenAI -> Gemini
    role_mapping = {
        'user': 'user',
        'assistant': 'model',
        'system': 'user',  # Gemini'de system role yok, user olarak ekle
    }
    
    for msg in messages or []:
        # Dict kontrolü
        if not isinstance(msg, dict):
            logger.warning(f"Skipping non-dict message: {type(msg)}")
            continue
        
        # Role ve content al
        original_role = msg.get('role', 'user')
        role = role_mapping.get(original_role, 'user')
        content = str(msg.get('content', '')).strip()
        
        # Boş içerik kontrolü
        if not content:
            logger.debug(f"Skipping empty message with role: {original_role}")
            continue
        
        # System mesajını özel işle
        if original_role == 'system':
            content = f"[System Instruction]: {content}"
        
        # Gemini formatında ekle
        contents.append({
            'role': role,
            'parts': [{'text': content}]
        })
    
    # Boş contents kontrolü - en az bir mesaj olmalı
    if not contents:
        logger.warning("No valid messages found, adding default greeting")
        contents.append({
            'role': 'user',
            'parts': [{'text': 'Merhaba'}]
        })
    
    logger.debug(f"Built {len(contents)} content items for Gemini API")
    return contents


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def _extract_text_from_response(data: Dict[str, Any]) -> str:
    """
    Gemini API yanıtından text içeriğini çıkar.
    
    Gemini yanıt formatı:
    {
        "candidates": [{
            "content": {
                "parts": [{"text": "..."}],
                "role": "model"
            },
            "finishReason": "STOP"
        }]
    }
    
    Args:
        data: Gemini API JSON yanıtı
    
    Returns:
        Çıkarılan text içeriği veya hata mesajı
    """
    try:
        # Candidates array'ini al
        candidates = data.get('candidates', [])
        
        if not candidates:
            # Error response kontrolü
            if 'error' in data:
                error_info = data['error']
                error_message = error_info.get('message', 'Bilinmeyen hata')
                error_code = error_info.get('code', 'UNKNOWN')
                logger.error(f"Gemini API error: [{error_code}] {error_message}")
                return f"[Gemini Hatası] {error_message}"
            
            logger.warning("No candidates in response")
            return ''
        
        # İlk candidate'i al
        first_candidate = candidates[0]
        
        # Content al
        content = first_candidate.get('content', {})
        
        # Parts array'ini al
        parts = content.get('parts', [])
        
        if not parts:
            logger.warning("No parts in candidate content")
            return ''
        
        # İlk part'tan text al
        first_part = parts[0]
        text = first_part.get('text', '')
        
        # Finish reason kontrolü
        finish_reason = first_candidate.get('finishReason', 'UNKNOWN')
        if finish_reason not in ('STOP', 'MAX_TOKENS', 'UNKNOWN'):
            logger.warning(f"Unusual finish reason: {finish_reason}")
        
        return text
        
    except Exception as e:
        logger.exception(f"Error parsing Gemini response: {e}")
        return f"[Parse Hatası] {str(e)}"


def _parse_sse_line(line: str) -> Optional[str]:
    """
    SSE (Server-Sent Events) satırını parse et ve text çıkar.
    
    SSE formatı:
    data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
    
    Args:
        line: SSE satırı
    
    Returns:
        Çıkarılan text veya None
    """
    # Boş satır kontrolü
    line = line.strip()
    if not line:
        return None
    
    # "data: " prefix kontrolü
    if not line.startswith('data:'):
        return None
    
    # JSON kısmını al
    json_str = line[5:].strip()
    
    # Boş veya [DONE] kontrolü
    if not json_str or json_str == '[DONE]':
        return None
    
    try:
        data = json.loads(json_str)
        text = _extract_text_from_response(data)
        return text if text and not text.startswith('[') else None
    except json.JSONDecodeError as e:
        logger.debug(f"SSE JSON parse error: {e}")
        return None
    except Exception as e:
        logger.debug(f"SSE parse error: {e}")
        return None


# =============================================================================
# NON-STREAMING GENERATION
# =============================================================================

async def generate(
    messages: List[Dict[str, Any]], 
    model_id: str = 'gemini-flash',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    top_k: int = 40,
) -> str:
    """
    Gemini API ile tek seferlik (non-streaming) yanıt üret.
    
    Bu fonksiyon yanıtın tamamını bekler ve döndürür.
    Streaming için stream() fonksiyonunu kullanın.
    
    Args:
        messages: OpenAI formatında mesaj listesi
        model_id: Kullanılacak model ID
        temperature: Yaratıcılık seviyesi (0.0-1.0)
        max_tokens: Maksimum token sayısı
        top_p: Nucleus sampling parametresi
        top_k: Top-k sampling parametresi
    
    Returns:
        Model yanıtı (string)
    
    Raises:
        Hata durumunda "[Hata] ..." formatında string döner
    """
    logger.info(f"Starting non-streaming generation with model: {model_id}")
    
    # API key kontrolü
    api_key = _get_api_key()
    if not api_key:
        error_msg = '[Hata] GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.'
        logger.error(error_msg)
        return error_msg
    
    # Model resolve
    actual_model = _resolve_model(model_id)
    
    # Request payload oluştur
    payload = {
        'contents': _build_contents(messages),
        'generationConfig': {
            'temperature': max(0.0, min(1.0, temperature)),
            'maxOutputTokens': max_tokens,
            'topP': max(0.0, min(1.0, top_p)),
            'topK': max(1, top_k),
        },
        'safetySettings': [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
        ],
    }
    
    # Fallback modelleri dene
    models_to_try = [actual_model] + [m for m in FALLBACK_MODELS if m != actual_model]
    
    for try_model in models_to_try:
        # Endpoint URL oluştur
        url = f"{API_BASE_URL}/{try_model}:generateContent?key={api_key}"
        
        logger.debug(f"Trying model: {try_model}")
        
        # Retry döngüsü
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Timeout ayarları
                timeout = httpx.Timeout(
                    TIMEOUT_TOTAL,
                    connect=TIMEOUT_CONNECT,
                    read=TIMEOUT_READ
                )
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
                    
                    # Status code kontrolü
                    status = response.status_code
                    
                    if status == 200:
                        # Başarılı
                        result = _extract_text_from_response(response.json())
                        logger.info(f"Generation successful with model: {try_model}")
                        return result
                    
                    elif status == 404:
                        # Model bulunamadı - sonraki modele geç
                        logger.warning(f"Model not found: {try_model}")
                        break
                    
                    elif status == 429:
                        # Rate limit - retry with backoff
                        if attempt < MAX_RETRIES:
                            delay = RETRY_DELAY_BASE * (2 ** attempt)
                            logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                            await asyncio.sleep(delay)
                            continue
                        return '[Hata] API rate limit aşıldı. Lütfen biraz bekleyin.'
                    
                    elif status in (401, 403):
                        # Authentication error
                        return '[Hata] API anahtarı geçersiz veya yetkisiz.'
                    
                    elif status >= 500:
                        # Server error - retry
                        if attempt < MAX_RETRIES:
                            delay = RETRY_DELAY_BASE * (2 ** attempt)
                            logger.warning(f"Server error {status}, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        return f'[Hata] Sunucu hatası: {status}'
                    
                    else:
                        # Diğer hatalar
                        logger.error(f"Unexpected status {status}: {response.text[:500]}")
                        break
                        
            except httpx.TimeoutException as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Timeout, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                return '[Hata] İstek zaman aşımına uğradı.'
                
            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                break
                
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                break
    
    return '[Hata] Tüm modeller başarısız oldu. Lütfen daha sonra tekrar deneyin.'


# =============================================================================
# STREAMING GENERATION - GERÇEK ANLIK STREAMING
# =============================================================================

async def stream(
    messages: List[Dict[str, Any]], 
    model_id: str = 'gemini-flash',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    top_k: int = 40,
) -> AsyncGenerator[str, None]:
    """
    Gemini API ile GERÇEK ANLIK streaming yanıt üret.
    
    Bu fonksiyon Gemini'nin streamGenerateContent endpoint'ini kullanır
    ve her token/chunk anında yield edilir. Buffering YOKTUR.
    
    SSE (Server-Sent Events) formatında veri alınır ve parse edilir.
    
    Args:
        messages: OpenAI formatında mesaj listesi
        model_id: Kullanılacak model ID
        temperature: Yaratıcılık seviyesi (0.0-1.0)
        max_tokens: Maksimum token sayısı
        top_p: Nucleus sampling parametresi
        top_k: Top-k sampling parametresi
    
    Yields:
        Her chunk/token anında yield edilir
    
    Example:
        async for chunk in stream(messages, "gemini-flash"):
            print(chunk, end="", flush=True)
    """
    logger.info(f"Starting STREAMING generation with model: {model_id}")
    
    # API key kontrolü
    api_key = _get_api_key()
    if not api_key:
        error_msg = '[Hata] GEMINI_API_KEY bulunamadı.'
        logger.error(error_msg)
        yield error_msg
        return
    
    # Model resolve
    actual_model = _resolve_model(model_id)
    
    # Request payload oluştur
    payload = {
        'contents': _build_contents(messages),
        'generationConfig': {
            'temperature': max(0.0, min(1.0, temperature)),
            'maxOutputTokens': max_tokens,
            'topP': max(0.0, min(1.0, top_p)),
            'topK': max(1, top_k),
        },
        'safetySettings': [
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
        ],
    }
    
    # Fallback modelleri dene
    models_to_try = [actual_model] + [m for m in FALLBACK_MODELS if m != actual_model]
    
    for try_model in models_to_try:
        # STREAMING endpoint URL - alt=sse parametresi kritik!
        url = f"{API_BASE_URL}/{try_model}:streamGenerateContent?key={api_key}&alt=sse"
        
        logger.debug(f"Trying streaming with model: {try_model}")
        
        try:
            # Timeout ayarları - streaming için daha uzun
            timeout = httpx.Timeout(
                TIMEOUT_TOTAL,
                connect=TIMEOUT_CONNECT,
                read=None  # Streaming için read timeout yok
            )
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                # STREAMING REQUEST - client.stream() kullan
                async with client.stream('POST', url, json=payload) as response:
                    
                    # Status code kontrolü
                    status = response.status_code
                    
                    if status == 404:
                        logger.warning(f"Model not found for streaming: {try_model}")
                        continue  # Sonraki modele geç
                    
                    if status == 429:
                        logger.warning(f"Rate limited during streaming: {try_model}")
                        yield '[Hata] Rate limit aşıldı.'
                        return
                    
                    if status in (401, 403):
                        yield '[Hata] API anahtarı geçersiz.'
                        return
                    
                    if status != 200:
                        logger.warning(f"Streaming failed with status {status}: {try_model}")
                        continue  # Sonraki modele geç
                    
                    # === GERÇEK STREAMING BAŞLIYOR ===
                    logger.info(f"Streaming started successfully with model: {try_model}")
                    
                    # Buffer - satırları biriktirmek için
                    buffer = ''
                    total_chars = 0
                    chunk_count = 0
                    
                    # Byte-level streaming - EN HIZLI yöntem
                    async for raw_bytes in response.aiter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                        # Boş chunk kontrolü
                        if not raw_bytes:
                            continue
                        
                        # Bytes -> String decode
                        try:
                            text_chunk = raw_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            # Partial UTF-8 sequence, buffer'a ekle ve devam et
                            continue
                        
                        # Buffer'a ekle
                        buffer += text_chunk
                        
                        # Satır satır işle - her \n gördüğümüzde
                        while '\n' in buffer:
                            # İlk satırı al
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            # Boş satır kontrolü
                            if not line:
                                continue
                            
                            # SSE data satırı kontrolü
                            if not line.startswith('data:'):
                                continue
                            
                            # JSON kısmını al
                            json_str = line[5:].strip()
                            
                            # Boş veya [DONE] kontrolü
                            if not json_str or json_str == '[DONE]':
                                continue
                            
                            # JSON parse et
                            try:
                                data = json.loads(json_str)
                                text = _extract_text_from_response(data)
                                
                                # Geçerli text varsa ANLIK yield et
                                if text and not text.startswith('['):
                                    chunk_count += 1
                                    total_chars += len(text)
                                    
                                    # === ANLIK YIELD - BUFFERING YOK ===
                                    yield text
                                    
                            except json.JSONDecodeError:
                                # JSON parse hatası - atla
                                pass
                            except Exception as e:
                                logger.debug(f"Chunk parse error: {e}")
                    
                    # Buffer'da kalan veri varsa işle
                    if buffer.strip():
                        remaining = buffer.strip()
                        if remaining.startswith('data:'):
                            json_str = remaining[5:].strip()
                            if json_str and json_str != '[DONE]':
                                try:
                                    data = json.loads(json_str)
                                    text = _extract_text_from_response(data)
                                    if text and not text.startswith('['):
                                        yield text
                                        total_chars += len(text)
                                        chunk_count += 1
                                except Exception:
                                    pass
                    
                    # Başarılı streaming tamamlandı
                    logger.info(
                        f"Streaming completed: model={try_model}, "
                        f"chunks={chunk_count}, total_chars={total_chars}"
                    )
                    return  # Başarılı, fonksiyondan çık
                    
        except httpx.TimeoutException as e:
            logger.warning(f"Streaming timeout for {try_model}: {e}")
            continue  # Sonraki modele geç
            
        except httpx.RequestError as e:
            logger.error(f"Streaming request error for {try_model}: {e}")
            continue  # Sonraki modele geç
            
        except Exception as e:
            logger.exception(f"Unexpected streaming error for {try_model}: {e}")
            continue  # Sonraki modele geç
    
    # Hiçbir model çalışmadı
    logger.error("All models failed for streaming")
    yield '[Hata] Streaming başarısız oldu. Lütfen tekrar deneyin.'


# =============================================================================
# HEALTH CHECK
# =============================================================================

async def health_check() -> Dict[str, Any]:
    """
    Gemini API sağlık kontrolü yap.
    
    Basit bir test isteği göndererek API'nin çalışıp çalışmadığını kontrol eder.
    
    Returns:
        {
            "ok": bool,
            "model": str,
            "error": str (optional),
            "latency_ms": int (optional)
        }
    """
    import time
    
    api_key = _get_api_key()
    if not api_key:
        return {
            'ok': False,
            'error': 'API key bulunamadı',
            'model': None
        }
    
    start_time = time.time()
    
    try:
        # Basit test mesajı
        test_messages = [{'role': 'user', 'content': 'Merhaba'}]
        result = await generate(test_messages, max_tokens=10)
        
        latency = int((time.time() - start_time) * 1000)
        
        is_ok = not result.startswith('[Hata]')
        
        return {
            'ok': is_ok,
            'model': 'Gemini 2.5 Flash',
            'latency_ms': latency,
            'error': result if not is_ok else None
        }
        
    except Exception as e:
        return {
            'ok': False,
            'model': 'Gemini 2.5 Flash',
            'error': str(e)
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'generate',
    'stream',
    'health_check',
    'MODEL_MAP',
    'DEFAULT_MODEL',
]