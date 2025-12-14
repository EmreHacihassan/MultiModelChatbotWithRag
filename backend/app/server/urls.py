"""
URL Configuration - MyChatbot Backend.

Tüm API endpoint'lerinin merkezi yönlendirmesi.
Debug modunda otomatik API dokümantasyonu sağlar.

Endpoints:
- /admin/          - Django admin paneli
- /health/         - Basit health check (api.views)
- /health/detailed/ - Detaylı sistem durumu
- /models          - Mevcut model listesi
- /sessions/*      - Oturum yönetimi (CRUD)
- /chat            - Non-streaming chat
- /chat/stream     - SSE streaming chat
- /ws/chat         - WebSocket streaming (asgi.py'de)

Bonus:
- /debug/routes/   - Tüm URL'lerin listesi (DEBUG modunda)
- /debug/config/   - Sistem konfigürasyonu (DEBUG modunda)
"""

from django.contrib import admin
from django.urls import path
from django.http import JsonResponse
from django.conf import settings as django_settings
import time

from api.views import health
from backend.core.routers import (
    models_router, 
    sessions_router, 
    chat_router, 
    sse_router, 
    health_router,
    rag_router,
)


# =============================================================================
# BONUS: DEBUG ENDPOINTS
# =============================================================================

def debug_routes(request):
    """
    [BONUS] Tüm kayıtlı URL pattern'lerini listele.
    Sadece DEBUG=True modunda çalışır.
    """
    if not django_settings.DEBUG:
        return JsonResponse({'error': 'Only available in DEBUG mode'}, status=403)
    
    from django.urls import get_resolver
    
    def extract_patterns(resolver, prefix=''):
        """URL pattern'lerini recursive olarak çıkar."""
        patterns = []
        for pattern in resolver.url_patterns:
            full_path = prefix + str(pattern.pattern)
            
            if hasattr(pattern, 'url_patterns'):
                # Nested URLconf
                patterns.extend(extract_patterns(pattern, full_path))
            else:
                # Endpoint
                view_name = getattr(pattern.callback, '__name__', str(pattern.callback))
                patterns.append({
                    'path': '/' + full_path.replace('^', '').replace('$', ''),
                    'name': pattern.name,
                    'view': view_name,
                })
        return patterns
    
    all_patterns = extract_patterns(get_resolver())
    
    # WebSocket endpoint'lerini manuel ekle
    websocket_endpoints = [
        {'path': '/ws/chat', 'name': 'websocket_chat', 'view': 'ChatConsumer', 'type': 'websocket'},
    ]
    
    return JsonResponse({
        'timestamp': int(time.time() * 1000),
        'total_http_routes': len(all_patterns),
        'total_ws_routes': len(websocket_endpoints),
        'http_routes': sorted(all_patterns, key=lambda x: x['path']),
        'websocket_routes': websocket_endpoints,
    })


def debug_config(request):
    """
    [BONUS] Sistem konfigürasyonunu göster.
    Hassas bilgiler maskelenir. Sadece DEBUG modunda çalışır.
    """
    if not django_settings.DEBUG:
        return JsonResponse({'error': 'Only available in DEBUG mode'}, status=403)
    
    import sys
    import platform
    import os
    
    def mask_key(key):
        """API key'leri maskele."""
        if not key:
            return None
        if len(key) <= 8:
            return '*' * len(key)
        return key[:4] + '*' * (len(key) - 8) + key[-4:]
    
    return JsonResponse({
        'timestamp': int(time.time() * 1000),
        'environment': {
            'debug': django_settings.DEBUG,
            'python_version': sys.version,
            'platform': platform.platform(),
            'django_version': django_settings.VERSION if hasattr(django_settings, 'VERSION') else 'unknown',
        },
        'server': {
            'allowed_hosts': django_settings.ALLOWED_HOSTS,
            'cors_origins': getattr(django_settings, 'CORS_ALLOWED_ORIGINS', []),
            'csrf_trusted': getattr(django_settings, 'CSRF_TRUSTED_ORIGINS', []),
            'timezone': django_settings.TIME_ZONE,
            'language': django_settings.LANGUAGE_CODE,
        },
        'api_keys': {
            'gemini': mask_key(getattr(django_settings, 'GEMINI_API_KEY', None)),
            'huggingface': mask_key(getattr(django_settings, 'HF_API_KEY', None)),
            'openrouter': mask_key(getattr(django_settings, 'OPENROUTER_API_KEY', None)),
            'together': mask_key(getattr(django_settings, 'TOGETHER_API_KEY', None)),
        },
        'features': {
            'websocket_enabled': True,
            'sse_streaming_enabled': True,
            'session_persistence': True,
            'rate_limiting': True,
            'auto_title': True,
        },
        'paths': {
            'base_dir': str(django_settings.BASE_DIR),
            'static_root': getattr(django_settings, 'STATIC_ROOT', None),
            'database': django_settings.DATABASES.get('default', {}).get('NAME', 'unknown'),
        },
    })


def api_index(request):
    """
    [BONUS] API ana sayfası - Hoş geldin mesajı ve dokümantasyon.
    """
    return JsonResponse({
        'name': 'MyChatbot API',
        'version': '2.0.0',
        'status': 'operational',
        'timestamp': int(time.time() * 1000),
        'description': 'AI-powered chatbot with multi-model support and real-time streaming',
        'documentation': {
            'endpoints': '/debug/routes/' if django_settings.DEBUG else 'Disabled in production',
            'config': '/debug/config/' if django_settings.DEBUG else 'Disabled in production',
        },
        'features': [
            'Multi-model support (Gemini, Gemma, Qwen, Llama)',
            'Real-time streaming via WebSocket and SSE',
            'Session persistence',
            'Markdown rendering',
            'Rate limiting',
        ],
        'links': {
            'health': '/health/',
            'health_detailed': '/health/detailed/',
            'models': '/models',
            'chat': '/chat',
            'chat_stream': '/chat/stream',
            'websocket': 'ws://localhost:8000/ws/chat',
        },
    })


# =============================================================================
# URL PATTERNS
# =============================================================================

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Index (Bonus)
    path('', api_index, name='api_index'),
    path('api/', api_index, name='api_index_alt'),
    
    # Health checks - health_router() tarafından sağlanıyor
]

# Router'lardan gelen URL'leri ekle
urlpatterns += models_router()      # /models
urlpatterns += sessions_router()    # /sessions/*
urlpatterns += chat_router()        # /chat
urlpatterns += sse_router()         # /chat/stream
urlpatterns += health_router()      # /health/, /health/detailed/
urlpatterns += rag_router()         # /rag/*

# Debug endpoints (sadece DEBUG modunda)
if django_settings.DEBUG:
    urlpatterns += [
        path('debug/routes/', debug_routes, name='debug_routes'),
        path('debug/config/', debug_config, name='debug_config'),
    ]


# =============================================================================
# BONUS: URL Pattern Validasyonu (başlangıçta çalışır)
# =============================================================================

def _validate_url_patterns():
    """URL pattern'lerinin benzersiz olduğunu doğrula."""
    seen_paths = {}
    duplicates = []
    
    for pattern in urlpatterns:
        path_str = str(pattern.pattern)
        if path_str in seen_paths:
            duplicates.append(path_str)
        seen_paths[path_str] = True
    
    if duplicates and django_settings.DEBUG:
        import logging
        logger = logging.getLogger('django.urls')
        logger.warning(f"Duplicate URL patterns detected: {duplicates}")

# Validasyonu çalıştır
try:
    _validate_url_patterns()
except Exception:
    pass  # Başlangıç hatalarını sessizce geç