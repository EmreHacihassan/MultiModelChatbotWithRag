"""
Django Settings - MyChatbot Backend.

Bu dosya Django uygulamasının tüm ayarlarını içerir.
Production'a geçerken DEBUG=False yapın ve SECRET_KEY'i değiştirin.

Bonus Özellikler:
- Otomatik .env yükleme
- Çoklu CORS origin desteği (3002 dahil)
- Detaylı logging konfigürasyonu
- Performance optimizasyonları
- Güvenlik başlıkları
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root: MyChatbot/
BASE_DIR = Path(__file__).resolve().parents[3]

# Backend app directory
APP_DIR = Path(__file__).resolve().parent

# .env dosyasını yükle (çoklu kaynak desteği)
ENV_PATHS = [
    os.path.join(str(BASE_DIR), 'configs', 'env', '.env'),
    os.path.join(str(BASE_DIR), '.env'),
    os.path.join(str(APP_DIR), '.env'),
]

for env_path in ENV_PATHS:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# Secret key - Production'da mutlaka değiştirin!
SECRET_KEY = os.getenv(
    'SECRET_KEY', 
    os.getenv('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production-immediately')
)

# Debug mode - Production'da False yapın
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes', 'on')

# Allowed hosts - Production'da spesifik domain'ler belirtin
ALLOWED_HOSTS = [
    host.strip() 
    for host in os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,0.0.0.0,*').split(',')
    if host.strip()
]

# =============================================================================
# APPLICATION DEFINITION
# =============================================================================

INSTALLED_APPS = [
    # Django core apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'corsheaders',      # Cross-Origin Resource Sharing
    'channels',         # WebSocket & async support
    
    # Project apps
    'api',
]

MIDDLEWARE = [
    # CORS - EN ÜSTTE olmalı!
    'corsheaders.middleware.CorsMiddleware',
    
    # Security
    'django.middleware.security.SecurityMiddleware',
    
    # Session & Auth
    'django.contrib.sessions.middleware.SessionMiddleware',
    
    # Common
    'django.middleware.common.CommonMiddleware',
    
    # CSRF - API için csrf_exempt kullanıyoruz
    'django.middleware.csrf.CsrfViewMiddleware',
    
    # Auth
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    
    # Messages
    'django.contrib.messages.middleware.MessageMiddleware',
    
    # Clickjacking protection
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'server.urls'

# =============================================================================
# CORS CONFIGURATION - PORT 3002 DAHİL
# =============================================================================

# İzin verilen origin'ler - Frontend URL'leri
CORS_ALLOWED_ORIGINS = [
    # Port 3002 - Ana frontend portu
    'http://localhost:3002',
    'http://127.0.0.1:3002',
    
    # Vite varsayılan portu (yedek)
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    
    # Alternatif portlar
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3001',
    
    # Production (örnek)
    # 'https://mychatbot.example.com',
]

# .env'den ek origin'ler
_env_origins = os.getenv('CORS_ALLOWED_ORIGINS', os.getenv('ALLOWED_ORIGINS', ''))
if _env_origins:
    _extra = [o.strip() for o in _env_origins.split(',') if o.strip()]
    CORS_ALLOWED_ORIGINS.extend(_extra)

# CORS ek ayarları
CORS_ALLOW_CREDENTIALS = True  # Cookie'leri kabul et

# DEBUG modunda tüm origin'lere izin ver (geliştirme kolaylığı)
CORS_ALLOW_ALL_ORIGINS = DEBUG

# İzin verilen HTTP metodları
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# İzin verilen header'lar
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'accept-language',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'cache-control',
    'pragma',
]

# Preflight cache süresi (saniye)
CORS_PREFLIGHT_MAX_AGE = 86400  # 24 saat

# Expose edilecek header'lar (client'ın okuyabileceği)
CORS_EXPOSE_HEADERS = [
    'content-length',
    'content-type',
    'x-request-id',
]

# =============================================================================
# CSRF CONFIGURATION
# =============================================================================

CSRF_TRUSTED_ORIGINS = [
    # Port 3002 - Ana frontend portu
    'http://localhost:3002',
    'http://127.0.0.1:3002',
    
    # Alternatif portlar
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3001',
]

# .env'den ek trusted origin'ler
_csrf_env = os.getenv('CSRF_TRUSTED_ORIGINS', '')
if _csrf_env:
    _csrf_extra = [o.strip() for o in _csrf_env.split(',') if o.strip()]
    CSRF_TRUSTED_ORIGINS.extend(_csrf_extra)

# =============================================================================
# TEMPLATES
# =============================================================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(str(BASE_DIR), 'templates'),
            os.path.join(str(BASE_DIR), 'frontend', 'dist'),  # Production build
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# =============================================================================
# ASGI / WSGI APPLICATION
# =============================================================================

WSGI_APPLICATION = 'server.wsgi.application'
ASGI_APPLICATION = 'server.asgi.application'

# =============================================================================
# CHANNELS (WebSocket Support)
# =============================================================================

# Geliştirme için InMemoryChannelLayer
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
        # InMemory için timeout ayarı
        'CONFIG': {
            'capacity': 1000,  # Maximum message buffer
            'expiry': 60,      # Message expiry in seconds
        },
    }
}

# Production için Redis (yorum satırını kaldırın)
# CHANNEL_LAYERS = {
#     'default': {
#         'BACKEND': 'channels_redis.core.RedisChannelLayer',
#         'CONFIG': {
#             'hosts': [(os.getenv('REDIS_HOST', 'localhost'), int(os.getenv('REDIS_PORT', 6379)))],
#             'capacity': 1500,
#             'expiry': 60,
#         },
#     },
# }

# =============================================================================
# DATABASE
# =============================================================================

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(str(BASE_DIR), 'backend', 'app', 'db.sqlite3'),
        # SQLite performans optimizasyonları
        'OPTIONS': {
            'timeout': 20,  # Lock timeout
        },
    }
}

# Production için PostgreSQL örneği:
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': os.getenv('DB_NAME', 'mychatbot'),
#         'USER': os.getenv('DB_USER', 'postgres'),
#         'PASSWORD': os.getenv('DB_PASSWORD', ''),
#         'HOST': os.getenv('DB_HOST', 'localhost'),
#         'PORT': os.getenv('DB_PORT', '5432'),
#         'CONN_MAX_AGE': 60,  # Connection pooling
#         'OPTIONS': {
#             'connect_timeout': 10,
#         },
#     }
# }

# =============================================================================
# DEFAULT AUTO FIELD
# =============================================================================

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# =============================================================================
# PASSWORD VALIDATION
# =============================================================================

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# =============================================================================
# INTERNATIONALIZATION
# =============================================================================

LANGUAGE_CODE = 'tr-tr'
TIME_ZONE = 'Europe/Istanbul'
USE_I18N = True
USE_TZ = True

# Desteklenen diller (bonus)
LANGUAGES = [
    ('tr', 'Türkçe'),
    ('en', 'English'),
]

# =============================================================================
# STATIC FILES
# =============================================================================

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(str(BASE_DIR), 'staticfiles')

STATICFILES_DIRS = [
    # Frontend production build (varsa)
    # os.path.join(str(BASE_DIR), 'frontend', 'dist', 'assets'),
]

# Static file finders
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

# =============================================================================
# MEDIA FILES (User uploads)
# =============================================================================

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(str(BASE_DIR), 'media')

# =============================================================================
# API KEYS (Provider keys from .env)
# =============================================================================

# Google Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# HuggingFace
HF_API_KEY = os.getenv('HF_API_KEY', os.getenv('HUGGINGFACE_API_KEY', ''))

# OpenRouter (gelecek için)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

# Together AI (gelecek için)
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')

# Ollama host
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# =============================================================================
# CACHING (Bonus - Performance)
# =============================================================================

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'mychatbot-cache',
        'TIMEOUT': 300,  # 5 dakika
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        },
    }
}

# Production için Redis cache:
# CACHES = {
#     'default': {
#         'BACKEND': 'django.core.cache.backends.redis.RedisCache',
#         'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
#     }
# }

# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400 * 7  # 7 gün
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log dizinini oluştur
LOG_DIR = os.path.join(str(BASE_DIR), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} {module}:{lineno} - {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '[{levelname}] {asctime} - {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
        },
        'colored': {
            'format': '{levelname} {asctime} {name} - {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
        },
    },
    
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
    },
    
    'handlers': {
        'console': {
            'level': 'DEBUG' if DEBUG else 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': sys.stdout,
        },
        'file_django': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'django.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',
        },
        'file_api': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'api.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',
        },
        'file_websocket': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'websocket.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 3,
            'formatter': 'verbose',
            'encoding': 'utf-8',
        },
        'file_errors': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_DIR, 'errors.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 10,
            'formatter': 'verbose',
            'encoding': 'utf-8',
        },
    },
    
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    
    'loggers': {
        # Django core
        'django': {
            'handlers': ['console', 'file_django'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'file_errors'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.server': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console'] if DEBUG else [],
            'level': 'DEBUG' if DEBUG and os.getenv('SQL_DEBUG') else 'WARNING',
            'propagate': False,
        },
        
        # Project loggers
        'api': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'core.routers': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'adapters': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'adapters.gemini': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'adapters.huggingface': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'adapters.ollama': {
            'handlers': ['console', 'file_api'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'websockets': {
            'handlers': ['console', 'file_websocket'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'websockets.consumers': {
            'handlers': ['console', 'file_websocket'],
            'level': 'DEBUG',
            'propagate': False,
        },
        
        # Third party
        'channels': {
            'handlers': ['console', 'file_websocket'],
            'level': 'INFO',
            'propagate': False,
        },
        'httpx': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'httpcore': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}

# =============================================================================
# SECURITY SETTINGS (Production)
# =============================================================================

if not DEBUG:
    # XSS Protection
    SECURE_BROWSER_XSS_FILTER = True
    
    # Content type sniffing protection
    SECURE_CONTENT_TYPE_NOSNIFF = True
    
    # Clickjacking protection
    X_FRAME_OPTIONS = 'DENY'
    
    # HSTS (HTTP Strict Transport Security)
    # SECURE_HSTS_SECONDS = 31536000  # 1 yıl
    # SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    # SECURE_HSTS_PRELOAD = True
    
    # HTTPS redirect
    # SECURE_SSL_REDIRECT = True
    
    # Secure cookies
    # SESSION_COOKIE_SECURE = True
    # CSRF_COOKIE_SECURE = True

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Data upload limits
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
DATA_UPLOAD_MAX_NUMBER_FIELDS = 1000
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # 5 MB

# =============================================================================
# BONUS: FEATURE FLAGS
# =============================================================================

FEATURES = {
    # Aktif özellikler
    'WEBSOCKET_STREAMING': True,
    'SSE_STREAMING': True,
    'SESSION_PERSISTENCE': True,
    'RATE_LIMITING': True,
    'AUTO_TITLE': True,
    'MARKDOWN_RENDERING': True,
    'CODE_HIGHLIGHTING': True,
    
    # Deneysel özellikler
    'WEB_SEARCH': False,  # Gelecekte eklenecek
    'FILE_UPLOAD': False,  # Gelecekte eklenecek
    'VOICE_INPUT': False,  # Gelecekte eklenecek
    'IMAGE_GENERATION': False,  # Gelecekte eklenecek
}

# =============================================================================
# BONUS: APPLICATION METADATA
# =============================================================================

APP_NAME = 'MyChatbot'
APP_VERSION = '2.0.0'
APP_DESCRIPTION = 'AI-powered chatbot with multi-model support'
APP_AUTHOR = 'MyChatbot Team'

# Django version for debugging
try:
    import django
    VERSION = django.VERSION
except Exception:
    VERSION = 'unknown'

# =============================================================================
# STARTUP VALIDATION
# =============================================================================

def _validate_settings():
    """Başlangıçta ayarları doğrula."""
    warnings = []
    
    # API key kontrolü
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY is not set. Gemini adapter will not work.")
    
    if not HF_API_KEY:
        warnings.append("HF_API_KEY is not set. HuggingFace adapter will not work.")
    
    # Secret key kontrolü
    if SECRET_KEY == 'dev-secret-key-change-in-production-immediately':
        warnings.append("Using default SECRET_KEY. Change this in production!")
    
    # Log warnings
    if warnings and DEBUG:
        import logging
        logger = logging.getLogger('django.settings')
        for warn in warnings:
            logger.warning(f"[Settings] {warn}")

# Validasyonu çalıştır (import sırasında)
try:
    _validate_settings()
except Exception:
    pass