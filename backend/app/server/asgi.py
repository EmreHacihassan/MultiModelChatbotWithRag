"""
ASGI Configuration - WebSocket + HTTP Streaming Desteği.

Uvicorn ile çalıştırın:
    uvicorn backend.app.server.asgi:application --host 0.0.0.0 --port 8000 --reload

Daphne ile çalıştırın:
    daphne -b 0.0.0.0 -p 8000 backend.app.server.asgi:application
"""

import os
import sys

# =============================================================================
# PATH SETUP
# =============================================================================

# Project root: MyChatbot/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# App root: MyChatbot/backend/app
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# =============================================================================
# DJANGO SETUP
# =============================================================================

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

# Django'yu başlat
from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()

# =============================================================================
# CHANNELS SETUP
# =============================================================================

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from backend.websockets.routing import websocket_urlpatterns

# =============================================================================
# APPLICATION
# =============================================================================

application = ProtocolTypeRouter({
    # HTTP - Django ASGI (streaming destekli)
    "http": django_asgi_app,
    
    # WebSocket - Channels
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})