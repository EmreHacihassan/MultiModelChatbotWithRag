from django.contrib import admin
from django.urls import path
from api.views import health
from backend.core.routers import models_router, sessions_router, chat_router, sse_router

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health),
]
urlpatterns += models_router()
urlpatterns += sessions_router()
urlpatterns += chat_router()
urlpatterns += sse_router()