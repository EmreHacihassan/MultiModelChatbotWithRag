#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # Ensure project root (MyChatbot) on PYTHONPATH
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Ensure app dir (backend/app) on PYTHONPATH so 'server' package resolves
    APP_DIR = os.path.abspath(os.path.dirname(__file__))
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Port belirtilmemişse otomatik 0.0.0.0:8001 ekle
    if len(sys.argv) >= 2 and sys.argv[1] == 'runserver':
        has_addr = any(':' in arg or arg.isdigit() for arg in sys.argv[2:])
        if not has_addr:
            sys.argv.append('0.0.0.0:8001')

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()