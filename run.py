#!/usr/bin/env python
"""
MyChatbot - Ultimate Launcher v3.0
==================================

Tek komutla tüm sistemi başlatır:
- Backend (Django + Uvicorn + WebSocket)
- Frontend (Vite + React)

Özellikler:
- Otomatik Python sürüm kontrolü (3.11+ gerekli)
- Otomatik port kontrolü ve temizleme
- Renkli terminal çıktısı
- Graceful shutdown (Ctrl+C)
- Health check
- Detaylı loglama

Kullanım:
    python run.py              # Her ikisini başlat
    python run.py --backend    # Sadece backend
    python run.py --frontend   # Sadece frontend
    python run.py --check      # Sistem kontrolü
"""

import os
import sys
import subprocess
import signal
import time
import socket
import threading
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Uygulama konfigürasyonu."""
    # Proje dizinleri
    BASE_DIR: Path = Path(__file__).resolve().parent
    BACKEND_DIR: Path = BASE_DIR / "backend" / "app"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    VENV_DIR: Path = BASE_DIR / ".venv"
    
    # Portlar
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 3002
    
    # Backend ayarları
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_WORKERS: int = 1
    BACKEND_RELOAD: bool = True
    
    # Timeout ayarları
    STARTUP_TIMEOUT: int = 30
    SHUTDOWN_TIMEOUT: int = 10
    HEALTH_CHECK_INTERVAL: float = 0.5
    
    # Python sürüm gereksinimleri
    MIN_PYTHON_VERSION: Tuple[int, int] = (3, 11)


CONFIG = Config()


# =============================================================================
# COLORS & STYLING
# =============================================================================

class Colors:
    """ANSI renk kodları."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Renkler
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Arka plan
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


class Icons:
    """Unicode ikonları."""
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    ROCKET = "🚀"
    GLOBE = "🌐"
    SERVER = "⚡"
    WARNING = "⚠"
    INFO = "ℹ"
    STOP = "⏹"
    CLOCK = "⏱"
    PYTHON = "🐍"
    PACKAGE = "📦"


def print_banner():
    """Başlangıç banner'ı yazdır."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Colors.WHITE}███╗   ███╗██╗   ██╗ ██████╗██╗  ██╗ █████╗ ████████╗{Colors.CYAN}        ║
║   {Colors.WHITE}████╗ ████║╚██╗ ██╔╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝{Colors.CYAN}        ║
║   {Colors.WHITE}██╔████╔██║ ╚████╔╝ ██║     ███████║███████║   ██║{Colors.CYAN}           ║
║   {Colors.WHITE}██║╚██╔╝██║  ╚██╔╝  ██║     ██╔══██║██╔══██║   ██║{Colors.CYAN}           ║
║   {Colors.WHITE}██║ ╚═╝ ██║   ██║   ╚██████╗██║  ██║██║  ██║   ██║{Colors.CYAN}           ║
║   {Colors.WHITE}╚═╝     ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝{Colors.CYAN}           ║
║                                                                  ║
║   {Colors.YELLOW}Ultimate Launcher v3.0{Colors.CYAN}                                      ║
║   {Colors.DIM}Django + React + WebSocket Streaming{Colors.CYAN}                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def log(message: str, level: str = "info", prefix: str = ""):
    """Renkli log mesajı yazdır."""
    timestamp = time.strftime("%H:%M:%S")
    
    level_config = {
        "info": (Colors.BLUE, Icons.INFO),
        "success": (Colors.GREEN, Icons.CHECK),
        "warning": (Colors.YELLOW, Icons.WARNING),
        "error": (Colors.RED, Icons.CROSS),
        "start": (Colors.MAGENTA, Icons.ROCKET),
        "stop": (Colors.RED, Icons.STOP),
    }
    
    color, icon = level_config.get(level, (Colors.WHITE, ""))
    prefix_str = f"[{prefix}] " if prefix else ""
    
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {color}{icon} {prefix_str}{message}{Colors.RESET}")


# =============================================================================
# SYSTEM CHECKS
# =============================================================================

class SystemChecker:
    """Sistem gereksinimlerini kontrol eder."""
    
    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """Python sürümünü kontrol et."""
        current = sys.version_info[:2]
        required = CONFIG.MIN_PYTHON_VERSION
        
        if current >= required:
            return True, f"Python {current[0]}.{current[1]} {Icons.CHECK}"
        else:
            return False, f"Python {required[0]}.{required[1]}+ gerekli, mevcut: {current[0]}.{current[1]}"
    
    @staticmethod
    def check_port(port: int) -> Tuple[bool, str]:
        """Port kullanılabilirliğini kontrol et."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return False, f"Port {port} kullanımda"
                return True, f"Port {port} kullanılabilir"
        except Exception as e:
            return False, f"Port kontrolü başarısız: {e}"
    
    @staticmethod
    def check_venv() -> Tuple[bool, str]:
        """Virtual environment kontrolü."""
        venv_python = CONFIG.VENV_DIR / "Scripts" / "python.exe"
        if not CONFIG.VENV_DIR.exists():
            return False, "Virtual environment bulunamadı (.venv)"
        if not venv_python.exists():
            return False, "venv python.exe bulunamadı"
        return True, f"venv aktif: {CONFIG.VENV_DIR}"
    
    @staticmethod
    def check_dependencies() -> Tuple[bool, str]:
        """Kritik bağımlılıkları kontrol et."""
        required = ['django', 'uvicorn', 'channels', 'aiohttp']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            return False, f"Eksik paketler: {', '.join(missing)}"
        return True, "Tüm bağımlılıklar mevcut"
    
    @staticmethod
    def check_node() -> Tuple[bool, str]:
        """Node.js kontrolü."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, f"Node.js {version}"
            return False, "Node.js çalıştırılamadı"
        except FileNotFoundError:
            return False, "Node.js bulunamadı"
        except Exception as e:
            return False, f"Node.js kontrolü başarısız: {e}"
    
    @staticmethod
    def check_npm_packages() -> Tuple[bool, str]:
        """npm paketlerini kontrol et."""
        node_modules = CONFIG.FRONTEND_DIR / "node_modules"
        if not node_modules.exists():
            return False, "node_modules bulunamadı (npm install gerekli)"
        return True, "npm paketleri mevcut"
    
    @classmethod
    def run_all_checks(cls) -> bool:
        """Tüm kontrolleri çalıştır."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
        print(f"  {Icons.INFO} SİSTEM KONTROLÜ")
        print(f"{'='*60}{Colors.RESET}\n")
        
        checks = [
            ("Python Sürümü", cls.check_python_version),
            ("Virtual Environment", cls.check_venv),
            ("Python Bağımlılıkları", cls.check_dependencies),
            ("Node.js", cls.check_node),
            ("npm Paketleri", cls.check_npm_packages),
            (f"Backend Port ({CONFIG.BACKEND_PORT})", lambda: cls.check_port(CONFIG.BACKEND_PORT)),
            (f"Frontend Port ({CONFIG.FRONTEND_PORT})", lambda: cls.check_port(CONFIG.FRONTEND_PORT)),
        ]
        
        all_passed = True
        
        for name, check_fn in checks:
            passed, message = check_fn()
            status = f"{Colors.GREEN}{Icons.CHECK}" if passed else f"{Colors.RED}{Icons.CROSS}"
            print(f"  {status} {Colors.WHITE}{name}:{Colors.RESET} {message}")
            
            if not passed:
                all_passed = False
        
        print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}\n")
        
        return all_passed


# =============================================================================
# PORT MANAGER
# =============================================================================

class PortManager:
    """Port yönetimi."""
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Port kullanımda mı kontrol et."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False
    
    @staticmethod
    def kill_process_on_port(port: int) -> bool:
        """Belirtilen portu kullanan işlemi sonlandır."""
        try:
            if sys.platform == 'win32':
                # Windows için
                result = subprocess.run(
                    f'netstat -ano | findstr :{port}',
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                for line in result.stdout.strip().split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        pid = parts[-1]
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                        log(f"Port {port} üzerindeki işlem (PID: {pid}) sonlandırıldı", "warning")
                        return True
            else:
                # Linux/Mac için
                subprocess.run(f'fuser -k {port}/tcp', shell=True, capture_output=True)
                return True
                
        except Exception as e:
            log(f"Port temizleme hatası: {e}", "error")
        
        return False
    
    @classmethod
    def ensure_port_available(cls, port: int) -> bool:
        """Portun kullanılabilir olduğundan emin ol."""
        if not cls.is_port_in_use(port):
            return True
        
        log(f"Port {port} kullanımda, temizleniyor...", "warning")
        cls.kill_process_on_port(port)
        time.sleep(1)
        
        return not cls.is_port_in_use(port)


# =============================================================================
# PROCESS MANAGER
# =============================================================================

class ProcessManager:
    """Alt süreç yöneticisi."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True
        self._lock = threading.Lock()
    
    def start_backend(self) -> bool:
        """Backend sunucusunu başlat."""
        log("Backend başlatılıyor...", "start", "BACKEND")
        
        # Port kontrolü
        if not PortManager.ensure_port_available(CONFIG.BACKEND_PORT):
            log(f"Port {CONFIG.BACKEND_PORT} kullanılamıyor!", "error", "BACKEND")
            return False
        
        # Uvicorn komutu
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.app.server.asgi:application",
            "--host", CONFIG.BACKEND_HOST,
            "--port", str(CONFIG.BACKEND_PORT),
            "--reload" if CONFIG.BACKEND_RELOAD else "",
            "--log-level", "info",
        ]
        
        # Boş stringleri temizle
        cmd = [c for c in cmd if c]
        
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["DJANGO_SETTINGS_MODULE"] = "backend.app.server.settings"
            
            process = subprocess.Popen(
                cmd,
                cwd=str(CONFIG.BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            
            with self._lock:
                self.processes['backend'] = process
            
            # Output thread başlat
            thread = threading.Thread(
                target=self._stream_output,
                args=(process, "BACKEND", Colors.GREEN),
                daemon=True
            )
            thread.start()
            
            # Başlangıç kontrolü
            if self._wait_for_port(CONFIG.BACKEND_PORT, "Backend"):
                log(f"Backend hazır: http://localhost:{CONFIG.BACKEND_PORT}", "success", "BACKEND")
                return True
            else:
                log("Backend başlatılamadı!", "error", "BACKEND")
                return False
                
        except Exception as e:
            log(f"Backend başlatma hatası: {e}", "error", "BACKEND")
            return False
    
    def start_frontend(self) -> bool:
        """Frontend geliştirme sunucusunu başlat."""
        log("Frontend başlatılıyor...", "start", "FRONTEND")
        
        # Port kontrolü
        if not PortManager.ensure_port_available(CONFIG.FRONTEND_PORT):
            log(f"Port {CONFIG.FRONTEND_PORT} kullanılamıyor!", "error", "FRONTEND")
            return False
        
        # node_modules kontrolü
        node_modules = CONFIG.FRONTEND_DIR / "node_modules"
        if not node_modules.exists():
            log("npm install çalıştırılıyor...", "info", "FRONTEND")
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            subprocess.run([npm_cmd, "install"], cwd=str(CONFIG.FRONTEND_DIR), check=True)
        
        # Vite komutu
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        cmd = [npm_cmd, "run", "dev", "--", "--port", str(CONFIG.FRONTEND_PORT)]
        
        try:
            env = os.environ.copy()
            env["BROWSER"] = "none"  # Otomatik tarayıcı açmayı engelle
            
            process = subprocess.Popen(
                cmd,
                cwd=str(CONFIG.FRONTEND_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                shell=False,
            )
            
            with self._lock:
                self.processes['frontend'] = process
            
            # Output thread başlat
            thread = threading.Thread(
                target=self._stream_output,
                args=(process, "FRONTEND", Colors.CYAN),
                daemon=True
            )
            thread.start()
            
            # Başlangıç kontrolü
            if self._wait_for_port(CONFIG.FRONTEND_PORT, "Frontend"):
                log(f"Frontend hazır: http://localhost:{CONFIG.FRONTEND_PORT}", "success", "FRONTEND")
                return True
            else:
                log("Frontend başlatılamadı!", "error", "FRONTEND")
                return False
                
        except Exception as e:
            log(f"Frontend başlatma hatası: {e}", "error", "FRONTEND")
            return False
    
    def _stream_output(self, process: subprocess.Popen, name: str, color: str):
        """Süreç çıktısını stream et."""
        try:
            for line in iter(process.stdout.readline, ''):
                if not self.running:
                    break
                if line:
                    line = line.rstrip()
                    # Gereksiz satırları filtrele
                    if any(skip in line.lower() for skip in ['watching for', 'hmr', 'vite']):
                        continue
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {color}[{name}]{Colors.RESET} {line}")
        except Exception:
            pass
    
    def _wait_for_port(self, port: int, name: str, timeout: int = None) -> bool:
        """Portun açılmasını bekle."""
        timeout = timeout or CONFIG.STARTUP_TIMEOUT
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if PortManager.is_port_in_use(port):
                return True
            time.sleep(CONFIG.HEALTH_CHECK_INTERVAL)
        
        return False
    
    def stop_all(self):
        """Tüm süreçleri durdur."""
        self.running = False
        
        print(f"\n{Colors.YELLOW}{Icons.STOP} Sunucular durduruluyor...{Colors.RESET}\n")
        
        with self._lock:
            for name, process in self.processes.items():
                if process and process.poll() is None:
                    log(f"{name} durduruluyor...", "stop", name.upper())
                    try:
                        if sys.platform == 'win32':
                            process.terminate()
                        else:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        
                        process.wait(timeout=CONFIG.SHUTDOWN_TIMEOUT)
                        log(f"{name} durduruldu", "success", name.upper())
                    except subprocess.TimeoutExpired:
                        process.kill()
                        log(f"{name} zorla durduruldu", "warning", name.upper())
                    except Exception as e:
                        log(f"{name} durdurma hatası: {e}", "error", name.upper())
            
            self.processes.clear()
    
    def is_healthy(self) -> bool:
        """Tüm süreçlerin çalışıp çalışmadığını kontrol et."""
        with self._lock:
            for name, process in self.processes.items():
                if process.poll() is not None:
                    return False
        return True


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MyChatbotLauncher:
    """Ana uygulama başlatıcı."""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Sinyal işleyicilerini ayarla."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        if sys.platform == 'win32':
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Sinyal işleyici."""
        self.shutdown()
    
    def shutdown(self):
        """Uygulamayı kapat."""
        self.process_manager.stop_all()
        
        print(f"\n{Colors.GREEN}{Icons.CHECK} MyChatbot kapatıldı.{Colors.RESET}\n")
        sys.exit(0)
    
    def run(self, backend: bool = True, frontend: bool = True):
        """Uygulamayı başlat."""
        print_banner()
        
        # Python sürüm kontrolü
        passed, msg = SystemChecker.check_python_version()
        if not passed:
            log(msg, "error")
            log(f"Python {CONFIG.MIN_PYTHON_VERSION[0]}.{CONFIG.MIN_PYTHON_VERSION[1]}+ yükleyin.", "info")
            sys.exit(1)
        
        log(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", "success")
        
        # Çalışma dizinini ayarla
        os.chdir(CONFIG.BASE_DIR)
        
        # Başlatma
        started = []
        
        if backend:
            if self.process_manager.start_backend():
                started.append("Backend")
            else:
                log("Backend başlatılamadı!", "error")
                self.shutdown()
        
        if frontend:
            if self.process_manager.start_frontend():
                started.append("Frontend")
            else:
                log("Frontend başlatılamadı!", "error")
                self.shutdown()
        
        if not started:
            log("Hiçbir servis başlatılamadı!", "error")
            sys.exit(1)
        
        # Başarı mesajı
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}")
        print(f"  {Icons.ROCKET} MyChatbot HAZIR!")
        print(f"{'='*60}{Colors.RESET}\n")
        
        if backend:
            print(f"  {Colors.GREEN}{Icons.SERVER} Backend:{Colors.RESET}  http://localhost:{CONFIG.BACKEND_PORT}")
            print(f"  {Colors.GREEN}{Icons.SERVER} WebSocket:{Colors.RESET} ws://localhost:{CONFIG.BACKEND_PORT}/ws/chat/")
            print(f"  {Colors.GREEN}{Icons.SERVER} API:{Colors.RESET}       http://localhost:{CONFIG.BACKEND_PORT}/api/")
        
        if frontend:
            print(f"  {Colors.CYAN}{Icons.GLOBE} Frontend:{Colors.RESET}  http://localhost:{CONFIG.FRONTEND_PORT}")
        
        print(f"\n  {Colors.YELLOW}{Icons.INFO} Durdurmak için:{Colors.RESET} Ctrl+C\n")
        print(f"{Colors.DIM}{'─'*60}{Colors.RESET}\n")
        
        # Ana döngü
        try:
            while True:
                if not self.process_manager.is_healthy():
                    log("Bir veya daha fazla servis çöktü!", "error")
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Ana giriş noktası."""
    parser = argparse.ArgumentParser(
        description="MyChatbot - Ultimate Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run.py              Tüm servisleri başlat
  python run.py --backend    Sadece backend
  python run.py --frontend   Sadece frontend
  python run.py --check      Sistem kontrolü
        """
    )
    
    parser.add_argument(
        '--backend', '-b',
        action='store_true',
        help='Sadece backend başlat'
    )
    
    parser.add_argument(
        '--frontend', '-f',
        action='store_true',
        help='Sadece frontend başlat'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Sistem kontrolü yap'
    )
    
    parser.add_argument(
        '--port-backend',
        type=int,
        default=8000,
        help='Backend portu (varsayılan: 8000)'
    )
    
    parser.add_argument(
        '--port-frontend',
        type=int,
        default=3002,
        help='Frontend portu (varsayılan: 3002)'
    )
    
    args = parser.parse_args()
    
    # Port konfigürasyonu
    CONFIG.BACKEND_PORT = args.port_backend
    CONFIG.FRONTEND_PORT = args.port_frontend
    
    # Sistem kontrolü
    if args.check:
        print_banner()
        all_passed = SystemChecker.run_all_checks()
        sys.exit(0 if all_passed else 1)
    
    # Başlatma modları
    if args.backend and args.frontend:
        # Her ikisi de belirtilmiş
        backend = True
        frontend = True
    elif args.backend:
        backend = True
        frontend = False
    elif args.frontend:
        backend = False
        frontend = True
    else:
        # Varsayılan: her ikisi
        backend = True
        frontend = True
    
    # Başlat
    launcher = MyChatbotLauncher()
    launcher.run(backend=backend, frontend=frontend)


if __name__ == "__main__":
    main()