# 🤖 MyChatbot - Multi-Model AI Chatbot with RAG & Agents

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Django](https://img.shields.io/badge/Django-4.2+-092E20.svg)](https://www.djangoproject.com/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Streaming-green.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Modern, çok modelli yapay zeka sohbet uygulaması. **Gemini**, **HuggingFace** ve **Ollama** modellerini destekler. Gerçek zamanlı streaming, AI Agents ve RAG (Retrieval-Augmented Generation) özellikleri içerir.

![MyChatbot Screenshot](https://via.placeholder.com/800x400?text=MyChatbot+Interface)

## ✨ Özellikler

### 🎯 Temel Özellikler
- **Çoklu Model Desteği**: Gemini, HuggingFace (Llama, Qwen, Mistral), Ollama
- **Gerçek Zamanlı Streaming**: WebSocket ile anlık yanıt görüntüleme
- **Oturum Yönetimi**: Sohbet geçmişi kaydetme ve yükleme
- **Markdown Desteği**: Zengin metin formatlaması ve kod vurgulama

### 🤖 AI Agents (Yeni!)
- **ReAct Pattern**: Düşün → Hareket Et → Gözlemle döngüsü
- **8 Yerleşik Araç**:
  - 🔢 Calculator - Matematiksel hesaplamalar
  - 🐍 Python Executor - Kod çalıştırma
  - 🔍 Web Search - DuckDuckGo arama
  - 📅 DateTime - Tarih/saat bilgisi
  - 📖 Wikipedia - Ansiklopedi araması
  - 📋 JSON Parser - JSON işleme
  - 📝 Text Analyzer - Metin analizi
  - 📏 Unit Converter - Birim dönüşümü

### 📚 RAG Pipeline (Yeni!)
- PDF, TXT, MD dosya desteği
- Akıllı metin parçalama (chunking)
- Sentence Transformers ile embedding
- FAISS vektör indeksleme
- Benzerlik tabanlı arama

### 🎨 Modern UI
- Dark theme tasarım
- Responsive layout
- Model seçici dropdown
- Agent/RAG mod göstergeleri
- Gerçek zamanlı düşünce baloncukları

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.11+
- Node.js 18+
- Git

### Kurulum

```bash
# Repoyu klonla
git clone https://github.com/EmreHacihassan/MultiModelChatbotWithRag.git
cd MultiModelChatbotWithRag

# Python sanal ortamı oluştur
python -m venv .venv

# Aktive et (Windows)
.venv\Scripts\activate

# Aktive et (macOS/Linux)
source .venv/bin/activate

# Python bağımlılıklarını yükle
pip install -r requirements.txt

# Frontend bağımlılıklarını yükle
cd frontend && npm install && cd ..
```

### API Anahtarları

`.env` dosyasını `configs/env/` altına oluşturun:

```env
# Gemini API Key (Google AI Studio'dan alın)
GEMINI_API_KEY=your_gemini_api_key

# HuggingFace API Key (huggingface.co/settings/tokens)
HF_API_KEY=your_huggingface_api_key
```

### Çalıştırma

```bash
# Tek komutla her şeyi başlat
python run.py

# Sadece backend
python run.py --backend

# Sadece frontend
python run.py --frontend
```

Tarayıcınızda açın: **http://localhost:3002**

## 📁 Proje Yapısı

```
MyChatbot/
├── backend/
│   ├── adapters/          # Model adapterleri
│   │   ├── gemini.py      # Google Gemini API
│   │   ├── huggingface.py # HuggingFace Inference
│   │   └── ollama.py      # Ollama (yerel)
│   ├── agents/            # AI Agents sistemi
│   │   └── __init__.py    # ReAct pattern, tools
│   ├── app/server/        # Django ayarları
│   ├── core/routers/      # REST API endpoint'leri
│   └── websockets/        # WebSocket consumer
├── frontend/
│   └── src/
│       ├── components/    # React bileşenleri
│       ├── pages/         # Sayfa bileşenleri
│       └── lib/           # API utilities
├── rag/
│   └── pipelines/         # RAG pipeline
├── configs/
│   ├── env/               # Environment dosyaları
│   └── models/            # Model konfigürasyonları
├── data/
│   └── sessions/          # Sohbet geçmişi
├── run.py                 # Ultimate Launcher
├── requirements.txt       # Python bağımlılıkları
├── docker-compose.yml     # Docker yapılandırması
└── README.md
```

## 🔧 Konfigürasyon

### Desteklenen Modeller

| Model | Provider | Açıklama |
|-------|----------|----------|
| `gemini-3-pro` | Google | En güçlü Gemini |
| `gemini-flash` | Google | Hızlı ve ekonomik |
| `hf-llama-3.1-70b` | HuggingFace | En güçlü açık kaynak |
| `hf-qwen-2.5-72b` | HuggingFace | Alibaba Qwen |
| `ollama:qwen2.5` | Ollama | Yerel model |

### Model Ekleme

`configs/models/models.yaml` dosyasını düzenleyin:

```yaml
custom_model:
  id: "custom-model"
  name: "Custom Model"
  provider: "hf"
  model_id: "org/model-name"
  description: "Model açıklaması"
```

## 🐳 Docker ile Çalıştırma

```bash
# Build ve başlat
docker-compose up --build

# Arka planda çalıştır
docker-compose up -d

# Logları izle
docker-compose logs -f
```

## 🔌 API Endpoint'leri

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/models` | GET | Model listesi |
| `/sessions` | GET | Oturum listesi |
| `/sessions/{id}` | GET | Oturum detayı |
| `/sessions` | POST | Yeni oturum |
| `/chat` | POST | Senkron chat |
| `/chat/stream` | POST | SSE streaming |
| `/health/` | GET | Sağlık kontrolü |

### WebSocket

```javascript
// Bağlantı
const ws = new WebSocket('ws://localhost:8000/ws/chat/');

// Mesaj gönder
ws.send(JSON.stringify({
  modelId: 'gemini-flash',
  messages: [{ role: 'user', content: 'Merhaba!' }],
  useAgent: false,  // Agent modu
  useRag: false     // RAG modu
}));

// Yanıt al
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.delta) console.log(data.delta);  // Streaming token
  if (data.done) console.log('Tamamlandı');
};
```

## 🧪 Test

```bash
# Backend testleri
cd backend && python -m pytest

# Frontend testleri
cd frontend && npm test
```

## 📝 Geliştirme Notları

### Yeni Tool Ekleme (Agents)

```python
# backend/agents/__init__.py

@dataclass
class MyTool(Tool):
    name: str = "my_tool"
    description: str = "Tool açıklaması"
    parameters: Dict = field(default_factory=lambda: {
        "input": "Parametre açıklaması"
    })
    
    async def execute(self, **kwargs) -> ToolResult:
        result = do_something(kwargs.get('input'))
        return ToolResult(
            success=True,
            output=str(result),
            tool_name=self.name
        )
```

### Yeni Adapter Ekleme

```python
# backend/adapters/custom.py

async def stream(messages, model_id, **kwargs):
    """Streaming yanıt üret."""
    async for token in api_call(messages):
        yield token

async def generate(messages, model_id, **kwargs):
    """Tam yanıt üret."""
    return await full_api_call(messages)
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👤 Geliştirici

**Emre Hacıhassan**

- GitHub: [@EmreHacihassan](https://github.com/EmreHacihassan)

## 🙏 Teşekkürler

- [Google Gemini](https://ai.google.dev/) - AI modelleri
- [HuggingFace](https://huggingface.co/) - Inference API
- [Ollama](https://ollama.ai/) - Yerel model çalıştırma
- [React](https://reactjs.org/) - Frontend framework
- [Django](https://www.djangoproject.com/) - Backend framework

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
