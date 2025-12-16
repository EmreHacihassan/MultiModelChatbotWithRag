MyChatbot

Bu repo benim uzun süredir kafamda olan bir fikrin somut hale gelmiş hali.  
“Bir chatbot yapayım” diye değil, *gerçekten kullanılan AI mimarileri nasıl çalışıyor* görmek için yazıldı.  
Basit prompt → cevap olayından özellikle kaçındım.


Birden fazla AI modeliyle konuşabiliyorsun (Gemini, HuggingFace, Ollama)
Kendi PDF / DOCX / TXT dosyalarını yüklüyorsun
Sonra *sadece o dökümanlara bakarak* sana cevap veriyor (RAG)
Cevaplar WebSocket üzerinden anlık geliyor
- Gerekirse AI kendi kendine araç kullanıyor (hesaplama, arama, Python vs.)



Benim derdim şuydu:
- Tek modele bağımlı olmadan sistem kurmak
- Kendi verisini AI’ya bağlamak
- Gerçek zamanlı bir deneyim vermek
- AI’ya tool kullandırmak
- Ve bütün bunları gerçekten çalışır halde tutmak


React (Frontend) ↓ WebSocket / REST Django + Channels (Backend) ↓ AI Adapter Layer ↓ RAG Pipeline (SentenceTransformer + ChromaDB)

- Frontend: React 19 + Vite  
- Backend: Django + Channels  
- Embedding: Sentence Transformers  
- Vector DB: ChromaDB  

---

## RAG Nasıl Çalışıyor?

1. Döküman yüklüyorsun  
2. Metin küçük parçalara bölünüyor  
3. Her parça embedding’e çevriliyor  
4. ChromaDB’ye kaydediliyor  
5. Soru gelince en alakalı parçalar çekiliyor  
6. Model sadece bunları kullanarak cevap üretiyor  

Sayfa numarası içeren sorularda (“9. sayfada ne var?” gibi) semantik tahmin yapılmıyor.  
Direkt sayfa bazlı arama devreye giriyor.


Bu projede asıl zaman alan şey kod yazmak değil, *yanlış kararları fark edip düzeltmekti*.

### Embedding Modeli Yavaşlığı  
Model her sorguda yeniden yükleniyordu.  
İlk RAG isteği dakikalar sürüyordu.  
Çözüm: Singleton + backend açılışında preload.

### Thread Problemi  
Daemon thread kullandığım için embedding işlemi yarıda kalıyordu.  
Ekranda “processing” yazıp kalıyordu.  
Çözüm: Kritik işleri senkron yapmak.

### UTF-8 BOM Saçmalığı  
Windows yüzünden JSON dosyaları bazen okunmuyordu.  
Çözüm: Okurken utf-8-sig, yazarken düz utf-8.

### Chunking Sonsuz Döngü  
Bazı edge-case’lerde start değeri hiç ilerlemiyordu.  
Program donuyordu.  
Çözüm: İlerleme garantisi + iterasyon limiti.

Bunların hiçbiri tutorial’da anlatılmıyor.  
Ama gerçek projede mutlaka çıkıyor.



Agent modu açıkken chatbot şu araçları kullanabiliyor:

- Calculator  
- PythonExecutor  
- WebSearch  
- DateTime  
- Wikipedia  
- JSONParser  
- TextAnalyzer  
- UnitConverter  

Model ihtiyaca göre bu araçları çağırıp sonucu cevabına ekliyor.




```bash
python run.py

Frontend: http://localhost:3002

Backend: http://localhost:8000

WebSocket: ws://localhost:8000/ws/chat/



---

Dosya Yapısı (Kritik Olanlar)

MyChatbot/
├── run.py
├── backend/
│   ├── adapters/
│   ├── agents/
│   ├── websockets/
├── rag/
│   └── pipelines/
├── frontend/
│   └── src/
├── configs/
└── data/


---

İleride Ne Var?

Multi-modal RAG (görsel + metin)

Konuşma bağlamını hatırlayan RAG

PDF / Excel tablo sorgulama

URL’den direkt döküman ekleme



---

Kapanış

Bu proje “AI bana yazsın” diye yapılmadı.
AI sistemleri gerçekte nerede patlıyor, neden patlıyor ve nasıl toparlanıyor görmek için yapıldı.

Okuyup geçen değil, açıp kurcalayan biri için anlamlı.