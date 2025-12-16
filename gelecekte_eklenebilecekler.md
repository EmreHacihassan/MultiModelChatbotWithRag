# 🚀 MyChatbot - Gelecekte Eklenebilecek Özellikler

Bu döküman, RAG sistemi için planlanan ileri düzey özellikleri ve uygulama detaylarını içerir.

---

## 📋 Özellik Listesi

| # | Özellik | Zorluk | Etki | Durum |
|---|---------|--------|------|-------|
| 1 | Multi-Modal RAG (Görsel + Metin) | ⭐⭐⭐ | 🔥🔥🔥 | 📋 Planlandı |
| 2 | Conversational RAG (Bağlam Hatırlama) | ⭐⭐ | 🔥🔥🔥 | 📋 Planlandı |
| 3 | Table QA (Tablo Sorgulama) | ⭐⭐⭐ | 🔥🔥🔥 | 📋 Planlandı |
| 4 | Çoklu Döküman Karşılaştırma | ⭐⭐ | 🔥🔥🔥 | 📋 Planlandı |
| 5 | Otomatik Özet ve Rapor | ⭐ | 🔥🔥 | 📋 Planlandı |
| 6 | Semantik Kod Arama | ⭐⭐ | 🔥🔥 | 📋 Planlandı |
| 7 | Web URL'den Döküman Ekleme | ⭐⭐ | 🔥🔥 | 📋 Planlandı |
| 8 | Zaman Bazlı Sorgulama | ⭐⭐ | 🔥🔥 | 📋 Planlandı |
| 9 | Kaynak Güvenilirlik Skoru | ⭐⭐ | 🔥🔥 | 📋 Planlandı |
| 10 | Otomatik Soru Önerisi | ⭐ | 🔥🔥 | 📋 Planlandı |

---

## 1. 🖼️ Multi-Modal RAG (Görsel + Metin)

### Açıklama
PDF'lerdeki grafik, tablo, resim ve diyagramları da anlayabilen sistem. Sadece metin değil, görsel içerikleri de analiz edebilir.

### Kullanım Senaryosu
```
Kullanıcı: "Bu PDF'deki pasta grafiğinde en büyük dilim hangi kategori?"
AI: "Grafikteki en büyük dilim %45 ile 'Teknoloji' kategorisi. İkinci sırada %30 ile 'Sağlık' var."

Kullanıcı: "Şemadaki akış diyagramını açıkla"
AI: "Diyagramda 5 adımlı bir süreç var: 1. Başvuru → 2. İnceleme → 3. Onay → ..."
```

### Teknik Gereksinimler
- **Model**: GPT-4V, Gemini Pro Vision, veya LLaVA
- **Kütüphaneler**: `pdf2image`, `pytesseract`, `Pillow`
- **Ek Depolama**: Görsel embedding'ler için CLIP modeli

### Uygulama Adımları
```python
# 1. PDF'den görsel çıkarma
from pdf2image import convert_from_path

def extract_images_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    image_data = []
    for i, img in enumerate(images):
        # OCR ile metin çıkar
        text = pytesseract.image_to_string(img)
        # Görsel embedding oluştur
        embedding = clip_model.encode(img)
        image_data.append({
            'page': i + 1,
            'image': img,
            'ocr_text': text,
            'embedding': embedding
        })
    return image_data

# 2. Görsel arama
def search_images(query, image_embeddings):
    query_embedding = clip_model.encode(query)
    similarities = cosine_similarity(query_embedding, image_embeddings)
    return sorted(similarities, reverse=True)[:5]
```

### Tahmini Süre
- Geliştirme: 2-3 gün
- Test: 1 gün

---

## 2. 💬 Conversational RAG (Bağlam Hatırlama)

### Açıklama
Önceki soruları ve cevapları hatırlayarak takip sorularına doğru cevap verebilen sistem. "O", "bu", "onun" gibi referansları çözebilir.

### Kullanım Senaryosu
```
Kullanıcı: "Bu kitapta Ahmet kim?"
AI: "Ahmet, romanın ana karakteri ve bir yazılım mühendisi. İstanbul'da yaşıyor."

Kullanıcı: "Peki onun karısı ne iş yapıyor?"  ← "onun" = Ahmet
AI: "Ahmet'in karısı Ayşe, bir ilkokul öğretmeni olarak çalışıyor."

Kullanıcı: "Çocukları var mı?"  ← Hala Ahmet ve Ayşe'den bahsediyor
AI: "Evet, Ahmet ve Ayşe'nin iki çocuğu var: 8 yaşında Elif ve 5 yaşında Can."
```

### Teknik Gereksinimler
- **Conversation Memory**: Son N mesajı context'e ekle
- **Coreference Resolution**: "o", "bu" gibi referansları çöz
- **Query Rewriting**: Eksik bağlamı tamamla

### Uygulama Adımları
```python
class ConversationalRAG:
    def __init__(self, rag_pipeline, memory_size=10):
        self.rag = rag_pipeline
        self.memory_size = memory_size
        self.conversation_history = []
    
    def rewrite_query(self, query: str, history: list) -> str:
        """Bağlamı tamamlayarak sorguyu yeniden yaz."""
        if not history:
            return query
        
        # Son konuşmaları context olarak kullan
        context = "\n".join([
            f"Kullanıcı: {h['user']}\nAI: {h['assistant']}"
            for h in history[-3:]
        ])
        
        rewrite_prompt = f"""
        Önceki konuşma:
        {context}
        
        Yeni soru: {query}
        
        Bu soruyu, önceki konuşma bağlamını da içerecek şekilde yeniden yaz.
        Örnek: "onun karısı" → "Ahmet'in karısı"
        """
        
        return llm.complete(rewrite_prompt)
    
    def chat(self, query: str) -> str:
        # 1. Sorguyu yeniden yaz
        rewritten = self.rewrite_query(query, self.conversation_history)
        
        # 2. RAG araması yap
        docs = self.rag.search(rewritten)
        
        # 3. Cevap üret (conversation history ile)
        response = self.generate_response(query, docs, self.conversation_history)
        
        # 4. History'ye ekle
        self.conversation_history.append({
            'user': query,
            'assistant': response,
            'rewritten_query': rewritten
        })
        
        # Memory limit
        if len(self.conversation_history) > self.memory_size:
            self.conversation_history.pop(0)
        
        return response
```

### Tahmini Süre
- Geliştirme: 1-2 gün
- Test: 0.5 gün

---

## 3. 📊 Table QA (Tablo Sorgulama)

### Açıklama
PDF ve Excel'deki tabloları SQL benzeri doğal dil sorguları ile sorgulama. Sayısal analiz, filtreleme, toplama işlemleri.

### Kullanım Senaryosu
```
Kullanıcı: "2023 yılında en çok satış yapan ürün hangisi?"
AI: "Tabloya göre 2023'te en çok satan ürün 'Laptop Pro X' (15,230 adet). 
     İkinci sırada 'Tablet Y' (12,100 adet) var."

Kullanıcı: "Toplam satış geliri ne kadar?"
AI: "2023 yılı toplam satış geliri: 4,523,000 TL"

Kullanıcı: "Satışları aya göre karşılaştır"
AI: "En yüksek satış Aralık ayında (523,000 TL), en düşük Şubat'ta (212,000 TL)."
```

### Teknik Gereksinimler
- **Kütüphaneler**: `pandas`, `tabula-py`, `openpyxl`
- **Tablo Çıkarma**: PDF'den tablo algılama
- **NL2SQL**: Doğal dili SQL'e çevirme (veya Pandas query)

### Uygulama Adımları
```python
import pandas as pd
import tabula

class TableQA:
    def __init__(self):
        self.tables = {}  # file_id -> list of DataFrames
    
    def extract_tables_from_pdf(self, pdf_path: str) -> list:
        """PDF'den tabloları çıkar."""
        tables = tabula.read_pdf(pdf_path, pages='all')
        return tables
    
    def extract_tables_from_excel(self, excel_path: str) -> list:
        """Excel'den tabloları çıkar."""
        xlsx = pd.ExcelFile(excel_path)
        tables = [pd.read_excel(xlsx, sheet) for sheet in xlsx.sheet_names]
        return tables
    
    def query_table(self, table: pd.DataFrame, query: str) -> str:
        """Doğal dil sorgusu ile tabloyu sorgula."""
        # Tablo şemasını LLM'e ver
        schema = f"Sütunlar: {list(table.columns)}\nSatır sayısı: {len(table)}"
        
        # LLM'den Pandas kodu iste
        code_prompt = f"""
        Tablo şeması: {schema}
        İlk 3 satır: {table.head(3).to_string()}
        
        Kullanıcı sorusu: {query}
        
        Bu soruyu cevaplamak için gerekli Pandas kodunu yaz.
        Sadece kodu yaz, açıklama yazma.
        Değişken adı 'df' olsun.
        """
        
        pandas_code = llm.complete(code_prompt)
        
        # Kodu güvenli şekilde çalıştır
        result = self.safe_execute(pandas_code, {'df': table})
        
        return result
    
    def safe_execute(self, code: str, variables: dict):
        """Pandas kodunu güvenli şekilde çalıştır."""
        allowed_modules = {'pd': pd, 'np': np}
        local_vars = {**allowed_modules, **variables}
        
        try:
            exec(code, {"__builtins__": {}}, local_vars)
            return local_vars.get('result', 'Sonuç bulunamadı')
        except Exception as e:
            return f"Sorgu çalıştırılamadı: {e}"
```

### Tahmini Süre
- Geliştirme: 2-3 gün
- Test: 1 gün

---

## 4. 🔗 Çoklu Döküman Karşılaştırma

### Açıklama
Birden fazla dökümanı karşılaştırıp farklılıkları, benzerlikleri ve çelişkileri tespit etme.

### Kullanım Senaryosu
```
Kullanıcı: "Bu iki sözleşme arasındaki farklar neler?"
AI: "İki sözleşme arasındaki temel farklar:

1. **Süre**: Sözleşme A: 12 ay, Sözleşme B: 24 ay
2. **Fiyat**: A: 10,000 TL/ay, B: 8,500 TL/ay
3. **Fesih**: A: 30 gün önceden bildirim, B: 60 gün
4. **Ceza**: A: 2 aylık ücret, B: 3 aylık ücret

⚠️ Çelişki: A'da 'tek taraflı fesih yok' yazıyor, B'de 'tek taraflı fesih mümkün'."

Kullanıcı: "Hangi sözleşme daha avantajlı?"
AI: "Kısa vadede A daha avantajlı (daha esnek fesih), uzun vadede B daha avantajlı (daha düşük aylık ücret)."
```

### Teknik Gereksinimler
- **Diff Algoritması**: Metinler arası fark bulma
- **Semantic Similarity**: Benzer bölümleri eşleştirme
- **Contradiction Detection**: Çelişkileri tespit etme

### Uygulama Adımları
```python
class DocumentComparator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def compare_documents(self, doc_ids: list, aspects: list = None) -> dict:
        """İki veya daha fazla dökümanı karşılaştır."""
        
        # Varsayılan karşılaştırma boyutları
        if aspects is None:
            aspects = ['süre', 'fiyat', 'koşullar', 'yükümlülükler', 'fesih']
        
        results = {
            'similarities': [],
            'differences': [],
            'contradictions': []
        }
        
        for aspect in aspects:
            # Her döküman için ilgili bölümü bul
            doc_contents = {}
            for doc_id in doc_ids:
                chunks = self.rag.search(
                    query=aspect,
                    filter={'document_id': doc_id}
                )
                doc_contents[doc_id] = chunks
            
            # Karşılaştır
            comparison = self.compare_aspect(aspect, doc_contents)
            
            if comparison['type'] == 'similar':
                results['similarities'].append(comparison)
            elif comparison['type'] == 'different':
                results['differences'].append(comparison)
            elif comparison['type'] == 'contradiction':
                results['contradictions'].append(comparison)
        
        return results
    
    def compare_aspect(self, aspect: str, doc_contents: dict) -> dict:
        """Belirli bir boyutta karşılaştırma yap."""
        prompt = f"""
        Aşağıdaki döküman bölümlerini '{aspect}' açısından karşılaştır.
        
        Dökümanlar:
        {json.dumps(doc_contents, ensure_ascii=False, indent=2)}
        
        Çıktı formatı:
        - type: "similar" | "different" | "contradiction"
        - aspect: karşılaştırma boyutu
        - details: her döküman için detay
        - summary: özet karşılaştırma
        """
        
        return llm.complete(prompt, output_format='json')
```

### Tahmini Süre
- Geliştirme: 2 gün
- Test: 1 gün

---

## 5. 📝 Otomatik Özet ve Rapor

### Açıklama
Dökümanlardan otomatik özet, madde işaretli liste, executive summary veya detaylı rapor oluşturma.

### Kullanım Senaryosu
```
Kullanıcı: "Bu 50 sayfalık raporu 5 maddede özetle"
AI: "📋 Rapor Özeti:
1. Şirket 2023'te %20 büyüme kaydetti
2. Yeni ürün lansmanı Q2'de gerçekleşti
3. Müşteri memnuniyeti %85'e yükseldi
4. Operasyonel maliyetler %10 düştü
5. 2024 hedefi: Uluslararası pazara açılım"

Kullanıcı: "Detaylı bir executive summary yaz"
AI: "## Executive Summary
### Genel Bakış
[Detaylı özet...]

### Finansal Performans
[Tablo ve grafikler...]

### Öneriler
[Aksiyon maddeleri...]"
```

### Teknik Gereksinimler
- **Summarization**: Extractive veya Abstractive özet
- **Template Engine**: Farklı rapor formatları
- **Section Detection**: Bölüm başlıklarını algılama

### Uygulama Adımları
```python
class DocumentSummarizer:
    TEMPLATES = {
        'bullet_points': "Dökümanı {count} madde ile özetle.",
        'executive_summary': "Yönetici özeti formatında detaylı özet yaz.",
        'key_findings': "Ana bulgular ve sonuçları listele.",
        'action_items': "Gerekli aksiyon maddelerini çıkar.",
    }
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def summarize(self, doc_id: str, template: str = 'bullet_points', **kwargs) -> str:
        """Dökümanı belirtilen formatta özetle."""
        
        # Tüm döküman içeriğini al
        chunks = self.rag.get_all_chunks(doc_id)
        full_text = "\n".join([c['text'] for c in chunks])
        
        # Çok uzunsa hierarchical summarization
        if len(full_text) > 10000:
            return self.hierarchical_summarize(chunks, template, **kwargs)
        
        prompt = self.TEMPLATES[template].format(**kwargs)
        
        return llm.complete(f"{prompt}\n\nDöküman:\n{full_text}")
    
    def hierarchical_summarize(self, chunks: list, template: str, **kwargs) -> str:
        """Büyük dökümanlar için kademeli özet."""
        
        # 1. Her chunk'ı özetle
        chunk_summaries = []
        for chunk in chunks:
            summary = llm.complete(f"Kısaca özetle:\n{chunk['text']}")
            chunk_summaries.append(summary)
        
        # 2. Özetleri birleştir ve son özeti yap
        combined = "\n".join(chunk_summaries)
        final_prompt = self.TEMPLATES[template].format(**kwargs)
        
        return llm.complete(f"{final_prompt}\n\nİçerik:\n{combined}")
    
    def generate_report(self, doc_ids: list, report_type: str = 'comprehensive') -> str:
        """Birden fazla döküman için rapor oluştur."""
        
        sections = []
        
        for doc_id in doc_ids:
            doc_info = self.rag.get_document_info(doc_id)
            summary = self.summarize(doc_id, 'executive_summary')
            
            sections.append({
                'title': doc_info['file_name'],
                'summary': summary
            })
        
        # Rapor şablonu
        report = f"""
# 📊 Döküman Analiz Raporu
Tarih: {datetime.now().strftime('%d.%m.%Y')}
Analiz edilen döküman sayısı: {len(doc_ids)}

---

"""
        for section in sections:
            report += f"## {section['title']}\n{section['summary']}\n\n---\n\n"
        
        return report
```

### Tahmini Süre
- Geliştirme: 1 gün
- Test: 0.5 gün

---

## 6. 🔍 Semantik Kod Arama

### Açıklama
Kod dosyalarını (Python, JavaScript, vb.) yükleyip doğal dil ile arama yapabilme. Fonksiyon, class ve metotları anlama.

### Kullanım Senaryosu
```
Kullanıcı: "Kullanıcı authentication yapan fonksiyon hangisi?"
AI: "auth/views.py dosyasındaki `login_user()` fonksiyonu kullanıcı doğrulaması yapıyor:

```python
def login_user(request):
    # ... kod ...
```

Ayrıca `verify_token()` fonksiyonu da ilgili."

Kullanıcı: "Bu fonksiyonu kullanan yerler neresi?"
AI: "login_user() fonksiyonu 3 yerde kullanılıyor:
1. api/routes.py:45 - /login endpoint'i
2. middleware/auth.py:23 - session kontrolü
3. tests/test_auth.py:12 - unit test"
```

### Teknik Gereksinimler
- **Code Parser**: AST parsing (tree-sitter)
- **Code Embeddings**: CodeBERT veya StarCoder
- **Symbol Extraction**: Fonksiyon, class, import çıkarma

### Uygulama Adımları
```python
import ast
from tree_sitter import Language, Parser

class CodeSearchEngine:
    SUPPORTED_LANGUAGES = ['python', 'javascript', 'typescript']
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.code_index = {}
    
    def index_code_file(self, file_path: str) -> list:
        """Kod dosyasını indexle."""
        
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Dile göre parse et
        language = self.detect_language(file_path)
        symbols = self.extract_symbols(code, language)
        
        indexed = []
        for symbol in symbols:
            embedding = self.embedding_model.encode(
                f"{symbol['type']} {symbol['name']}: {symbol['docstring']}\n{symbol['code']}"
            )
            
            indexed.append({
                'file': file_path,
                'type': symbol['type'],  # function, class, method
                'name': symbol['name'],
                'line': symbol['line'],
                'code': symbol['code'],
                'docstring': symbol['docstring'],
                'embedding': embedding
            })
        
        return indexed
    
    def extract_symbols(self, code: str, language: str) -> list:
        """Koddan sembolleri çıkar."""
        
        if language == 'python':
            return self.extract_python_symbols(code)
        elif language in ['javascript', 'typescript']:
            return self.extract_js_symbols(code)
    
    def extract_python_symbols(self, code: str) -> list:
        """Python kodundan fonksiyon ve class'ları çıkar."""
        
        tree = ast.parse(code)
        symbols = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'code': ast.get_source_segment(code, node),
                    'docstring': ast.get_docstring(node) or ''
                })
            elif isinstance(node, ast.ClassDef):
                symbols.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'code': ast.get_source_segment(code, node),
                    'docstring': ast.get_docstring(node) or ''
                })
        
        return symbols
    
    def search_code(self, query: str, top_k: int = 5) -> list:
        """Doğal dil ile kod ara."""
        
        query_embedding = self.embedding_model.encode(query)
        
        results = []
        for symbol in self.code_index.values():
            similarity = cosine_similarity(query_embedding, symbol['embedding'])
            results.append({**symbol, 'score': similarity})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
```

### Tahmini Süre
- Geliştirme: 2-3 gün
- Test: 1 gün

---

## 7. 🌐 Web URL'den Döküman Ekleme

### Açıklama
URL vererek web sayfasını veya online dökümanı RAG sistemine ekleme. HTML parsing, PDF download, sitemap crawling.

### Kullanım Senaryosu
```
Kullanıcı: "https://docs.python.org/3/tutorial adresini ekle"
AI: "✅ Python Tutorial sayfası eklendi!
     - 45 sayfa işlendi
     - 230 metin parçası oluşturuldu
     - Hazır sorulabilir!"

Kullanıcı: "Python'da list comprehension nasıl kullanılır?"
AI: "[Tutorial'dan] List comprehension şöyle kullanılır:
     squares = [x**2 for x in range(10)]
     ..."
```

### Teknik Gereksinimler
- **Web Scraping**: `requests`, `BeautifulSoup`, `trafilatura`
- **PDF Download**: URL'den PDF indirme
- **Rate Limiting**: Siteleri yormamak için gecikme
- **Robots.txt**: Kurallara uyum

### Uygulama Adımları
```python
import requests
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse

class WebDocumentLoader:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MyChatbot/1.0 (RAG Document Indexer)'
        })
    
    def load_url(self, url: str, crawl_depth: int = 0) -> dict:
        """URL'den içerik yükle."""
        
        # URL tipini belirle
        if url.endswith('.pdf'):
            return self.load_pdf_url(url)
        else:
            return self.load_html_url(url, crawl_depth)
    
    def load_html_url(self, url: str, crawl_depth: int = 0) -> dict:
        """HTML sayfasını yükle ve işle."""
        
        # Robots.txt kontrolü
        if not self.check_robots_txt(url):
            return {'error': 'robots.txt tarafından engellendi'}
        
        # Sayfayı indir
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        # Ana içeriği çıkar (reklam, menü vs. hariç)
        text = trafilatura.extract(response.text)
        
        if not text:
            # Fallback: BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Script ve style'ları kaldır
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)
        
        # Metadata
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else urlparse(url).path
        
        # RAG'a ekle
        result = self.rag.add_text(
            text=text,
            metadata={
                'source': 'web',
                'url': url,
                'title': title,
                'crawled_at': datetime.now().isoformat()
            }
        )
        
        # Crawl depth > 0 ise linkleri takip et
        if crawl_depth > 0:
            links = self.extract_links(response.text, url)
            for link in links[:10]:  # Max 10 link
                self.load_url(link, crawl_depth - 1)
        
        return result
    
    def load_pdf_url(self, url: str) -> dict:
        """PDF URL'sini indir ve işle."""
        
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        
        # Geçici dosyaya kaydet
        temp_path = f"/tmp/{hashlib.md5(url.encode()).hexdigest()}.pdf"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # RAG'a ekle
        result = self.rag.add_document(
            temp_path,
            metadata={'source': 'web', 'url': url}
        )
        
        # Temp dosyayı sil
        os.remove(temp_path)
        
        return result
    
    def check_robots_txt(self, url: str) -> bool:
        """robots.txt kurallarını kontrol et."""
        from urllib.robotparser import RobotFileParser
        
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            return rp.can_fetch('MyChatbot', url)
        except:
            return True  # robots.txt yoksa izin ver
```

### Tahmini Süre
- Geliştirme: 1-2 gün
- Test: 0.5 gün

---

## 8. ⏰ Zaman Bazlı Sorgulama

### Açıklama
Dökümanları tarih/zaman bilgisine göre filtreleme. "Geçen yıl", "son 3 ayda", "2022'de" gibi sorgular.

### Kullanım Senaryosu
```
Kullanıcı: "2022 raporunda gelir ne kadardı?"
AI: "2022 Yıllık Raporu'na göre toplam gelir 5.2 milyon TL idi."

Kullanıcı: "2023 ile karşılaştır"
AI: "2022 → 2023 karşılaştırması:
     - Gelir: 5.2M → 6.8M (+30%)
     - Gider: 4.1M → 4.5M (+10%)
     - Kâr: 1.1M → 2.3M (+109%)"

Kullanıcı: "Son 3 yılın trendini göster"
AI: "[Grafik açıklaması] Gelir sürekli artış trendinde..."
```

### Teknik Gereksinimler
- **Date Extraction**: Dökümanlardan tarih çıkarma
- **Date Parser**: Doğal dil tarih ifadelerini parse etme
- **Time Filter**: Metadata'da tarih filtresi

### Uygulama Adımları
```python
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import re

class TemporalRAG:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def parse_time_expression(self, query: str) -> dict:
        """Doğal dil zaman ifadesini parse et."""
        
        now = datetime.now()
        
        patterns = {
            r'(\d{4})(?:\s*yılı)?': lambda m: {
                'start': datetime(int(m.group(1)), 1, 1),
                'end': datetime(int(m.group(1)), 12, 31)
            },
            r'geçen\s*yıl': lambda m: {
                'start': datetime(now.year - 1, 1, 1),
                'end': datetime(now.year - 1, 12, 31)
            },
            r'son\s*(\d+)\s*ay': lambda m: {
                'start': now - relativedelta(months=int(m.group(1))),
                'end': now
            },
            r'son\s*(\d+)\s*yıl': lambda m: {
                'start': now - relativedelta(years=int(m.group(1))),
                'end': now
            },
            r'bu\s*yıl': lambda m: {
                'start': datetime(now.year, 1, 1),
                'end': now
            },
        }
        
        for pattern, handler in patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                return handler(match)
        
        return None
    
    def search_with_time(self, query: str, top_k: int = 5) -> list:
        """Zaman filtreli arama."""
        
        time_range = self.parse_time_expression(query)
        
        if time_range:
            # Metadata filtresi oluştur
            filter_dict = {
                'document_date': {
                    '$gte': time_range['start'].isoformat(),
                    '$lte': time_range['end'].isoformat()
                }
            }
            return self.rag.search(query, top_k=top_k, filter=filter_dict)
        else:
            return self.rag.search(query, top_k=top_k)
    
    def extract_document_date(self, text: str, filename: str) -> datetime:
        """Döküman tarihini çıkar."""
        
        # 1. Dosya adından
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return datetime(int(year_match.group(1)), 1, 1)
        
        # 2. İçerikten
        date_patterns = [
            r'Tarih:\s*(\d{1,2}[./]\d{1,2}[./]\d{4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'(\d{4})\s*(?:Yılı|yılı|YILI)',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return date_parser.parse(match.group(1), dayfirst=True)
                except:
                    continue
        
        return None
```

### Tahmini Süre
- Geliştirme: 1-2 gün
- Test: 0.5 gün

---

## 9. 🎯 Kaynak Güvenilirlik Skoru

### Açıklama
Farklı kaynakların güvenilirliğini skorlama ve kullanıcıya gösterme. Resmi döküman vs blog yazısı vs forum mesajı.

### Kullanım Senaryosu
```
Kullanıcı: "Python'da memory leak nasıl önlenir?"

AI: "Bu konuda 3 farklı kaynaktan bilgi buldum:

📗 **Python Resmi Dökümantasyonu** (Güvenilirlik: %95)
   gc modülü ve weakref kullanımı öneriliyor.

📘 **Real Python Blog** (Güvenilirlik: %80)
   Profiling araçları ve best practice'ler anlatılmış.

📕 **Stack Overflow Cevabı** (Güvenilirlik: %60)
   Topluluk tarafından onaylanmış pratik çözümler.

⚠️ Not: Resmi dökümantasyonu temel almanızı öneririm."
```

### Teknik Gereksinimler
- **Source Classification**: Kaynak tipini belirleme
- **Credibility Scoring**: Güvenilirlik puanlama
- **Cross-reference**: Çapraz doğrulama

### Uygulama Adımları
```python
class SourceCredibility:
    # Kaynak tipi skorları
    SOURCE_SCORES = {
        'official_docs': 0.95,      # Resmi dökümantasyon
        'academic_paper': 0.90,     # Akademik makale
        'technical_book': 0.85,     # Teknik kitap
        'reputable_blog': 0.75,     # Güvenilir blog
        'tutorial': 0.70,           # Tutorial
        'forum_accepted': 0.65,     # Kabul edilen forum cevabı
        'blog_post': 0.55,          # Blog yazısı
        'forum_post': 0.40,         # Forum mesajı
        'unknown': 0.30,            # Bilinmeyen kaynak
    }
    
    # Domain bazlı güvenilirlik
    TRUSTED_DOMAINS = {
        'docs.python.org': 'official_docs',
        'developer.mozilla.org': 'official_docs',
        'arxiv.org': 'academic_paper',
        'realpython.com': 'reputable_blog',
        'stackoverflow.com': 'forum_accepted',
    }
    
    def __init__(self):
        self.cross_reference_cache = {}
    
    def calculate_credibility(self, chunk: dict) -> float:
        """Chunk için güvenilirlik skoru hesapla."""
        
        metadata = chunk.get('metadata', {})
        
        # 1. Kaynak tipi skoru
        source_type = self.classify_source(metadata)
        base_score = self.SOURCE_SCORES.get(source_type, 0.30)
        
        # 2. Çapraz doğrulama bonusu
        cross_ref_bonus = self.check_cross_reference(chunk['text'])
        
        # 3. Tarih cezası (eski içerik)
        date_penalty = self.calculate_date_penalty(metadata.get('date'))
        
        # 4. Final skor
        final_score = min(1.0, base_score + cross_ref_bonus - date_penalty)
        
        return round(final_score, 2)
    
    def classify_source(self, metadata: dict) -> str:
        """Kaynağı sınıflandır."""
        
        url = metadata.get('url', '')
        file_type = metadata.get('file_type', '')
        
        # URL'den domain kontrolü
        for domain, source_type in self.TRUSTED_DOMAINS.items():
            if domain in url:
                return source_type
        
        # Dosya tipine göre
        if file_type == 'pdf':
            if 'academic' in metadata.get('title', '').lower():
                return 'academic_paper'
            return 'technical_book'
        
        return 'unknown'
    
    def check_cross_reference(self, text: str) -> float:
        """Çapraz doğrulama - aynı bilgi başka kaynaklarda var mı?"""
        
        # Basit implementasyon: aynı anahtar kavramlar kaç kaynakta geçiyor
        # Gerçek implementasyonda semantic similarity kullanılır
        
        # Bonus: 0 - 0.15 arası
        return 0.0
    
    def calculate_date_penalty(self, date_str: str) -> float:
        """Eski içerik cezası."""
        
        if not date_str:
            return 0.05  # Tarih yoksa küçük ceza
        
        try:
            doc_date = date_parser.parse(date_str)
            age_years = (datetime.now() - doc_date).days / 365
            
            if age_years > 5:
                return 0.15
            elif age_years > 2:
                return 0.05
            return 0
        except:
            return 0.05
    
    def format_credibility_display(self, score: float) -> str:
        """Güvenilirlik skorunu görsel formatta göster."""
        
        percentage = int(score * 100)
        
        if score >= 0.9:
            emoji = "📗"
            label = "Çok Güvenilir"
        elif score >= 0.7:
            emoji = "📘"
            label = "Güvenilir"
        elif score >= 0.5:
            emoji = "📙"
            label = "Orta"
        else:
            emoji = "📕"
            label = "Düşük"
        
        return f"{emoji} {label} (%{percentage})"
```

### Tahmini Süre
- Geliştirme: 1-2 gün
- Test: 0.5 gün

---

## 10. 💡 Otomatik Soru Önerisi

### Açıklama
Döküman yüklendikten sonra "Bu döküman hakkında şunları sorabilirsiniz" şeklinde akıllı soru önerileri.

### Kullanım Senaryosu
```
[Kullanıcı "sözleşme.pdf" yükledi]

AI: "✅ Sözleşme başarıyla yüklendi!

💡 Bu döküman hakkında sorabilecekleriniz:
1. Sözleşmenin süresi ne kadar?
2. Aylık/yıllık ücret ne kadar?
3. Fesih koşulları neler?
4. Tarafların yükümlülükleri neler?
5. Gizlilik maddeleri var mı?

📊 Döküman özellikleri:
- 12 sayfa
- Türkçe
- Hizmet sözleşmesi"
```

### Teknik Gereksinimler
- **Content Analysis**: Döküman içeriğini analiz etme
- **Question Generation**: İçerikten soru üretme
- **Document Classification**: Döküman tipini belirleme

### Uygulama Adımları
```python
class QuestionSuggester:
    # Döküman tipi bazlı soru şablonları
    QUESTION_TEMPLATES = {
        'contract': [
            "Sözleşmenin süresi ne kadar?",
            "Taraflar kimler?",
            "Ücret/bedel ne kadar?",
            "Fesih koşulları neler?",
            "Ceza maddeleri var mı?",
        ],
        'report': [
            "Raporun ana bulguları neler?",
            "Hangi dönem ele alınmış?",
            "Öneriler neler?",
            "Sonuç ve değerlendirme ne?",
        ],
        'manual': [
            "Temel özellikler neler?",
            "Nasıl kurulur/başlatılır?",
            "Sık karşılaşılan sorunlar neler?",
            "Teknik özellikler neler?",
        ],
        'academic': [
            "Çalışmanın amacı ne?",
            "Kullanılan metodoloji ne?",
            "Ana bulgular neler?",
            "Sonuç ve öneriler neler?",
        ],
        'default': [
            "Bu döküman ne hakkında?",
            "Ana konular neler?",
            "Önemli noktalar neler?",
        ]
    }
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def classify_document(self, text: str, filename: str) -> str:
        """Döküman tipini belirle."""
        
        text_lower = text.lower()
        
        # Anahtar kelime bazlı sınıflandırma
        if any(w in text_lower for w in ['sözleşme', 'taraflar', 'madde', 'yükümlülük']):
            return 'contract'
        elif any(w in text_lower for w in ['rapor', 'bulgu', 'analiz', 'sonuç']):
            return 'report'
        elif any(w in text_lower for w in ['kurulum', 'kullanım', 'özellik', 'manual']):
            return 'manual'
        elif any(w in text_lower for w in ['abstract', 'methodology', 'references', 'özet']):
            return 'academic'
        
        return 'default'
    
    def generate_questions(self, doc_id: str, max_questions: int = 5) -> list:
        """Döküman için soru önerileri üret."""
        
        # Döküman içeriğini al
        chunks = self.rag.get_all_chunks(doc_id)
        full_text = "\n".join([c['text'] for c in chunks[:5]])  # İlk 5 chunk
        filename = self.rag.get_document_info(doc_id)['file_name']
        
        # Döküman tipini belirle
        doc_type = self.classify_document(full_text, filename)
        
        # Şablon sorularını al
        template_questions = self.QUESTION_TEMPLATES.get(doc_type, self.QUESTION_TEMPLATES['default'])
        
        # LLM ile döküman-spesifik sorular üret
        custom_questions = self.generate_custom_questions(full_text, doc_type)
        
        # Birleştir ve sınırla
        all_questions = template_questions + custom_questions
        return all_questions[:max_questions]
    
    def generate_custom_questions(self, text: str, doc_type: str) -> list:
        """LLM ile döküman-spesifik sorular üret."""
        
        prompt = f"""
        Aşağıdaki {doc_type} dökümanı için kullanıcının sorabileceği 
        3 spesifik soru öner. Döküman içeriğine özgü, genel olmayan sorular olsun.
        
        Döküman (ilk kısım):
        {text[:2000]}
        
        Sadece soruları listele, başka bir şey yazma:
        1.
        2.
        3.
        """
        
        response = llm.complete(prompt)
        
        # Parse et
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # "1. Soru" formatını temizle
                question = re.sub(r'^\d+\.\s*', '', line)
                questions.append(question)
        
        return questions
    
    def get_document_summary_with_questions(self, doc_id: str) -> dict:
        """Döküman özeti ve soru önerileri."""
        
        doc_info = self.rag.get_document_info(doc_id)
        questions = self.generate_questions(doc_id)
        
        return {
            'file_name': doc_info['file_name'],
            'page_count': doc_info.get('page_count', 'N/A'),
            'chunk_count': doc_info.get('chunk_count', 0),
            'document_type': self.classify_document(
                self.rag.get_all_chunks(doc_id)[0]['text'],
                doc_info['file_name']
            ),
            'suggested_questions': questions,
            'message': f"✅ {doc_info['file_name']} başarıyla yüklendi!\n\n💡 Bu döküman hakkında sorabilecekleriniz:"
        }
```

### Tahmini Süre
- Geliştirme: 1 gün
- Test: 0.5 gün

---

## 📅 Uygulama Öncelik Sırası

Önerilen geliştirme sırası (bağımlılıklar ve etki göz önünde):

### Faz 1 - Temel İyileştirmeler (1 hafta)
1. ✅ Conversational RAG - En çok istenen özellik
2. ✅ Otomatik Soru Önerisi - Kullanıcı deneyimi
3. ✅ Otomatik Özet - Hızlı değer

### Faz 2 - Gelişmiş Özellikler (2 hafta)
4. Web URL'den Döküman
5. Çoklu Döküman Karşılaştırma
6. Zaman Bazlı Sorgulama

### Faz 3 - Uzman Özellikler (2-3 hafta)
7. Table QA
8. Semantik Kod Arama
9. Kaynak Güvenilirlik Skoru

### Faz 4 - İleri Düzey (3-4 hafta)
10. Multi-Modal RAG

---

## 🛠️ Gerekli Ek Kütüphaneler

```txt
# requirements.txt'e eklenecekler

# Multi-Modal
pdf2image==1.16.3
pytesseract==0.3.10
Pillow>=10.0.0
# clip-by-openai  # Opsiyonel

# Table QA
tabula-py==2.9.0
openpyxl==3.1.2

# Web Scraping
trafilatura==1.6.0
beautifulsoup4==4.12.2

# Code Search
tree-sitter==0.21.0

# Date Parsing
python-dateutil==2.8.2
```

---

## 📝 Notlar

- Her özellik bağımsız olarak eklenebilir
- Mevcut RAG altyapısı üzerine inşa edilecek
- Test coverage minimum %80 hedefleniyor
- Her özellik için API endpoint'i eklenecek
- Frontend entegrasyonu ayrı task olarak planlanacak
