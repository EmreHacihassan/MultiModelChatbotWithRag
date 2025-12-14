#!/usr/bin/env python
"""RAG Test Script"""
import sys
import time
sys.path.insert(0, '.')

print("=" * 50)
print("RAG Pipeline Test")
print("=" * 50)

# Test dosyası oluştur
print("\n1. Test dosyası oluşturuluyor...")
with open('test_document.txt', 'w', encoding='utf-8') as f:
    f.write("""
    Bu bir test dökümanıdır.
    
    RAG (Retrieval-Augmented Generation) sistemi için test içeriği.
    
    Bu döküman şunları içerir:
    - Türkçe karakterler: ğüşiöçĞÜŞİÖÇ
    - Önemli bilgiler
    - Test verileri
    
    Python programlama dili hakkında bilgiler:
    Python, yüksek seviyeli, genel amaçlı bir programlama dilidir.
    Guido van Rossum tarafından 1991 yılında oluşturulmuştur.
    
    Django framework'ü hakkında:
    Django, Python ile yazılmış ücretsiz ve açık kaynaklı bir web framework'üdür.
    """)
print("✓ Test dosyası oluşturuldu: test_document.txt")

# RAG Pipeline'ı yükle
print("\n2. RAG Pipeline yükleniyor...")
start = time.time()
from rag.pipelines import RAGPipeline
print(f"   Import süresi: {time.time()-start:.2f}s")

start = time.time()
rag = RAGPipeline()
print(f"   Init süresi: {time.time()-start:.2f}s")
print("✓ RAG Pipeline hazır!")

# Döküman ekle
print("\n3. Döküman ekleniyor...")
start = time.time()
result = rag.add_document('test_document.txt')
print(f"   Ekleme süresi: {time.time()-start:.2f}s")
print(f"   Sonuç: {result}")

# Search test
print("\n4. Arama testi yapılıyor...")
start = time.time()
search_result = rag.search("Python nedir?")
print(f"   Arama süresi: {time.time()-start:.2f}s")
print(f"   Bulunan sonuç: {len(search_result)} döküman")

# Dökümanları listele
print("\n5. Dökümanlar listeleniyor...")
docs = rag.list_documents()
print(f"   Toplam döküman: {len(docs)}")
for doc in docs:
    print(f"   - {doc.get('file_name')}: {doc.get('chunk_count')} chunk")

print("\n" + "=" * 50)
print("✓ Tüm testler tamamlandı!")
print("=" * 50)
