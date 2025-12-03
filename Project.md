Proje Dokümantasyonu — Dinamik Çoklu Model Chatbot Sistemi
uzun kod severim senden istemediğim özellikleri senin bonus olarak eklemeni  severim 
türkçe cevap ver

1. Genel Bakış

Bu proje, kullanıcıya farklı yapay zeka modelleriyle tek bir arayüz üzerinden etkileşim kurma imkânı sağlayan dinamik, çoklu-model chatbot sistemidir. Sistem hem yerel modelleri (ör. Ollama) hem de API anahtarıyla çalışan bulut modellerini destekler.
Hedef; esnek, modüler bir yapı sağlayarak yeni modellerin veya yapay zeka yeteneklerinin eklenmesini son derece kolay bir hâle getirmektir.


---

2. Sistem Amacı

Bu sistemin hedefleri:

1. Birden fazla yapay zeka modeline bağlanabilen ölçeklenebilir bir chatbot sunmak.


2. Model değişimini sorunsuz bir hâle getirmek.


3. Aynı arayüz üzerinde yerel + bulut AI kullanımını mümkün kılmak.


4. Gerçek zamanlı iletişimi WebSocket destekli altyapıyla sağlamak.


5. RAG, akışlı yanıt (streaming) ve zengin mesaj türleri gibi özellikleri desteklemek.


6. Geliştiricilere minimum eforla yeni modelleri entegre edebilme imkânı sunmak.


7. Çoklu sağlayıcıları tek bir backend katmanında yönetebilmek.




---

3. Mimari Özeti

Sistem mimarisi şu bileşenlerden oluşur:

Frontend (React veya Vue)

Modern, sade bir sohbet arayüzü sağlar.

Model seçimi için bir seçim menüsü içerir.

Kullanıcı isteklerini modele uygun formatta backend’e gönderir.

Modellerden gelen yanıtları gerçek zamanlı işler.


Backend (Django + Opsiyonel WebSocket Katmanı)

Yapay zeka isteklerini merkezî bir noktada toplar.

Hangi modelin çağrıldığını belirler.

İlgili model bağlayıcılarını (connectors) çalıştırır.

Yanıtları akışlı veya tam hâlde frontend’e iletir.


Model Bağlayıcıları (Connectors)

Her model için:

İstek formatlama,

Yanıt işleme,

Hata yönetimi gibi süreçleri gerçekleştirir.


İsteğe Bağlı RAG Katmanı

Belgeleri işler,

Parçalara böler ve gömer,

Sorguya en uygun içerikleri bulur,

Model çağrısına ek bağlam sağlar.



---

4. Temel Özellikler

4.1 Çoklu Model Desteği

Sistem destekler:

Yerel modeller (Ollama)

API tabanlı bulut modelleri

Gelecekte eklenecek herhangi bir model


Tüm modeller ortak bir konfigürasyon formatında tanımlanır.

4.2 Dinamik Model Yönlendirme

Backend şu işlemleri otomatik yapar:

Kullanıcının seçtiği modeli tanımlar,

Doğru model bağlayıcısını çalıştırır,

İstek yapısını modele uygun şekilde oluşturur,

Yanıtı tek bir standart formatta döndürür.


4.3 Gerçek Zamanlı İletişim

Sistem, gerçek zamanlı iletişim için:

WebSocket

Server-Sent-Events

Akışlı yanıt iletimi
gibi yöntemleri destekler.


4.4 RAG Yetenekleri

RAG sisteminde:

Belgeler yüklenir

Metinler gömülür

Arama yapılır

İlgili parçalar modele bağlam olarak verilir


Bu katman tamamen opsiyoneldir.

4.5 Hata Yönetimi ve Loglama

Tutarlı hata mesajları

API/donanım kaynaklı hataların özel işlenmesi

Terminalde yalnızca ERROR ve CRITICAL log seviyeleri



---

5. Kullanıcı Akışı

1. Kullanıcı arayüzü açar.


2. Bir model seçer.


3. Mesajını yazar ve gönderir.


4. Frontend, mesaj + model ID bilgisini backend’e yollar.


5. Backend:

Hedef modeli belirler

Bağlayıcıyı çalıştırır

(RAG aktifse) bağlam oluşturur

Yanıtı stream eder



6. Frontend, yanıtı gerçek zamanlı gösterir.




---

6. Sistem Bileşenleri

6.1 Frontend Bileşenleri

Sohbet penceresi

Mesaj balonları

Model seçici menü

Yükleme animasyonları

Akışlı metin göstergesi


6.2 Backend Bileşenleri

Yapay Zeka Router

Model bağlayıcıları

RAG modülü

Oturum / mesaj yönetimi

WebSocket yöneticisi


6.3 Konfigürasyon Katmanı

Şu bilgiler burada tutulur:

Yerel modeller

Bulut modeller

API anahtarları

RAG ayarları

Maksimum token sınırları



---

7. Model Yönetim Mantığı

Her model için:

İsim

Arayüzde gösterilir.

Sağlayıcı Tipi

Yerel

Bulut

Özel model


İstek Formatı

Her sağlayıcıya özel:

Header'lar

Token limitleri

Sıcaklık değerleri

Streaming desteği


Yanıt Çözümleyici

Her modelin yanıt formatı farklıdır, backend bunları tek formatta birleştirir.


---

8. RAG Tasarımı

1. Belge Yükleme


2. Parçalama (Chunking)


3. Gömme (Embedding)


4. Vektör Depolama


5. Sorgu-Tabanlı Arama


6. Bağlam Oluşturma


7. Model Çağrısı



Tüm aşamalar ayrı modüllerdir; isteğe göre değiştirilebilir.


---

9. Yerel Model İş Akışı (Ollama)

İstek localhost’a gönderilir

Model adı ve mesaj yollanır

Yanıt stream edilir

Tek biçime dönüştürülerek frontend’e iletilir


Avantajları:

API anahtarı gerekmez

Offline çalışır

Gizlilik maksimumdur



---

10. Bulut Model İş Akışı

API anahtarları backend’de saklanır

Sağlayıcıya uygun istek yapısı oluşturulur

Tek biçimde frontend’e gönderilir


Desteklenen türler:

Together.ai

OpenRouter

Ücretsiz açık kaynak model sağlayıcıları



---

11. Genişletilebilirlik

Yeni Model Ekleme

1. Konfigürasyona model bilgisi eklenir


2. Bağlayıcı uygulanır (veya hazır bir tanesi kullanılır)


3. Router’a kayıt edilir



RAG Ekleme

Tek tıkla aktif/pasif edilebilir.

Frontend Geliştirmeleri

Hazır prompt butonları

Prompt kütüphanesi

Sohbet dışa aktarma

Dosya yükleme desteği

Kullanıcı hafızası sistemi



---

12. Güvenlik

JWT kullanılmasa bile sistem:

API anahtarlarını frontend’e asla sızdırmaz

Backend tarafında hız limitleme ve doğrulama yapar

Log seviyesini minimumda tutar



---

13. Performans

Performans için:

Lazy-load RAG

Streaming yanıtlar

Zaman aşımı kontrolleri

Hafif vektör indeksleme



---

14. Gelecekteki Geliştirmeler

1. Ajan benzeri karar verme


2. Araç kullanımı (tool calling)


3. Kullanıcı hafızası


4. Çoklu sohbet sekmeleri


5. Dosya tabanlı gelişmiş RAG


6. Çok modlu (image/document) giriş desteği


7. Profil tabanlı model önerileri


8. Uzun dönemli oturum kaydı




---

15. Özet

Bu sistem; profesyonel, modüler, genişletilebilir, yerel ve bulut modelleri aynı arayüzde çalıştırabilen dinamik bir çoklu-model chatbot mimarisidir.
Gerçek zamanlı iletişim, RAG desteği, kolay model ekleme, temiz yapı ve uzun vadeli büyümeye uygun tasarım ilkeleri üzerine kurulmuştur.

Üstteki dokümantasyon, bir AI modelinin veya geliştiricinin projeyi tam anlamıyla kavrayabilmesi için gereken bütün yapısal ve işlevsel açıklamaları içerir.


---

İstersen bunu PDF formatı, sunum formatı, daha sade versiyon, daha ileri seviye teknik versiyon, veya roadmap hâline getirebilirim.

proje ile ilgili en önemli kısım şu api anahtarımız gemini flash 
gemini 3 pro 
ve huggingfaceten mistral 7b modeli için olacak yani api anahtarı ile bu 3 modeli kullanacağız 
ollama ile de qwen kullanacağız 
mesaj geçmişini de yeni bir depolama dosyası açıp oraya kaydetsin bu arayüzde solda geçmiş mesajlar bölümü olsun 
her yeni konuşma için yeni dosya açılsın projemde ve konuşma geçmişini oraya kaydetsin 
canım hangi mesaj geçmişinden devam etmek isterse ona geçebileyim 
proje klasörümden bir dosyaya kayıt edecek yani 
ve web socket kullanarak gelen mesajların anlık gözükmesini sağlayacağız 
henüz yarısı geldiyse o yarısı gözükecek

(uzun kod severim söylemediğim ekstra özellikler severim)