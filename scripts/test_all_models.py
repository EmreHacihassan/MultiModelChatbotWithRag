import asyncio
import sys
import os
import time

# Proje ana dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.adapters import huggingface as hf
from backend.adapters import gemini as gem

async def test_single_model(adapter, model_id, model_name, use_fallback=True):
    print(f"⏳ Deneniyor: {model_name} ({model_id})...", end='', flush=True)
    start = time.time()
    try:
        # Adapter'a göre argümanları ayarla
        kwargs = {}
        if adapter == hf:
            kwargs['fallback'] = use_fallback

        # Çok kısa bir timeout ile dene (15 saniye)
        response = await asyncio.wait_for(
            adapter.generate([{'role': 'user', 'content': 'Merhaba'}], model_id, **kwargs),
            timeout=15
        )
        elapsed = time.time() - start
        
        if response and not response.startswith('[Hata]'):
            print(f"\r✅ BAŞARILI: {model_name} ({elapsed:.1f}s)")
            return True, response[:50].replace('\n', ' ') + "..."
        else:
            print(f"\r❌ BAŞARISIZ: {model_name} - {response[:50]}...")
            return False, response
            
    except asyncio.TimeoutError:
        print(f"\r❌ ZAMAN AŞIMI: {model_name} (15s)")
        return False, "Timeout"
    except Exception as e:
        print(f"\r❌ HATA: {model_name} - {str(e)}")
        return False, str(e)

async def main():
    print("="*60)
    print("🤖 TÜM MODELLER TEST EDİLİYOR")
    print("="*60)
    
    results = []
    
    # 1. HuggingFace Modelleri
    print("\n--- HuggingFace Modelleri ---")
    for model_id, config in hf.MODELS.items():
        # HuggingFace için fallback'i KAPAT (False)
        success, msg = await test_single_model(hf, model_id, config['name'], use_fallback=False)
        results.append({'name': config['name'], 'success': success, 'msg': msg})
        time.sleep(2) # HF API'yi boğmamak için kısa bekleme

    # 2. Gemini Modelleri
    print("\n--- Gemini Modelleri ---")
    for model in gem.AVAILABLE_MODELS:
        # Gemini için varsayılan davranışı kullan
        success, msg = await test_single_model(gem, model['id'], model['name'], use_fallback=True)
        results.append({'name': model['name'], 'success': success, 'msg': msg})
        time.sleep(10) # Gemini Rate Limit'e takılmamak için 10 saniye bekle

    # Özet
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"Toplam Model: {total_count}")
    print(f"Çalışan:      {success_count}")
    print(f"Çalışmayan:   {total_count - success_count}")
    print("-" * 60)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['name']:<25} -> {r['msg']}")

if __name__ == "__main__":
    # Windows için event loop policy ayarı
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())