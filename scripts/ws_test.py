import asyncio
import json
import websockets

WS_URL = "ws://localhost:8000/ws/chat"

async def main():
    print(f"🔌 Bağlanıyor: {WS_URL}")
    
    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Bağlandı!")
            
            # İlk mesajı al (connected)
            msg = await ws.recv()
            print(f"📩 {msg}")
            
            # ✅ GÜNCEL MODEL: gemini-flash
            payload = {
                "modelId": "gemini-flash",  # ✅ Gemini 2.5 Flash
                "messages": [
                    {"role": "user", "content": "Merhaba! Kısaca kendini tanıt."}
                ],
            }
            
            print(f"\n📤 Gönderiliyor: {payload['messages'][0]['content']}")
            print(f"📦 Model: {payload['modelId']}")
            await ws.send(json.dumps(payload))
            
            # Streaming yanıtı al
            print("\n📥 Yanıt (streaming):")
            print("-" * 50)
            
            full_response = ""
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                # Ping atla
                if data.get('type') == 'ping':
                    continue
                
                # Delta (token)
                if data.get('delta'):
                    print(data['delta'], end='', flush=True)
                    full_response += data['delta']
                
                # Tamamlandı
                elif data.get('done'):
                    print("\n" + "-" * 50)
                    print(f"✅ Tamamlandı!")
                    print(f"📊 İstatistikler: {data.get('stats')}")
                    break
                
                # Hata
                elif data.get('error'):
                    print(f"\n❌ Hata: {data}")
                    break
                
                # Durduruldu
                elif data.get('stopped'):
                    print(f"\n⏹️ Durduruldu")
                    break
            
            print(f"\n📝 Toplam karakter: {len(full_response)}")
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Bağlantı kapandı: {e}")
    except ConnectionRefusedError:
        print("❌ Bağlantı reddedildi! Backend çalışıyor mu?")
        print("   Başlatmak için: python -m uvicorn backend.app.server.asgi:application --reload")
    except Exception as e:
        print(f"❌ Hata: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("  MyChatbot WebSocket Test")
    print("=" * 50)
    asyncio.run(main())