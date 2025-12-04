import asyncio
import json
import websockets

# WS port matches Daphne
WS_URL = "ws://localhost:8000/ws/chat"

async def main():
    async with websockets.connect(WS_URL) as ws:
        payload = {
            "modelId": "hf-mistral-7b",
            "messages": [{"role": "user", "content": "Merhaba! Bu bir WS testi."}],
        }
        await ws.send(json.dumps(payload))
        async for msg in ws:
            print(msg)
            try:
                data = json.loads(msg)
                if data.get("done"):
                    break
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())