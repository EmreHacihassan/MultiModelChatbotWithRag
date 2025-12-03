import axios from "axios";
import { API_BASE, WS_BASE } from "../config/env";
import { createParser } from "eventsource-parser";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// Modeller: backend varsa onu kullan, yoksa statik liste
export async function listModels() {
  try {
    const { data } = await client.get("/models");
    if (Array.isArray(data) && data.length) return data;
  } catch {}
  return [
    { id: "gemini-flash", name: "Gemini Flash", provider: "gemini", streaming: true },
    { id: "gemini-3-pro", name: "Gemini 3 Pro", provider: "gemini", streaming: true },
    { id: "hf-mistral-7b", name: "HuggingFace · Mistral 7B", provider: "hf", streaming: true },
    { id: "ollama:qwen", name: "Ollama · Qwen", provider: "ollama", streaming: true },
  ];
}

// Oturumlar (dosya tabanlı kalıcılık beklenir)
export async function listSessions() {
  const { data } = await client.get("/sessions");
  return data;
}
export async function createSession(title = "Yeni Sohbet") {
  const { data } = await client.post("/sessions", { title });
  return data;
}
export async function getSession(sessionId) {
  const { data } = await client.get(`/sessions/${encodeURIComponent(sessionId)}`);
  return data;
}
export async function renameSession(sessionId, title) {
  const { data } = await client.patch(`/sessions/${encodeURIComponent(sessionId)}`, { title });
  return data;
}
export async function deleteSession(sessionId) {
  await client.delete(`/sessions/${encodeURIComponent(sessionId)}`);
}
export async function appendMessage(sessionId, message) {
  await client.post(`/sessions/${encodeURIComponent(sessionId)}/messages`, message);
}

// Non-stream fallback (tek seferlik yanıt)
export async function sendMessage({ sessionId, modelId, messages }) {
  const { data } = await client.post("/chat", { sessionId, modelId, messages });
  return data;
}

/**
 * Akışlı sohbet: 1) WebSocket, 2) SSE-POST, 3) HTTP fallback
 * Döndürdüğü değer: akışı durduran stopper fonksiyonu
 */
export async function streamChat({ sessionId, modelId, messages, onDelta, onDone, onError }) {
  let currentStopper = null;

  // Öncelik: WebSocket
  try {
    const wsUrl = normalizeWs(`${WS_BASE}/chat`);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: "chat", sessionId, modelId, messages }));
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "delta") onDelta?.(msg.delta || "");
        else if (msg.type === "done") {
          onDone?.(msg.final || "");
          try { socket.close(); } catch {}
        } else if (msg.type === "error") {
          onError?.(msg.error || "Unknown error");
          try { socket.close(); } catch {}
        }
      } catch {
        onDelta?.(evt.data || "");
      }
    };

    const startSse = () => {
      currentStopper = sseFallback({ sessionId, modelId, messages, onDelta, onDone, onError });
    };

    socket.onerror = () => {
      try { socket.close(); } catch {}
      startSse();
    };

    socket.onclose = (e) => {
      if (e.code !== 1000) startSse();
    };

    currentStopper = () => {
      try { socket.close(1000, "client-cancel"); } catch {}
    };

    return () => currentStopper?.();
  } catch (wsErr) {
    // eslint-disable-next-line no-console
    console.warn("WebSocket init failed, switching to SSE:", wsErr);
    return (currentStopper = sseFallback({ sessionId, modelId, messages, onDelta, onDone, onError }));
  }
}

// SSE fallback: POST ile gövde (messages) gönderilir, chunk'lar eventsource-parser ile çözülür
function sseFallback({ sessionId, modelId, messages, onDelta, onDone, onError }) {
  let cancelled = false;

  (async () => {
    try {
      const resp = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, modelId, messages }),
      });
      if (!resp.ok || !resp.body) throw new Error(`SSE HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      const parser = createParser((event) => {
        if (cancelled) return;
        if (event.type === "event" || event.type === "message") {
          try {
            const payload = JSON.parse(event.data);
            if (payload.delta) onDelta?.(payload.delta);
            if (payload.done) onDone?.(payload.final || "");
          } catch {
            onDelta?.(event.data || "");
          }
        }
      });

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        parser.feed(decoder.decode(value, { stream: true }));
      }

      if (!cancelled) onDone?.("");
    } catch (err) {
      if (!cancelled) {
        // eslint-disable-next-line no-console
        console.warn("SSE error, switching to HTTP:", err);
        await httpFallback({ sessionId, modelId, messages, onDelta, onDone, onError });
      }
    }
  })();

  return () => { cancelled = true; };
}

// WS adres normalleştirme: göreli yolları olduğu gibi bırak
function normalizeWs(url) {
  if (url.startsWith("/")) return url; // aynı origin proxy yolu
  if (url.startsWith("http://")) return "ws://" + url.substring("http://".length);
  if (url.startsWith("https://")) return "wss://" + url.substring("https://".length);
  if (url.startsWith("ws://") || url.startsWith("wss://")) return url;
  return url; // bilinmeyen şema → dokunma
}

// HTTP fallback: tek metin döner
async function httpFallback({ sessionId, modelId, messages, onDelta, onDone, onError }) {
  try {
    const resp = await sendMessage({ sessionId, modelId, messages });
    const text = resp?.text || "";
    if (text) onDelta?.(text);
    onDone?.(text);
  } catch (err) {
    onError?.(err?.message || "HTTP fallback error");
  }
}