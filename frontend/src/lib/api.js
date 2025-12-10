import axios from 'axios';

// =============================================================================
// CONFIGURATION
// =============================================================================
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_API_WS_BASE ?? 'ws://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// =============================================================================
// UTILITY
// =============================================================================

function normalizeWS(url) {
  if (url.startsWith('http://')) return url.replace('http://', 'ws://');
  if (url.startsWith('https://')) return url.replace('https://', 'wss://');
  return url;
}

// =============================================================================
// MODELS API
// =============================================================================

export async function fetchModels() {
  const res = await api.get('/models');
  return res.data;
}

export { fetchModels as listModels };

// =============================================================================
// SESSIONS API
// =============================================================================

export async function listSessions() {
  const res = await api.get('/sessions');
  return res.data;
}

export async function createSession(title = 'Yeni Sohbet') {
  const res = await api.post('/sessions/create', { title });
  return res.data;
}

export async function getSession(id) {
  const res = await api.get(`/sessions/${id}`);
  return res.data;
}

export async function renameSession(id, title) {
  const res = await api.post(`/sessions/${id}/rename`, { title });
  return res.data;
}

export async function deleteSession(id) {
  const res = await api.post(`/sessions/${id}/delete`);
  return res.data;
}

export async function appendMessage(id, msg) {
  const res = await api.post(`/sessions/${id}/append`, msg);
  return res.data;
}

// =============================================================================
// CHAT - NON-STREAMING
// =============================================================================

export async function chatOnce(payload) {
  const res = await api.post('/chat', payload);
  return res.data?.text ?? '';
}

// =============================================================================
// CHAT - STREAMING (SSE + WebSocket + HTTP fallback)
// =============================================================================

export function streamChat({ 
  modelId, 
  messages, 
  onDelta, 
  onDone, 
  onError, 
  sessionId,
  // Agent callbacks
  useAgent = false,
  onThought,
  onToolCall,
  onToolResult,
  // RAG callbacks
  useRag = false,
  ragQuery,
  onRagContext,
}) {
  let stopped = false;
  let ws = null;
  let abortController = new AbortController();
  let accumulatedText = '';

  const stop = () => {
    stopped = true;
    abortController.abort();
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('__STOP__');
        ws.close(1000, 'client stop');
      }
    } catch {}
  };

  const handleDelta = (delta) => {
    if (stopped || !delta) return;
    accumulatedText += delta;
    onDelta?.(delta);
  };

  const handleDone = () => {
    if (stopped) return;
    onDone?.(accumulatedText);
  };

  const handleError = (err) => {
    if (stopped) return;
    onError?.(typeof err === 'string' ? err : err?.detail || err?.error || 'Hata');
  };

  // =========================================================================
  // TRANSPORT 1: WebSocket (with Agent & RAG support)
  // =========================================================================
  const tryWebSocket = () =>
    new Promise((resolve) => {
      if (stopped) return resolve(false);

      try {
        const wsUrl = `${normalizeWS(WS_BASE)}/ws/chat`;
        ws = new WebSocket(wsUrl);

        const timeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close();
            resolve(false);
          }
        }, 5000);

        ws.onopen = () => {
          clearTimeout(timeout);
          // Send payload with Agent/RAG options
          ws.send(JSON.stringify({ 
            modelId, 
            messages, 
            sessionId,
            useAgent,
            useRag,
            ragQuery,
          }));
        };

        ws.onmessage = (evt) => {
          if (stopped) return;
          try {
            const data = JSON.parse(evt.data);
            
            // Ping/pong
            if (data.type === 'ping' || data.type === 'pong' || data.type === 'connected') return;
            
            // Agent events
            if (data.type === 'thought' && onThought) {
              onThought(data.content);
              return;
            }
            if (data.type === 'tool_call' && onToolCall) {
              onToolCall(data.tool, data.input);
              return;
            }
            if (data.type === 'tool_result' && onToolResult) {
              onToolResult(data.tool, data.result);
              return;
            }
            
            // RAG events
            if (data.type === 'rag_context' && onRagContext) {
              onRagContext(data.docs);
              return;
            }
            
            // Standard events
            if (data.delta) handleDelta(data.delta);
            else if (data.done) { handleDone(); resolve(true); }
            else if (data.stopped) { handleDone(); resolve(true); }
            else if (data.error) { handleError(data); resolve(true); }
          } catch {}
        };

        ws.onclose = (evt) => {
          clearTimeout(timeout);
          if (!stopped && evt.code !== 1000) resolve(false);
          else if (!stopped) { handleDone(); resolve(true); }
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          resolve(false);
        };

      } catch {
        resolve(false);
      }
    });

  // =========================================================================
  // TRANSPORT 2: SSE (fetch + ReadableStream) - ANLIK
  // =========================================================================
  const trySSE = () =>
    new Promise(async (resolve) => {
      if (stopped) return resolve(false);

      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ modelId, messages, sessionId }),
          signal: abortController.signal,
        });

        if (!res.ok) {
          resolve(false);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done || stopped) break;

          // Decode chunk
          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events - ANLIK
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Son eksik satırı buffer'da tut

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data:')) continue;

            const jsonStr = trimmed.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const data = JSON.parse(jsonStr);

              if (data.delta) {
                // ANLIK delta işle
                handleDelta(data.delta);
              } else if (data.done) {
                handleDone();
              } else if (data.error) {
                handleError(data.error);
              }
              // keepalive ignore
            } catch {}
          }
        }

        if (!stopped) handleDone();
        resolve(true);

      } catch (e) {
        if (e.name === 'AbortError') {
          resolve(true);
        } else {
          resolve(false);
        }
      }
    });

  // =========================================================================
  // TRANSPORT 3: HTTP Fallback
  // =========================================================================
  const tryHTTP = () =>
    new Promise(async (resolve) => {
      if (stopped) return resolve(false);

      try {
        const res = await api.post('/chat', { modelId, messages }, {
          signal: abortController.signal,
        });

        const text = res.data?.text ?? '';
        if (text) handleDelta(text);
        handleDone();
        resolve(true);

      } catch (e) {
        if (e.name !== 'AbortError' && e.name !== 'CanceledError') {
          handleError(e.message);
        }
        resolve(false);
      }
    });

  // =========================================================================
  // EXECUTE
  // =========================================================================
  (async () => {
    // Try WebSocket first
    const wsOk = await tryWebSocket();
    if (wsOk || stopped) return;

    accumulatedText = '';

    // Try SSE
    const sseOk = await trySSE();
    if (sseOk || stopped) return;

    accumulatedText = '';

    // HTTP fallback
    await tryHTTP();
  })();

  return { stop };
}

// =============================================================================
// HEALTH
// =============================================================================

export async function healthCheck() {
  try {
    const res = await api.get('/health/', { timeout: 5000 });
    return { ok: true, ...res.data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export default {
  fetchModels,
  listModels: fetchModels,
  listSessions,
  createSession,
  getSession,
  renameSession,
  deleteSession,
  appendMessage,
  chatOnce,
  streamChat,
  healthCheck,
};