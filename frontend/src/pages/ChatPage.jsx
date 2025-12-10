import { useEffect, useRef, useState } from "react";
import ModelSelector from "../components/ModelSelector.jsx";
import MessageList from "../components/MessageList.jsx";
import InputBar from "../components/InputBar.jsx";
import {
  listModels,
  listSessions,
  createSession,
  getSession,
  appendMessage,
  renameSession,
  deleteSession,
  streamChat,
} from "../lib/api.js";

// =============================================================================
// FALLBACK MODEL LİSTESİ - Backend ulaşılamazsa kullanılır
// 2025 - HUGGINGFACE INFERENCE PROVIDERS (Together & Hyperbolic)
// =============================================================================
const FALLBACK_MODELS = [
  // =========================================================================
  // GEMINI (Google AI)
  // =========================================================================
  { id: "gemini-3-pro", name: "Gemini 3 Pro", provider: "gemini", streaming: true, description: "En güçlü Gemini modeli (2025)" },
  { id: "gemini-flash", name: "Gemini 2.5 Flash", provider: "gemini", streaming: true, description: "Hızlı ve güçlü" },
  { id: "gemini-pro", name: "Gemini 1.5 Pro", provider: "gemini", streaming: true, description: "Detaylı yanıtlar" },
  
  // =========================================================================
  // HUGGINGFACE INFERENCE PROVIDERS (2025 - Together & Hyperbolic)
  // router.huggingface.co/{provider}/v1/chat/completions
  // =========================================================================
  
  // Together Provider - Tier 1
  { id: "hf-llama-3.2-3b", name: "Llama 3.2 3B", provider: "hf", streaming: true, description: "Meta - Hızlı ve yetenekli", tier: 1 },
  { id: "hf-llama-3.1-8b", name: "Llama 3.1 8B", provider: "hf", streaming: true, description: "Meta - Güçlü ve dengeli", tier: 1 },
  { id: "hf-llama-3.1-70b", name: "Llama 3.1 70B", provider: "hf", streaming: true, description: "Meta - En güçlü açık kaynak", tier: 1 },
  { id: "hf-qwen-2.5-72b", name: "Qwen 2.5 72B", provider: "hf", streaming: true, description: "Alibaba - En güçlü Qwen", tier: 1 },
  
  // Together Provider - Tier 2
  { id: "hf-qwen-2.5-7b", name: "Qwen 2.5 7B", provider: "hf", streaming: true, description: "Alibaba - Çok dilli", tier: 2 },
  
  // Hyperbolic Provider
  { id: "hf-llama-3.2-3b-hyp", name: "Llama 3.2 3B (Hyp)", provider: "hf", streaming: true, description: "Meta - Hyperbolic üzerinde", tier: 2 },
  { id: "hf-qwen-2.5-72b-hyp", name: "Qwen 2.5 72B (Hyp)", provider: "hf", streaming: true, description: "Alibaba - Hyperbolic üzerinde", tier: 1 },
  
  // =========================================================================
  // OLLAMA (Yerel Modeller)
  // =========================================================================
  { id: "ollama:qwen2.5", name: "Qwen 2.5 (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Alibaba Qwen" },
  { id: "ollama:llama3.1", name: "Llama 3.1 (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Meta Llama" },
  { id: "ollama:mistral", name: "Mistral (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Mistral AI" },
  { id: "ollama:phi3", name: "Phi-3 (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Microsoft Phi" },
  { id: "ollama:gemma2", name: "Gemma 2 (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Google Gemma" },
  { id: "ollama:codellama", name: "CodeLlama (Ollama)", provider: "ollama", streaming: true, description: "Yerel - Kod yazımı" },
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function ChatPage() {
  // Sidebar: sessions
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);

  // Chat - Varsayılan model Gemini 3 Pro
  const [modelId, setModelId] = useState("gemini-3-pro");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const stopRef = useRef(null);

  // Bonus ayar: web araması anahtarı
  const [webSearch, setWebSearch] = useState(false);

  // Models - Backend'den dinamik alınacak
  const [modelList, setModelList] = useState(FALLBACK_MODELS);
  const [modelsLoaded, setModelsLoaded] = useState(false);

  // =========================================================================
  // INITIAL LOAD
  // =========================================================================
  
  useEffect(() => {
    (async () => {
      // Model listesini backend'den al
      try {
        const backendModels = await listModels();
        if (Array.isArray(backendModels) && backendModels.length > 0) {
          // Backend'den gelen modelleri normalize et
          const normalized = backendModels.map(m => ({
            id: m.id,
            name: m.name,
            provider: m.provider,
            streaming: m.streaming !== false,
            description: m.description || m.desc || '',
            desc: m.description || m.desc || '',
            tier: m.tier || null,
            context_window: m.context_window || null,
          }));
          setModelList(normalized);
          setModelsLoaded(true);
          console.log(`✓ ${normalized.length} model backend'den yüklendi`);
        }
      } catch (err) {
        console.warn("Backend model listesi alınamadı, fallback kullanılıyor:", err);
        setModelsLoaded(true); // Fallback ile devam et
      }
      
      // Session'ları yükle
      try {
        const s = await listSessions();
        setSessions(s || []);
        if (s?.length) {
          await selectSession(s[0]);
        } else {
          const created = await createSession("İlk Sohbet");
          setSessions([created]);
          await selectSession(created);
        }
      } catch (err) {
        console.error("Session yükleme hatası:", err);
      }
    })();
  }, []);

  // =========================================================================
  // MESSAGE HELPERS
  // =========================================================================
  
  const append = (msg) => setMessages((arr) => [...arr, msg]);
  
  const updateLastAssistant = (delta) =>
    setMessages((arr) => {
      const idx = [...arr].reverse().findIndex((x) => x.role === "assistant");
      if (idx === -1) return [...arr, { role: "assistant", content: delta }];
      const ri = arr.length - 1 - idx;
      const updated = [...arr];
      updated[ri] = { ...updated[ri], content: (updated[ri].content || "") + delta };
      return updated;
    });

  // =========================================================================
  // SESSION MANAGEMENT
  // =========================================================================
  
  const selectSession = async (sessionItem) => {
    setActiveSession({ id: sessionItem.id, title: sessionItem.title });
    try {
      const full = await getSession(sessionItem.id);
      setMessages(full?.messages || []);
    } catch {
      setMessages([]);
    }
  };

  const refreshSessions = async () => {
    try {
      const s = await listSessions();
      setSessions(s || []);
    } catch {}
  };

  const newSession = async () => {
    const created = await createSession("Yeni Sohbet");
    setSessions((s) => [created, ...s]);
    await selectSession(created);
  };

  const renameActive = async () => {
    if (!activeSession?.id) return;
    const val = prompt("Yeni başlık:", activeSession.title || "Sohbet");
    if (!val) return;
    await renameSession(activeSession.id, val);
    setActiveSession((as) => ({ ...as, title: val }));
    await refreshSessions();
  };

  const deleteActive = async () => {
    if (!activeSession?.id) return;
    const ok = confirm(`'${activeSession.title}' oturumunu silmek istiyor musunuz?`);
    if (!ok) return;
    await deleteSession(activeSession.id);
    await refreshSessions();
    const s = await listSessions();
    if (s?.length) await selectSession(s[0]);
    else await newSession();
  };

  const handleEditMessage = async (index, newText) => {
    setMessages((arr) => {
      const updated = [...arr];
      updated[index] = { ...updated[index], content: newText };
      return updated;
    });
  };

  // =========================================================================
  // CHAT HANDLERS
  // =========================================================================
  
  const handleStop = () => {
    try { stopRef.current?.(); } catch {}
    setStreaming(false);
    setLoading(false);
  };

  const handleSend = async (text) => {
    if (!activeSession?.id) return;
    
    // Model bilgisini al
    const currentModel = modelList.find(m => m.id === modelId);
    const modelName = currentModel?.name || modelId;
    
    // User mesajı ekle
    const userMsg = { role: "user", content: text, modelId };
    append(userMsg);
    await appendMessage(activeSession.id, userMsg);

    // Loading state
    setLoading(true);
    setStreaming(true);
    
    // Boş assistant mesajı ekle (modelName ile)
    append({ role: "assistant", content: "", modelId, modelName });

    try {
      const stopper = await streamChat({
        sessionId: activeSession.id,
        modelId,
        messages: [...messages, userMsg],
        onDelta: (delta) => updateLastAssistant(delta),
        onDone: async (finalText) => {
          setStreaming(false);
          setLoading(false);
          // modelName dahil kaydet
          await appendMessage(activeSession.id, { 
            role: "assistant", 
            content: finalText ?? "", 
            modelId, 
            modelName 
          });
          await refreshSessions();
        },
        onError: (errMsg) => {
          setStreaming(false);
          setLoading(false);
          updateLastAssistant(`\n[Hata] ${errMsg || "Akış başarısız"}\n`);
        },
      });
      stopRef.current = stopper;
    } catch (err) {
      setStreaming(false);
      setLoading(false);
      updateLastAssistant(`\n[Hata] ${err?.message || "Akış başlatılamadı"}\n`);
    }
  };

  const clearChat = () => setMessages([]);

  // =========================================================================
  // UI COMPONENTS
  // =========================================================================
  
  const Sidebar = () => (
    <aside style={{ 
      width: 280, 
      padding: 12, 
      borderRight: "1px solid #2a2f37",
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      boxSizing: "border-box",
      background: "#0d0f12",
    }}>
      {/* Header */}
      <div style={{ 
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        marginBottom: 12, 
        flexShrink: 0,
        paddingBottom: 12,
        borderBottom: "1px solid #2a2f37",
      }}>
        <strong style={{ color: "#e6e6e6", fontSize: 14 }}>💬 Sohbetler</strong>
        <div style={{ display: "flex", gap: 6 }}>
          <button 
            onClick={newSession} 
            disabled={loading || streaming} 
            style={{ 
              padding: "6px 12px", 
              background: "#4f8cff", 
              border: "none", 
              borderRadius: 6,
              color: "#fff",
              cursor: loading || streaming ? "not-allowed" : "pointer",
              fontSize: 12,
            }}
          >
            + Yeni
          </button>
        </div>
      </div>
      
      {/* Session List */}
      <div style={{ 
        display: "flex", 
        flexDirection: "column", 
        gap: 6, 
        flex: 1,
        overflowY: "auto",
        minHeight: 0,
      }}>
        {sessions.map((s) => (
          <div
            key={s.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <button
              onClick={() => selectSession(s)}
              style={{
                flex: 1,
                textAlign: "left",
                background: activeSession?.id === s.id ? "#253046" : "#1a1d22",
                border: activeSession?.id === s.id ? "1px solid #4f8cff" : "1px solid #2a2f37",
                borderRadius: 8,
                padding: "10px 12px",
                color: "#e6e6e6",
                cursor: "pointer",
                fontSize: 13,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={s.title || "Sohbet"}
            >
              {s.title || "Sohbet"}
            </button>
            {activeSession?.id === s.id && (
              <div style={{ display: "flex", gap: 4 }}>
                <button 
                  onClick={renameActive} 
                  disabled={loading || streaming}
                  style={{ 
                    padding: "6px", 
                    background: "#2a2f37", 
                    border: "1px solid #3a3f47",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                  title="Yeniden adlandır"
                >
                  ✏️
                </button>
                <button 
                  onClick={deleteActive} 
                  disabled={loading || streaming}
                  style={{ 
                    padding: "6px", 
                    background: "#5c2a2a", 
                    border: "1px solid #7a3a3a",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                  title="Sil"
                >
                  🗑️
                </button>
              </div>
            )}
          </div>
        ))}
        {sessions.length === 0 && (
          <div style={{ fontSize: 12, color: "#9aa0a6", textAlign: "center", padding: 20 }}>
            Henüz oturum yok.
            <br />
            <button onClick={newSession} style={{ marginTop: 10, padding: "6px 12px" }}>
              İlk sohbeti başlat
            </button>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div style={{ 
        paddingTop: 12, 
        borderTop: "1px solid #2a2f37", 
        fontSize: 11, 
        color: "#666",
        flexShrink: 0,
      }}>
        {modelsLoaded ? `${modelList.length} model aktif` : "Modeller yükleniyor..."}
      </div>
    </aside>
  );

  const Header = () => (
    <header style={{ 
      padding: 12, 
      borderBottom: "1px solid #2a2f37", 
      flexShrink: 0,
      background: "#0d0f12",
    }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <ModelSelector models={modelList} value={modelId} onChange={setModelId} />
          <div style={{ fontSize: 12, color: "#9aa0a6" }}>
            {activeSession?.title || "-"} 
            {streaming && <span style={{ color: "#5ac98f", marginLeft: 6 }}>● Streaming</span>}
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#666" }}>
          Ctrl+Enter ile gönder
        </div>
      </div>
    </header>
  );

  const ToolsDock = () => (
    <section 
      className="tools-dock" 
      aria-label="Araç Çubuğu" 
      style={{ 
        flexShrink: 0,
        padding: "8px 0",
        borderTop: "1px solid #2a2f37",
      }}
    >
      <div className="tool-group" style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        <button 
          className="tool-pill" 
          title="Dosya ekle (yakında)"
          style={{
            padding: "6px 12px",
            background: "#1a1d22",
            border: "1px solid #2a2f37",
            borderRadius: 20,
            color: "#9aa0a6",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          📎 Ekle
        </button>
        <button
          className={`tool-pill ${webSearch ? "primary" : ""}`}
          title="Web aramasını aç/kapat"
          onClick={() => setWebSearch((v) => !v)}
          style={{
            padding: "6px 12px",
            background: webSearch ? "#253046" : "#1a1d22",
            border: webSearch ? "1px solid #4f8cff" : "1px solid #2a2f37",
            borderRadius: 20,
            color: webSearch ? "#fff" : "#9aa0a6",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          🌐 {webSearch ? "Arama Açık" : "Arama Kapalı"}
        </button>
        <button 
          className="tool-pill" 
          onClick={clearChat} 
          title="Mesajları temizle"
          style={{
            padding: "6px 12px",
            background: "#1a1d22",
            border: "1px solid #2a2f37",
            borderRadius: 20,
            color: "#9aa0a6",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          🧹 Temizle
        </button>
      </div>
    </section>
  );

  // =========================================================================
  // RENDER
  // =========================================================================

  return (
    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", height: "100vh", overflow: "hidden" }}>
      <Sidebar />
      <div
        style={{ 
          display: "flex",
          flexDirection: "column",
          height: "100vh",
          overflow: "hidden",
          padding: "0 12px",
          boxSizing: "border-box",
          background: "#0f1114",
        }}
      >
        <Header />
        <main 
          className="panel" 
          style={{ 
            flex: 1, 
            minHeight: 0, 
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            margin: "8px 0",
          }}
        >
          <MessageList messages={messages} onEditMessage={handleEditMessage} />
        </main>
        <section aria-label="Mesaj Girişi" style={{ flexShrink: 0 }}>
          <InputBar onSend={handleSend} onStop={handleStop} disabled={loading || streaming} />
        </section>
        <ToolsDock />
        <div className="safe-bottom" aria-hidden="true" style={{ flexShrink: 0, height: 8 }} />
      </div>
    </div>
  );
}