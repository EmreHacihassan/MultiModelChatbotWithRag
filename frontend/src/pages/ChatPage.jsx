import { useEffect, useMemo, useRef, useState } from "react";
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

export default function ChatPage() {
  // Sidebar: sessions
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null); // {id, title}
  // Chat
  const [modelId, setModelId] = useState("gemini-flash");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const stopRef = useRef(null);

  // Models
  const [modelList, setModelList] = useState([
    { id: "gemini-flash", name: "Gemini Flash", provider: "gemini", streaming: true },
    { id: "gemini-3-pro", name: "Gemini 3 Pro", provider: "gemini", streaming: true },
    { id: "hf-mistral-7b", name: "HuggingFace · Mistral 7B", provider: "hf", streaming: true },
    { id: "ollama:qwen", name: "Ollama · Qwen", provider: "ollama", streaming: true },
  ]);

  // Initial load: models + sessions
  useEffect(() => {
    (async () => {
      try {
        const m = await listModels();
        if (Array.isArray(m) && m.length) setModelList(m);
      } catch {}
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
      } catch {}
    })();
  }, []);

  // Helpers
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
    // pick another or create new
    const s = await listSessions();
    if (s?.length) await selectSession(s[0]);
    else await newSession();
  };

  const handleEditMessage = async (index, newText) => {
    // local edit only; can persist via appendMessage/patch if needed
    setMessages((arr) => {
      const updated = [...arr];
      updated[index] = { ...updated[index], content: newText };
      return updated;
    });
  };

  const handleStop = () => {
    try {
      stopRef.current?.();
    } catch {}
    setStreaming(false);
    setLoading(false);
  };

  const handleSend = async (text /*, files*/) => {
    if (!activeSession?.id) return;
    const userMsg = { role: "user", content: text, modelId };
    append(userMsg);
    await appendMessage(activeSession.id, userMsg);

    setLoading(true);
    setStreaming(true);
    append({ role: "assistant", content: "" });

    try {
      const stopper = await streamChat({
        sessionId: activeSession.id,
        modelId,
        messages: [...messages, userMsg],
        onDelta: (delta) => {
          updateLastAssistant(delta);
        },
        onDone: async (finalText) => {
          setStreaming(false);
          setLoading(false);
          await appendMessage(activeSession.id, {
            role: "assistant",
            content: finalText ?? "",
            modelId,
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

  // Layout components
  const Sidebar = () => (
    <aside style={{ width: 280, padding: 12, borderRight: "1px solid #2a2f37" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <strong>Sohbetler</strong>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={newSession} disabled={loading || streaming} style={{ padding: "6px 10px" }}>Yeni</button>
          <button onClick={renameActive} disabled={!activeSession || loading || streaming} style={{ padding: "6px 10px", background: "#2a2f37" }}>Adlandır</button>
          <button onClick={deleteActive} disabled={!activeSession || loading || streaming} style={{ padding: "6px 10px", background: "#e66d6d" }}>Sil</button>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: "calc(100vh - 150px)", overflow: "auto" }}>
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => selectSession(s)}
            style={{
              textAlign: "left",
              background: activeSession?.id === s.id ? "#253046" : "#1a1d22",
              border: "1px solid #2a2f37",
              borderRadius: 8,
              padding: "8px 10px",
              color: "#e6e6e6",
              cursor: "pointer",
            }}
            title={s.title || "Sohbet"}
          >
            {s.title || "Sohbet"}
          </button>
        ))}
        {sessions.length === 0 ? (
          <div style={{ fontSize: 12, color: "#9aa0a6" }}>Henüz oturum yok.</div>
        ) : null}
      </div>
    </aside>
  );

  const Header = () => (
    <header style={{ padding: 12, borderBottom: "1px solid #2a2f37" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <ModelSelector models={modelList} value={modelId} onChange={setModelId} />
          <div style={{ fontSize: 12, color: "#9aa0a6" }}>
            Oturum: {activeSession?.title || "-"} {streaming ? "· Akış" : ""}
          </div>
        </div>
        <div style={{ fontSize: 12, color: "#9aa0a6" }}>
          İpucu: Göndermek için Ctrl/Cmd+Enter
        </div>
      </div>
    </header>
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", height: "100vh" }}>
      <Sidebar />
      <div className="container">
        <Header />
        <main className="panel">
          <MessageList messages={messages} onEditMessage={handleEditMessage} />
        </main>
        <footer>
          <InputBar onSend={handleSend} onStop={handleStop} disabled={loading || streaming} />
        </footer>
      </div>
    </div>
  );
}