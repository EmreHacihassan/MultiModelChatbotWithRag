import { useEffect, useMemo, useRef, useState } from "react";

// naive token estimator: ~4 chars = 1 token (rough)
function estimateTokens(txt) {
  const t = Math.ceil((txt || "").length / 4);
  return t;
}

// Toggle button component
function ToggleButton({ active, onClick, icon, label, color, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        background: active ? `${color}22` : "#2a2f37",
        border: active ? `2px solid ${color}` : "1px solid #3a3f47",
        borderRadius: 8,
        padding: "6px 12px",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        gap: 6,
        color: active ? color : "#9aa0a6",
        fontSize: 12,
        fontWeight: active ? "bold" : "normal",
        transition: "all 0.2s ease",
      }}
    >
      <span style={{ fontSize: 14 }}>{icon}</span>
      <span>{label}</span>
    </button>
  );
}

export default function InputBar({ onSend, disabled, onStop, useAgent, setUseAgent, useRag, setUseRag }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [maxTokens] = useState(4096); // can be dynamic per model
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const tokenCount = useMemo(() => estimateTokens(text), [text]);
  const nearLimit = tokenCount > maxTokens * 0.9;

  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend?.(t, files, { useAgent, useRag });
    setText("");
    setFiles([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = "60px";
    }
  };
  
  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...selectedFiles]);
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };
  
  const removeFile = (idx) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  useEffect(() => {
    const handler = (e) => {
      if (disabled) return;
      const isCtrlEnter = (e.ctrlKey || e.metaKey) && e.key === "Enter";
      if (isCtrlEnter) {
        e.preventDefault();
        submit();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [disabled, text, files, useAgent, useRag]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 8,
        padding: 12,
        borderTop: "1px solid #2a2f37",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {/* Mode toggles and file button */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <ToggleButton
            active={useAgent}
            onClick={() => {
              setUseAgent?.(!useAgent);
              if (!useAgent) setUseRag?.(false); // Agent ve RAG birlikte kullanılamaz
            }}
            icon="🤖"
            label="Agent"
            color="#9b59b6"
            title="AI Agent modu: Araçlar kullanarak karmaşık soruları çözer"
          />
          <ToggleButton
            active={useRag}
            onClick={() => {
              setUseRag?.(!useRag);
              if (!useRag) setUseAgent?.(false); // Agent ve RAG birlikte kullanılamaz
            }}
            icon="📚"
            label="RAG"
            color="#3498db"
            title="RAG modu: Yüklenen dökümanlardan bilgi alarak yanıt verir"
          />
          
          <div style={{ borderLeft: "1px solid #3a3f47", height: 24, margin: "0 4px" }} />
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.pdf,.docx,.md,.json"
            style={{ display: "none" }}
            onChange={handleFileSelect}
          />
          <button
            title="Dosya ekle (RAG için)"
            onClick={() => fileInputRef.current?.click()}
            style={{ 
              background: "#2a2f37", 
              border: "1px solid #3a3f47",
              padding: "6px 12px",
              borderRadius: 8,
              fontSize: 12,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: "#9aa0a6",
            }}
          >
            <span>📎</span>
            <span>Dosya</span>
          </button>
          
          <div style={{ marginLeft: "auto", fontSize: 11, color: nearLimit ? "#e66d6d" : "#666" }}>
            {tokenCount} / {maxTokens} token
          </div>
        </div>
        
        {/* File list */}
        {files.length > 0 && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {files.map((f, idx) => (
              <span
                key={idx}
                style={{
                  fontSize: 11,
                  background: "#1a1d22",
                  border: "1px solid #2a2f37",
                  borderRadius: 8,
                  padding: "4px 8px",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                📄 {f.name} ({(f.size / 1024).toFixed(1)}KB)
                <button
                  onClick={() => removeFile(idx)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#e66d6d",
                    cursor: "pointer",
                    padding: 0,
                    fontSize: 12,
                  }}
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
        
        {/* Mode info banner */}
        {(useAgent || useRag) && (
          <div style={{
            background: useAgent ? "#2a1a3a" : "#1a2a3a",
            border: useAgent ? "1px solid #5a3a7a" : "1px solid #3a5a7a",
            borderRadius: 8,
            padding: "6px 12px",
            fontSize: 11,
            color: useAgent ? "#c9a0dc" : "#7fb3d5",
          }}>
            {useAgent ? (
              <>
                <strong>🤖 Agent Modu:</strong> AI hesap makinesi, web arama, kod çalıştırma gibi araçları kullanarak karmaşık soruları adım adım çözecek.
              </>
            ) : (
              <>
                <strong>📚 RAG Modu:</strong> Yüklediğiniz dökümanlardan ilgili bilgileri bulup yanıtını bu kaynaklara dayandıracak.
              </>
            )}
          </div>
        )}
        
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={2}
          placeholder={
            useAgent 
              ? "Agent'a sorunuzu yazın... (örn: 'Python ile fibonacci hesapla')" 
              : useRag 
                ? "Dökümanlar hakkında sorunuzu yazın..."
                : "Mesajınızı yazın… (Ctrl/Cmd + Enter ile gönderin)"
          }
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            const el = textareaRef.current;
            if (!el) return;
            el.style.height = "auto";
            const next = Math.min(Math.max(el.scrollHeight, 60), 120);
            el.style.height = next + "px";
          }}
          disabled={disabled}
          style={{
            flex: 1,
            resize: "none",
            height: "60px",
            maxHeight: "120px",
            overflowY: "auto",
          }}
        />
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <button onClick={submit} disabled={disabled}>
          Gönder
        </button>
        <button
          onClick={() => onStop?.()}
          disabled={!disabled}
          style={{ background: "#e66d6d" }}
          title="Akışı durdur"
        >
          Durdur
        </button>
      </div>
    </div>
  );
}