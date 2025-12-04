import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import hljs from "highlight.js/lib/common";
import "highlight.js/styles/github-dark.css";
import DOMPurify from "dompurify";

// =============================================================================
// MARKED CONFIGURATION
// =============================================================================

marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: (code, lang) => {
    try {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    } catch {
      return code;
    }
  },
});

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function RoleLabel({ role, modelName }) {
  const roleConfig = {
    user: { color: "#4f8cff", icon: "👤", label: "Sen" },
    assistant: { color: "#5ac98f", icon: "🤖", label: "Asistan" },
    system: { color: "#9aa0a6", icon: "⚙️", label: "Sistem" },
  };
  
  const config = roleConfig[role] || roleConfig.assistant;
  
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 14 }}>{config.icon}</span>
      <strong style={{ color: config.color, fontSize: 13 }}>{config.label}</strong>
      {role === "assistant" && modelName && (
        <span 
          style={{ 
            fontSize: 11, 
            color: "#9aa0a6", 
            fontWeight: "normal",
            background: "#1a1d22",
            padding: "2px 8px",
            borderRadius: 12,
            border: "1px solid #2a2f37",
          }}
          title={`Model: ${modelName}`}
        >
          {modelName}
        </span>
      )}
    </div>
  );
}

function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Kopyalama hatası:", err);
    }
  };
  
  return (
    <button
      onClick={handleCopy}
      style={{
        background: copied ? "#2d4a2d" : "#2a2f37",
        border: copied ? "1px solid #5ac98f" : "1px solid #3a3f47",
        borderRadius: 6,
        color: copied ? "#5ac98f" : "#e6e6e6",
        fontSize: 11,
        padding: "4px 10px",
        cursor: "pointer",
        transition: "all 0.15s ease",
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
      title="Panoya kopyala"
    >
      {copied ? "✓ Kopyalandı" : "📋 Kopyala"}
    </button>
  );
}

function EditableText({ content, onSave }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(content || "");
  const inputRef = useRef(null);
  
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);
  
  if (!editing) {
    return (
      <button
        onClick={() => setEditing(true)}
        style={{
          background: "#2a2f37",
          border: "1px solid #3a3f47",
          borderRadius: 6,
          color: "#e6e6e6",
          fontSize: 11,
          padding: "4px 10px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 4,
        }}
        title="Mesajı düzenle"
      >
        ✏️ Düzenle
      </button>
    );
  }
  
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flex: 1 }}>
      <input 
        ref={inputRef}
        value={val} 
        onChange={(e) => setVal(e.target.value)} 
        style={{ 
          flex: 1,
          padding: "6px 10px",
          background: "#1a1d22",
          border: "1px solid #4f8cff",
          borderRadius: 6,
          color: "#e6e6e6",
          fontSize: 12,
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            onSave?.(val);
            setEditing(false);
          } else if (e.key === "Escape") {
            setVal(content || "");
            setEditing(false);
          }
        }}
      />
      <button
        onClick={() => { onSave?.(val); setEditing(false); }}
        style={{ 
          background: "#4f8cff", 
          border: "none", 
          borderRadius: 6, 
          color: "#fff", 
          fontSize: 11, 
          padding: "6px 12px",
          cursor: "pointer",
        }}
      >
        ✓ Kaydet
      </button>
      <button
        onClick={() => { setVal(content || ""); setEditing(false); }}
        style={{ 
          background: "#2a2f37", 
          border: "1px solid #3a3f47", 
          borderRadius: 6, 
          color: "#e6e6e6", 
          fontSize: 11, 
          padding: "6px 12px",
          cursor: "pointer",
        }}
      >
        ✕ İptal
      </button>
    </div>
  );
}

function Timestamp({ timestamp }) {
  if (!timestamp) return null;
  
  const date = new Date(timestamp);
  const timeStr = date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
  
  return (
    <span style={{ fontSize: 10, color: "#666" }}>
      {timeStr}
    </span>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function MessageList({ messages, onEditMessage }) {
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const empty = messages.length === 0;

  // Auto-scroll to bottom
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (autoScroll) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, autoScroll]);

  return (
    <div
      ref={containerRef}
      className="message-list-scroll"
      style={{
        overflowY: "auto",
        padding: 16,
        height: "100%",
        minHeight: 0,
        display: empty ? "flex" : "block",
        alignItems: empty ? "center" : undefined,
        justifyContent: empty ? "center" : undefined,
      }}
      onScroll={(e) => {
        const el = e.currentTarget;
        const bottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        setAutoScroll(bottom < 30);
      }}
    >
      {empty ? (
        <div style={{ 
          color: "#9aa0a6", 
          fontSize: 14, 
          textAlign: "center",
          maxWidth: 400,
        }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>💬</div>
          <div style={{ marginBottom: 8 }}>Henüz mesaj yok.</div>
          <div style={{ fontSize: 12, color: "#666" }}>
            Bir model seçin ve yazmaya başlayın.
          </div>
        </div>
      ) : (
        messages.map((m, i) => {
          const raw = marked.parse(m.content || "");
          const html = DOMPurify.sanitize(raw);
          const isUser = m.role === "user";
          
          return (
            <div
              key={i}
              className={`message ${m.role}`}
              style={{
                padding: 14,
                borderRadius: 12,
                marginBottom: 12,
                background: isUser ? "#1a2a3c" : "#1a3c2a",
                border: isUser ? "1px solid #2a3f5c" : "1px solid #2a5c3a",
                maxWidth: "95%",
                marginLeft: isUser ? "auto" : 0,
                marginRight: isUser ? 0 : "auto",
              }}
            >
              {/* Header */}
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: 8, 
                justifyContent: "space-between",
                marginBottom: 10,
                flexWrap: "wrap",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <RoleLabel role={m.role} modelName={m.modelName} />
                  <Timestamp timestamp={m.timestamp} />
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <CopyBtn text={m.content || ""} />
                  {isUser && (
                    <EditableText 
                      content={m.content || ""} 
                      onSave={(val) => onEditMessage?.(i, val)} 
                    />
                  )}
                </div>
              </div>
              
              {/* Content */}
              <div 
                style={{ 
                  marginTop: 8,
                  fontSize: 14,
                  lineHeight: 1.6,
                }} 
                className="markdown" 
                dangerouslySetInnerHTML={{ __html: html }} 
              />
            </div>
          );
        })
      )}
      
      {/* Scroll indicator */}
      {!autoScroll && messages.length > 3 && (
        <button
          onClick={() => {
            containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'smooth' });
            setAutoScroll(true);
          }}
          style={{
            position: "sticky",
            bottom: 10,
            left: "50%",
            transform: "translateX(-50%)",
            background: "#253046",
            border: "1px solid #4f8cff",
            borderRadius: 20,
            padding: "6px 16px",
            color: "#fff",
            fontSize: 12,
            cursor: "pointer",
            zIndex: 10,
          }}
        >
          ↓ En Alta Git
        </button>
      )}
    </div>
  );
}