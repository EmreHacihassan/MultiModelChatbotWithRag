import { useEffect, useRef, useState } from "react";
import { marked } from "marked";
import hljs from "highlight.js/lib/common";
import "highlight.js/styles/github-dark.css";
import DOMPurify from "dompurify";

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

function RoleLabel({ role }) {
  const color = role === "user" ? "#4f8cff" : role === "assistant" ? "#5ac98f" : "#9aa0a6";
  return <strong style={{ color }}>{role}</strong>;
}

function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text || "");
          setCopied(true);
          setTimeout(() => setCopied(false), 1000);
        } catch {}
      }}
      style={{
        background: "#2a2f37",
        border: "1px solid #3a3f47",
        borderRadius: 8,
        color: "#e6e6e6",
        fontSize: 12,
        padding: "4px 8px",
      }}
      title="Panoya kopyala"
    >
      {copied ? "Kopyalandı" : "Kopyala"}
    </button>
  );
}

function EditableText({ content, onSave }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(content || "");
  if (!editing)
    return (
      <button
        onClick={() => setEditing(true)}
        style={{
          background: "#2a2f37",
          border: "1px solid #3a3f47",
          borderRadius: 8,
          color: "#e6e6e6",
          fontSize: 12,
          padding: "4px 8px",
          marginLeft: 8,
        }}
        title="Mesajı düzelt"
      >
        Düzelt
      </button>
    );
  return (
    <span style={{ display: "flex", gap: 6, alignItems: "center", marginLeft: 8 }}>
      <input value={val} onChange={(e) => setVal(e.target.value)} style={{ flex: 1 }} />
      <button
        onClick={() => { onSave?.(val); setEditing(false); }}
        style={{ background: "#4f8cff", border: "none", borderRadius: 8, color: "#fff", fontSize: 12, padding: "4px 8px" }}
      >
        Kaydet
      </button>
      <button
        onClick={() => { setVal(content || ""); setEditing(false); }}
        style={{ background: "#2a2f37", border: "1px solid #3a3f47", borderRadius: 8, color: "#e6e6e6", fontSize: 12, padding: "4px 8px" }}
      >
        İptal
      </button>
    </span>
  );
}

export default function MessageList({ messages, onEditMessage }) {
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const empty = messages.length === 0;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (autoScroll) el.scrollTop = el.scrollHeight;
  }, [messages, autoScroll]);

  return (
    <div
      ref={containerRef}
      className="message-list-scroll"
      style={{
        overflowY: "auto",
        padding: 12,
        height: "100%",
        minHeight: 0,
        display: empty ? "flex" : "block",
        alignItems: empty ? "center" : undefined,
        justifyContent: empty ? "center" : undefined,
      }}
      onScroll={(e) => {
        const el = e.currentTarget;
        const bottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        setAutoScroll(bottom < 20);
      }}
    >
      {empty ? (
        <div style={{ color: "#9aa0a6", fontSize: 13 }}>
          Mesaj yok. Soldan yeni bir oturum oluşturup yazmaya başlayın.
        </div>
      ) : (
        messages.map((m, i) => {
          const raw = marked.parse(m.content || "");
          const html = DOMPurify.sanitize(raw);
          return (
            <div
              key={i}
              className={`message ${m.role}`}
              style={{
                padding: 12,
                borderRadius: 10,
                marginBottom: 10,
                background: m.role === "user" ? "#253046" : "#213524",
                border: "1px solid #2a2f37",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <RoleLabel role={m.role} />
                  {m.modelName ? <span className="model-badge">{` · ${m.modelName}`}</span> : null}
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <CopyBtn text={m.content || ""} />
                  <EditableText content={m.content || ""} onSave={(val) => onEditMessage?.(i, val)} />
                </div>
              </div>
              <div style={{ marginTop: 8 }} className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
            </div>
          );
        })
      )}
    </div>
  );
}