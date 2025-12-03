import { useEffect, useMemo, useRef, useState } from "react";

// naive token estimator: ~4 chars = 1 token (rough)
function estimateTokens(txt) {
  const t = Math.ceil((txt || "").length / 4);
  return t;
}

export default function InputBar({ onSend, disabled, onStop }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [maxTokens] = useState(4096); // can be dynamic per model
  const textareaRef = useRef(null);

  const tokenCount = useMemo(() => estimateTokens(text), [text]);
  const nearLimit = tokenCount > maxTokens * 0.9;

  const submit = () => {
    const t = text.trim();
    if (!t || disabled) return;
    onSend?.(t, files);
    setText("");
    setFiles([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = "60px";
    }
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
  }, [disabled, text, files]);

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
        <div style={{ display: "flex", gap: 8 }}>
          <button
            title="Dosya ekle (stub)"
            onClick={() => {
              // This is a stub; implement upload to backend as needed
              const fake = { name: "ornek.txt", size: 123 };
              setFiles((f) => [...f, fake]);
            }}
            style={{ background: "#2a2f37", border: "1px solid #3a3f47" }}
          >
            + Dosya
          </button>
          <div style={{ fontSize: 12, color: nearLimit ? "#e66d6d" : "#9aa0a6" }}>
            Tahmini token: {tokenCount} / {maxTokens}
          </div>
        </div>
        <textarea
          ref={textareaRef}
          rows={3}
          placeholder="Mesajınızı yazın… (Göndermek için Ctrl/Cmd + Enter)"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            // auto-resize
            const el = textareaRef.current;
            if (!el) return;
            el.style.height = "auto";
            el.style.height = Math.min(el.scrollHeight, 180) + "px";
          }}
          disabled={disabled}
          style={{ flex: 1, resize: "none" }}
        />
        {files.length > 0 ? (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {files.map((f, idx) => (
              <span
                key={idx}
                style={{
                  fontSize: 12,
                  background: "#1a1d22",
                  border: "1px solid #2a2f37",
                  borderRadius: 8,
                  padding: "4px 8px",
                }}
              >
                {f.name} · {f.size}B
              </span>
            ))}
          </div>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <button onClick={submit} disabled={disabled}>
          Gönder
        </button>
        <button
          onClick={() => onStop?.()}
          disabled={!disabled} // Stop only visible when streaming/disabled
          style={{ background: "#e66d6d" }}
          title="Akışı durdur"
        >
          Durdur
        </button>
      </div>
    </div>
  );
}