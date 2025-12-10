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
    thought: { color: "#f0ad4e", icon: "💭", label: "Düşünce" },
    tool: { color: "#9b59b6", icon: "🔧", label: "Araç" },
    rag: { color: "#3498db", icon: "📚", label: "Kaynak" },
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
// AGENT COMPONENTS
// =============================================================================

function ThoughtBubble({ content }) {
  return (
    <div style={{
      background: "linear-gradient(135deg, #2a2a1a 0%, #3a3a2a 100%)",
      border: "1px solid #5a5a3a",
      borderRadius: 12,
      padding: 12,
      marginBottom: 8,
      fontSize: 13,
      fontStyle: "italic",
      color: "#f0e68c",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <span>💭</span>
        <strong style={{ color: "#f0ad4e", fontSize: 11 }}>Düşünce</strong>
      </div>
      <div style={{ color: "#e6e6e6" }}>{content}</div>
    </div>
  );
}

function ToolCallCard({ tool, input, result, isExpanded, onToggle }) {
  const toolIcons = {
    calculator: "🔢",
    python_executor: "🐍",
    web_search: "🔍",
    datetime: "📅",
    wikipedia: "📖",
    json_parser: "📋",
    text_analyzer: "📝",
    unit_converter: "📏",
  };
  
  return (
    <div style={{
      background: "linear-gradient(135deg, #1a2a3a 0%, #2a3a4a 100%)",
      border: "1px solid #3a5a7a",
      borderRadius: 12,
      padding: 12,
      marginBottom: 8,
    }}>
      <div 
        style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: 8,
          cursor: "pointer",
        }}
        onClick={onToggle}
      >
        <span style={{ fontSize: 18 }}>{toolIcons[tool] || "🔧"}</span>
        <strong style={{ color: "#9b59b6", fontSize: 12 }}>{tool}</strong>
        <span style={{ color: "#666", fontSize: 11, marginLeft: "auto" }}>
          {isExpanded ? "▼" : "▶"}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ marginTop: 10 }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 10, color: "#666", marginBottom: 4 }}>GİRİŞ:</div>
            <code style={{ 
              background: "#0d1117", 
              padding: 8, 
              borderRadius: 6, 
              fontSize: 12,
              display: "block",
              color: "#58a6ff",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}>
              {typeof input === "object" ? JSON.stringify(input, null, 2) : input}
            </code>
          </div>
          
          {result && (
            <div>
              <div style={{ fontSize: 10, color: "#666", marginBottom: 4 }}>ÇIKIŞ:</div>
              <code style={{ 
                background: "#0d1117", 
                padding: 8, 
                borderRadius: 6, 
                fontSize: 12,
                display: "block",
                color: "#7ee787",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                maxHeight: 200,
                overflow: "auto",
              }}>
                {result}
              </code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RAGContextCard({ docs, isExpanded, onToggle }) {
  if (!docs || docs.length === 0) return null;
  
  return (
    <div style={{
      background: "linear-gradient(135deg, #1a3a2a 0%, #2a4a3a 100%)",
      border: "1px solid #3a7a5a",
      borderRadius: 12,
      padding: 12,
      marginBottom: 8,
    }}>
      <div 
        style={{ 
          display: "flex", 
          alignItems: "center", 
          gap: 8,
          cursor: "pointer",
        }}
        onClick={onToggle}
      >
        <span style={{ fontSize: 18 }}>📚</span>
        <strong style={{ color: "#3498db", fontSize: 12 }}>
          Bulunan Kaynaklar ({docs.length})
        </strong>
        <span style={{ color: "#666", fontSize: 11, marginLeft: "auto" }}>
          {isExpanded ? "▼" : "▶"}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ marginTop: 10 }}>
          {docs.map((doc, idx) => (
            <div 
              key={idx}
              style={{
                background: "#0d1117",
                borderRadius: 8,
                padding: 10,
                marginBottom: 8,
                border: "1px solid #2a4a3a",
              }}
            >
              <div style={{ 
                display: "flex", 
                justifyContent: "space-between", 
                alignItems: "center",
                marginBottom: 6,
              }}>
                <span style={{ fontSize: 11, color: "#3498db" }}>
                  📄 {doc.source || "Bilinmeyen Kaynak"}
                </span>
                {doc.score && (
                  <span style={{ 
                    fontSize: 10, 
                    color: "#5ac98f",
                    background: "#1a3c2a",
                    padding: "2px 6px",
                    borderRadius: 4,
                  }}>
                    Benzerlik: {(doc.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <div style={{ 
                fontSize: 12, 
                color: "#e6e6e6",
                lineHeight: 1.5,
              }}>
                {doc.content?.substring(0, 300)}
                {doc.content?.length > 300 && "..."}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function MessageList({ messages, onEditMessage, agentSteps, ragContext }) {
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedTools, setExpandedTools] = useState({});
  const [ragExpanded, setRagExpanded] = useState(true);
  const empty = messages.length === 0;

  // Auto-scroll to bottom
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (autoScroll) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, autoScroll, agentSteps, ragContext]);
  
  const toggleTool = (idx) => {
    setExpandedTools(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

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
        <>
          {/* RAG Context (if available) */}
          {ragContext && ragContext.length > 0 && (
            <RAGContextCard 
              docs={ragContext}
              isExpanded={ragExpanded}
              onToggle={() => setRagExpanded(!ragExpanded)}
            />
          )}
          
          {/* Messages */}
          {messages.map((m, i) => {
            const raw = marked.parse(m.content || "");
            const html = DOMPurify.sanitize(raw);
            const isUser = m.role === "user";
            
            return (
              <div key={i}>
                {/* Agent thoughts (show before assistant messages) */}
                {m.role === "assistant" && m.thoughts && m.thoughts.map((thought, tIdx) => (
                  <ThoughtBubble key={`thought-${tIdx}`} content={thought} />
                ))}
                
                {/* Agent tool calls */}
                {m.role === "assistant" && m.toolCalls && m.toolCalls.map((tc, tcIdx) => (
                  <ToolCallCard 
                    key={`tool-${tcIdx}`}
                    tool={tc.tool}
                    input={tc.input}
                    result={tc.result}
                    isExpanded={expandedTools[`${i}-${tcIdx}`] !== false}
                    onToggle={() => toggleTool(`${i}-${tcIdx}`)}
                  />
                ))}
                
                {/* Main message */}
                <div
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
                      {m.mode && (
                        <span style={{
                          fontSize: 10,
                          padding: "2px 6px",
                          borderRadius: 4,
                          background: m.mode === "agent" ? "#2a1a3a" : "#1a2a3a",
                          border: m.mode === "agent" ? "1px solid #5a3a7a" : "1px solid #3a5a7a",
                          color: m.mode === "agent" ? "#9b59b6" : "#3498db",
                        }}>
                          {m.mode === "agent" ? "🤖 Agent" : "📚 RAG"}
                        </span>
                      )}
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
              </div>
            );
          })}
          
          {/* Real-time agent steps (while streaming) */}
          {agentSteps && agentSteps.length > 0 && (
            <div style={{ 
              marginTop: 12, 
              padding: 12,
              background: "#1a1a2a",
              border: "1px solid #3a3a5a",
              borderRadius: 12,
            }}>
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: 6, 
                marginBottom: 10,
                color: "#f0ad4e",
                fontSize: 12,
              }}>
                <span className="loading-dots">⏳</span>
                <strong>Agent Çalışıyor...</strong>
              </div>
              {agentSteps.map((step, idx) => (
                <div key={idx} style={{ marginBottom: 8 }}>
                  {step.type === "thought" && (
                    <ThoughtBubble content={step.content} />
                  )}
                  {step.type === "tool_call" && (
                    <ToolCallCard 
                      tool={step.tool}
                      input={step.input}
                      result={step.result}
                      isExpanded={true}
                      onToggle={() => {}}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </>
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