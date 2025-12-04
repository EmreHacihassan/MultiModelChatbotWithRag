import { useEffect, useMemo, useState } from "react";

// =============================================================================
// STYLE CONFIGURATIONS
// =============================================================================

// Provider badge colors
const providerStyles = {
  gemini: { bg: "#1a3a5c", fg: "#64b5f6", label: "GEMINI", icon: "✨" },
  hf: { bg: "#3a1a5c", fg: "#ce93d8", label: "HF", icon: "🤗" },
  ollama: { bg: "#1a5c3a", fg: "#81c784", label: "OLLAMA", icon: "🦙" },
  generic: { bg: "#2a2f37", fg: "#9aa0a6", label: "OTHER", icon: "🤖" },
};

// Tier badge colors with descriptions
const tierStyles = {
  1: { bg: "#1a4a1a", fg: "#66ff66", label: "★★★", desc: "En Güvenilir" },
  2: { bg: "#4a4a1a", fg: "#ffff66", label: "★★", desc: "Güvenilir" },
  3: { bg: "#5a3a1a", fg: "#ffcc66", label: "★", desc: "İyi" },
  4: { bg: "#5a2a1a", fg: "#ff9966", label: "•", desc: "Yedek" },
  5: { bg: "#4a1a1a", fg: "#ff6666", label: "·", desc: "Deneysel" },
};

// =============================================================================
// BADGE COMPONENTS
// =============================================================================

function ProviderBadge({ provider }) {
  const style = providerStyles[provider] || providerStyles.generic;
  return (
    <span
      style={{
        background: style.bg,
        color: style.fg,
        fontSize: 10,
        padding: "2px 6px",
        borderRadius: 12,
        marginLeft: 6,
        border: `1px solid ${style.fg}33`,
        fontWeight: "bold",
        display: "inline-flex",
        alignItems: "center",
        gap: 3,
      }}
      title={`Provider: ${style.label}`}
    >
      <span>{style.icon}</span>
      {style.label}
    </span>
  );
}

function TierBadge({ tier }) {
  if (!tier) return null;
  const style = tierStyles[tier] || tierStyles[5];
  return (
    <span
      style={{
        background: style.bg,
        color: style.fg,
        fontSize: 10,
        padding: "2px 5px",
        borderRadius: 8,
        marginLeft: 4,
        border: `1px solid ${style.fg}33`,
      }}
      title={`Güvenilirlik: ${style.desc} (Tier ${tier})`}
    >
      {style.label}
    </span>
  );
}

function StreamingIndicator({ streaming }) {
  return streaming ? (
    <span 
      style={{ fontSize: 10, color: "#5ac98f", marginLeft: 4 }} 
      title="Streaming destekli"
    >
      ●
    </span>
  ) : (
    <span 
      style={{ fontSize: 10, color: "#e66d6d", marginLeft: 4 }} 
      title="Streaming yok"
    >
      ○
    </span>
  );
}

// =============================================================================
// MODEL OPTION COMPONENT
// =============================================================================

function ModelOption({ m, isSelected }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
        <strong style={{ fontSize: 13, color: isSelected ? "#fff" : "#e6e6e6" }}>
          {m.name}
        </strong>
        <ProviderBadge provider={m.provider || "generic"} />
        {m.tier && <TierBadge tier={m.tier} />}
        <StreamingIndicator streaming={m.streaming} />
      </div>
      {m.desc && (
        <div 
          title={m.desc} 
          style={{ 
            fontSize: 11, 
            color: isSelected ? "#b0b8c0" : "#9aa0a6", 
            marginTop: 2,
            lineHeight: 1.3,
          }}
        >
          {m.desc.length > 60 ? m.desc.slice(0, 60) + "…" : m.desc}
        </div>
      )}
      {m.context_window && (
        <div style={{ fontSize: 10, color: "#666", marginTop: 2 }}>
          Context: {(m.context_window / 1024).toFixed(0)}K tokens
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function ModelSelector({ models = [], value, onChange }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [filterProvider, setFilterProvider] = useState("all");

  // =========================================================================
  // ENRICH AND NORMALIZE MODELS
  // =========================================================================
  
  const enriched = useMemo(
    () =>
      models.map((m) => {
        const id = m.id || "";
        let provider = m.provider || "generic";
        
        // Auto-detect provider from ID if not set
        if (!m.provider) {
          if (id.startsWith("gemini")) provider = "gemini";
          else if (id.startsWith("hf")) provider = "hf";
          else if (id.startsWith("ollama")) provider = "ollama";
        }
        
        return {
          provider,
          streaming: m.streaming !== false,
          desc:
            m.desc ||
            m.description ||
            (provider === "gemini"
              ? "Google Gemini — ileri yetenekler, kalite."
              : provider === "hf"
              ? "HuggingFace — açık kaynak ekosistem."
              : provider === "ollama"
              ? "Ollama — yerel modeller, gizlilik."
              : "Model."),
          ...m,
        };
      }),
    [models]
  );

  // =========================================================================
  // FILTER LOGIC
  // =========================================================================
  
  const filtered = useMemo(() => {
    let result = enriched;
    
    // Provider filter
    if (filterProvider !== "all") {
      result = result.filter((m) => m.provider === filterProvider);
    }
    
    // Text query filter
    const q = query.trim().toLowerCase();
    if (q) {
      result = result.filter(
        (m) =>
          m.name?.toLowerCase().includes(q) ||
          m.id?.toLowerCase().includes(q) ||
          m.provider?.toLowerCase().includes(q) ||
          m.desc?.toLowerCase().includes(q)
      );
    }
    
    // Sort: Tier 1 first, then by provider
    result.sort((a, b) => {
      const tierA = a.tier || 99;
      const tierB = b.tier || 99;
      if (tierA !== tierB) return tierA - tierB;
      return (a.name || "").localeCompare(b.name || "");
    });
    
    return result;
  }, [query, enriched, filterProvider]);

  const current = enriched.find((m) => m.id === value) || enriched[0];

  // Provider counts for filter buttons
  const providerCounts = useMemo(() => {
    const counts = { all: enriched.length, gemini: 0, hf: 0, ollama: 0 };
    enriched.forEach((m) => {
      if (counts[m.provider] !== undefined) counts[m.provider]++;
    });
    return counts;
  }, [enriched]);

  // =========================================================================
  // KEYBOARD & CLICK HANDLERS
  // =========================================================================
  
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (!e.target.closest('.model-selector-container')) {
        setOpen(false);
      }
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, [open]);

  // =========================================================================
  // RENDER
  // =========================================================================

  return (
    <div 
      className="model-selector-container" 
      style={{ position: "relative", display: "flex", gap: 8, alignItems: "center" }}
    >
      <label style={{ fontSize: 13, color: "#9aa0a6" }}>Model:</label>
      
      {/* Trigger Button */}
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          background: "#1a1d22",
          border: "1px solid #2a2f37",
          borderRadius: 8,
          color: "#e6e6e6",
          padding: "8px 12px",
          display: "flex",
          alignItems: "center",
          gap: 6,
          minWidth: 220,
          cursor: "pointer",
          transition: "border-color 0.15s ease",
        }}
        title={current?.desc || "Model seçin"}
        onMouseOver={(e) => e.currentTarget.style.borderColor = "#4f8cff"}
        onMouseOut={(e) => e.currentTarget.style.borderColor = "#2a2f37"}
      >
        <span style={{ flex: 1, textAlign: "left", fontSize: 13 }}>
          {current?.name || "Seçiniz"}
        </span>
        {current && <ProviderBadge provider={current.provider || "generic"} />}
        {current?.tier && <TierBadge tier={current.tier} />}
        <span style={{ fontSize: 11, color: "#9aa0a6", marginLeft: 4 }}>
          {open ? "▲" : "▼"}
        </span>
      </button>

      {/* Dropdown */}
      {open && (
        <div
          style={{
            position: "absolute",
            top: 44,
            left: 52,
            zIndex: 1000,
            width: 500,
            background: "#0f1114",
            border: "1px solid #2a2f37",
            borderRadius: 12,
            boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
          }}
        >
          {/* Search + Filter */}
          <div style={{ padding: 12, borderBottom: "1px solid #2a2f37" }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Model ara (qwen, llama, mistral, gemini)…"
              autoFocus
              style={{ 
                width: "100%", 
                marginBottom: 10,
                padding: "10px 14px",
                background: "#1a1d22",
                border: "1px solid #2a2f37",
                borderRadius: 8,
                color: "#e6e6e6",
                fontSize: 13,
                outline: "none",
              }}
              onFocus={(e) => e.target.style.borderColor = "#4f8cff"}
              onBlur={(e) => e.target.style.borderColor = "#2a2f37"}
            />
            
            {/* Provider Filter Buttons */}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[
                { key: "all", label: "Tümü", icon: "🌐" },
                { key: "gemini", label: "Gemini", icon: "✨" },
                { key: "hf", label: "HuggingFace", icon: "🤗" },
                { key: "ollama", label: "Ollama", icon: "🦙" },
              ].map((f) => (
                <button
                  key={f.key}
                  onClick={() => setFilterProvider(f.key)}
                  style={{
                    padding: "5px 12px",
                    fontSize: 11,
                    background: filterProvider === f.key ? "#253046" : "#1a1d22",
                    border: filterProvider === f.key ? "1px solid #4f8cff" : "1px solid #2a2f37",
                    borderRadius: 6,
                    color: filterProvider === f.key ? "#fff" : "#9aa0a6",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    transition: "all 0.15s ease",
                  }}
                >
                  <span>{f.icon}</span>
                  {f.label} ({providerCounts[f.key] || 0})
                </button>
              ))}
            </div>
          </div>

          {/* Model List */}
          <div style={{ maxHeight: 380, overflow: "auto", padding: 8 }}>
            {filtered.length === 0 ? (
              <div style={{ fontSize: 12, color: "#9aa0a6", padding: 20, textAlign: "center" }}>
                <div style={{ fontSize: 24, marginBottom: 8 }}>🔍</div>
                Sonuç bulunamadı
              </div>
            ) : (
              <div style={{ display: "grid", gap: 6 }}>
                {filtered.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => {
                      onChange?.(m.id);
                      setOpen(false);
                      setQuery("");
                    }}
                    style={{
                      textAlign: "left",
                      background: value === m.id ? "#253046" : "#1a1d22",
                      border: value === m.id ? "1px solid #4f8cff" : "1px solid #2a2f37",
                      borderRadius: 8,
                      padding: "12px 14px",
                      color: "#e6e6e6",
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                    }}
                    title={m.desc || ""}
                    onMouseOver={(e) => {
                      if (value !== m.id) {
                        e.currentTarget.style.background = "#1f2228";
                        e.currentTarget.style.borderColor = "#3a3f47";
                      }
                    }}
                    onMouseOut={(e) => {
                      if (value !== m.id) {
                        e.currentTarget.style.background = "#1a1d22";
                        e.currentTarget.style.borderColor = "#2a2f37";
                      }
                    }}
                  >
                    <ModelOption m={m} isSelected={value === m.id} />
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div style={{ 
            padding: "10px 14px", 
            borderTop: "1px solid #2a2f37", 
            fontSize: 11, 
            color: "#666",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span>{filtered.length} / {enriched.length} model</span>
            <div style={{ display: "flex", gap: 12 }}>
              <span title="En Güvenilir">★★★ = Tier 1</span>
              <span title="Streaming destekli">● = Streaming</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}