import { useEffect, useMemo, useState } from "react";

// Provider badge colors
const providerStyles = {
  gemini: { bg: "#2a3c57", fg: "#a7c3ff" },
  hf: { bg: "#3a2a57", fg: "#d1a7ff" },
  ollama: { bg: "#2a573a", fg: "#a7ffd1" },
  generic: { bg: "#2a2f37", fg: "#9aa0a6" },
};

function ProviderBadge({ provider }) {
  const style = providerStyles[provider] || providerStyles.generic;
  return (
    <span
      style={{
        background: style.bg,
        color: style.fg,
        fontSize: 11,
        padding: "2px 6px",
        borderRadius: 12,
        marginLeft: 8,
        border: "1px solid #2a2f37",
      }}
    >
      {provider.toUpperCase()}
    </span>
  );
}

function ModelOption({ m }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <strong>{m.name}</strong>
        <ProviderBadge provider={m.provider || "generic"} />
        {m.streaming ? (
          <span style={{ fontSize: 11, color: "#5ac98f", marginLeft: 6 }}>Streaming</span>
        ) : (
          <span style={{ fontSize: 11, color: "#e66d6d", marginLeft: 6 }}>No stream</span>
        )}
      </div>
      {m.desc ? (
        <div title={m.desc} style={{ fontSize: 12, color: "#9aa0a6", marginTop: 2 }}>
          {m.desc.length > 64 ? m.desc.slice(0, 64) + "…" : m.desc}
        </div>
      ) : null}
    </div>
  );
}

export default function ModelSelector({ models = [], value, onChange }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  // Decorate and normalize
  const enriched = useMemo(
    () =>
      models.map((m) => {
        const id = m.id || "";
        let provider = m.provider || "generic";
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

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return enriched;
    return enriched.filter(
      (m) =>
        m.name?.toLowerCase().includes(q) ||
        m.id?.toLowerCase().includes(q) ||
        m.provider?.toLowerCase().includes(q)
    );
  }, [query, enriched]);

  const current = enriched.find((m) => m.id === value) || enriched[0];

  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div style={{ position: "relative", display: "flex", gap: 8, alignItems: "center" }}>
      <label style={{ fontSize: 13, color: "#9aa0a6" }}>Model:</label>
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
          gap: 8,
        }}
        title={current?.desc || ""}
      >
        <span>{current?.name || "Seçiniz"}</span>
        {current ? <ProviderBadge provider={current.provider || "generic"} /> : null}
        <span style={{ fontSize: 11, color: "#9aa0a6" }}>⌄</span>
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            top: 40,
            left: 52,
            zIndex: 1000,
            width: 420,
            background: "#0f1114",
            border: "1px solid #2a2f37",
            borderRadius: 10,
            boxShadow: "0 6px 20px rgba(0,0,0,0.35)",
          }}
        >
          <div style={{ display: "flex", gap: 8, padding: 8, borderBottom: "1px solid #2a2f37" }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ara (gemini, mistral, qwen)…"
              style={{ flex: 1 }}
            />
            <button onClick={() => setQuery("")} style={{ background: "#2a2f37" }}>
              Temizle
            </button>
          </div>
          <div style={{ maxHeight: 260, overflow: "auto", padding: 8, display: "grid", gap: 8 }}>
            {filtered.map((m) => (
              <button
                key={m.id}
                onClick={() => {
                  onChange?.(m.id);
                  setOpen(false);
                }}
                style={{
                  textAlign: "left",
                  background: value === m.id ? "#253046" : "#1a1d22",
                  border: "1px solid #2a2f37",
                  borderRadius: 8,
                  padding: "8px 10px",
                  color: "#e6e6e6",
                  cursor: "pointer",
                }}
                title={m.desc || ""}
              >
                <ModelOption m={m} />
              </button>
            ))}
            {filtered.length === 0 ? (
              <div style={{ fontSize: 12, color: "#9aa0a6" }}>Sonuç yok.</div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}