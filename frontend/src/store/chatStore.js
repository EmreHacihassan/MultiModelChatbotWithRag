import { useState } from "react";
export function useChatState() {
  const [modelId, setModelId] = useState("default");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);

  const append = (msg) => setMessages((m) => [...m, msg]);
  const updateLastAssistant = (delta) =>
    setMessages((m) => {
      const idx = [...m].reverse().findIndex((x) => x.role === "assistant");
      if (idx === -1) return [...m, { role: "assistant", content: delta }];
      const ri = m.length - 1 - idx;
      const updated = [...m];
      updated[ri] = { ...updated[ri], content: (updated[ri].content || "") + delta };
      return updated;
    });
  return { modelId, setModelId, messages, append, updateLastAssistant, loading, setLoading, streaming, setStreaming };
}
