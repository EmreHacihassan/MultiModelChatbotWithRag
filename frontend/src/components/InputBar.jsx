import { useEffect, useMemo, useRef, useState } from "react";
import { uploadRagDocument, listRagDocuments, clearRagDocuments } from "../lib/api.js";

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
  const [files, setFiles] = useState([]); // Seçilen dosyalar
  const [uploadedDocs, setUploadedDocs] = useState([]); // RAG'a yüklenmiş dökümanlar
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0); // Upload yüzdesi
  const [uploadStage, setUploadStage] = useState(""); // İşlem aşaması
  const [uploadError, setUploadError] = useState(null);
  const [maxTokens] = useState(4096); // can be dynamic per model
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const tokenCount = useMemo(() => estimateTokens(text), [text]);
  const nearLimit = tokenCount > maxTokens * 0.9;
  
  // RAG modu açıldığında yüklü dökümanları getir
  useEffect(() => {
    if (useRag) {
      loadUploadedDocs();
    }
  }, [useRag]);
  
  const loadUploadedDocs = async () => {
    try {
      const result = await listRagDocuments();
      setUploadedDocs(result.documents || []);
    } catch (err) {
      console.error("Döküman listesi alınamadı:", err);
    }
  };

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
  
  const handleFileSelect = async (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    
    // RAG modu aktifse hemen yükle
    if (useRag) {
      setUploading(true);
      setUploadError(null);
      setUploadProgress(0);
      setUploadStage("Dosya yükleniyor...");
      
      for (const file of selectedFiles) {
        try {
          setUploadStage(`${file.name} yükleniyor ve indeksleniyor...`);
          setUploadProgress(10);
          
          // Senkron embedding - backend işlem bitene kadar bekleyecek
          const result = await uploadRagDocument(file);
          console.log(`Döküman yüklendi: ${file.name}`, result);
          
          if (result.ok || result.success) {
            setUploadProgress(100);
            setUploadStage(`✅ ${file.name} - ${result.chunks_added || 0} parça indekslendi`);
            setUploadError(null);
          } else {
            setUploadError(`${file.name}: ${result.error || 'Bilinmeyen hata'}`);
          }
          
          // Kısa bekle
          await new Promise(r => setTimeout(r, 1500));
        } catch (err) {
          console.error(`Yükleme hatası: ${file.name}`, err);
          const errorMsg = err.response?.data?.error || err.message || 'Bilinmeyen hata';
          setUploadError(`${file.name}: ${errorMsg}`);
        }
      }
      
      // Yüklü dökümanları yenile
      await loadUploadedDocs();
      setUploading(false);
      setUploadProgress(0);
      setUploadStage("");
    } else {
      // Normal mod - dosyaları state'e ekle
      setFiles((prev) => [...prev, ...selectedFiles]);
    }
  };
  
  const removeFile = (idx) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };
  
  const clearAllDocs = async () => {
    if (!confirm("Tüm yüklenmiş dökümanları silmek istediğinizden emin misiniz?")) return;
    
    try {
      await clearRagDocuments();
      setUploadedDocs([]);
    } catch (err) {
      console.error("Dökümanlar temizlenemedi:", err);
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
        
        {/* RAG: Yüklenmiş dökümanlar */}
        {useRag && (
          <div style={{
            background: "#0d1a2a",
            border: "1px solid #2a4a6a",
            borderRadius: 8,
            padding: "8px 12px",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: "#7fb3d5", fontWeight: "bold" }}>
                📁 Yüklü Dökümanlar ({uploadedDocs.length})
              </span>
              {uploadedDocs.length > 0 && (
                <button
                  onClick={clearAllDocs}
                  style={{
                    background: "#3a2020",
                    border: "1px solid #5a3030",
                    borderRadius: 4,
                    padding: "2px 8px",
                    fontSize: 10,
                    color: "#e66d6d",
                    cursor: "pointer",
                  }}
                >
                  Tümünü Sil
                </button>
              )}
            </div>
            
            {uploading && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: "#5ac98f" }}>
                    ⏳ {uploadStage || "Döküman yükleniyor..."}
                  </span>
                  <span style={{ fontSize: 11, color: "#7fb3d5", fontWeight: "bold" }}>
                    {uploadProgress}%
                  </span>
                </div>
                {/* Progress bar */}
                <div style={{
                  width: "100%",
                  height: 4,
                  background: "#1a2a3a",
                  borderRadius: 2,
                  overflow: "hidden",
                }}>
                  <div style={{
                    width: `${uploadProgress}%`,
                    height: "100%",
                    background: uploadProgress === 100 ? "#5ac98f" : "#4f8cff",
                    transition: "width 0.3s ease",
                  }} />
                </div>
              </div>
            )}
            
            {uploadError && (
              <div style={{ fontSize: 11, color: "#e66d6d", marginBottom: 6 }}>
                ⚠️ {uploadError}
              </div>
            )}
            
            {uploadedDocs.length === 0 && !uploading ? (
              <div style={{ fontSize: 11, color: "#666" }}>
                Henüz döküman yüklenmedi. Dosya butonuna tıklayarak PDF, TXT, DOCX veya MD dosyası yükleyin.
              </div>
            ) : (
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {uploadedDocs.map((doc, idx) => (
                  <span
                    key={doc.hash || idx}
                    style={{
                      fontSize: 10,
                      background: "#1a2a3a",
                      border: "1px solid #3a5a7a",
                      borderRadius: 6,
                      padding: "3px 8px",
                      color: "#9dc3e6",
                    }}
                    title={`${doc.chunk_count || 0} parça, ${doc.char_count || 0} karakter`}
                  >
                    📄 {doc.file_name}
                  </span>
                ))}
              </div>
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