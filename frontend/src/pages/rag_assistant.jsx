// RAG AI Satellite Assistant Page
import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import api from "../services/api";
import Sidebar from "../components/Sidebar";

function RAGAssistant() {
  const [searchParams] = useSearchParams();
  const initialJobId = searchParams.get("job_id");

  const [prompt, setPrompt] = useState("");
  const [selectedJobId, setSelectedJobId] = useState(initialJobId || "");
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    {
      sender: "assistant",
      text: "Hello! I am your **LISS-IV Satellite RAG AI Copilot**.\n\nI can analyze your cloud-removal jobs, explain ISRO LISS-IV band spectral characteristics, calculate NDVI indices, or perform vector knowledge retrieval across remote sensing literature. How can I assist your geospatial workflow today?",
      citations: [],
      spatial_reference: null
    }
  ]);

  const chatEndRef = useRef(null);

  // Fetch available user jobs for context selection
  useEffect(() => {
    api.get("/history/")
      .then((res) => {
        setJobs(res.data);
      })
      .catch((err) => {
        console.error("Failed to load user jobs history:", err);
      });
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, loading]);

  const samplePrompts = [
    "What are the 4 spectral bands of LISS-IV?",
    "How does RAG improve cloud removal over standard U-Net?",
    "Analyze cloud cover percentage and quality metrics for my job",
    "How to calculate NDVI formula for LISS-IV NIR and Red bands?"
  ];

  const handleSend = async (textToSend) => {
    const queryText = textToSend || prompt;
    if (!queryText.trim()) return;

    const userMsg = {
      sender: "user",
      text: queryText
    };

    setChatHistory((prev) => [...prev, userMsg]);
    setPrompt("");
    setLoading(true);

    try {
      const response = await api.post("/api/v1/rag/query", {
        prompt: queryText,
        job_id: selectedJobId ? Number(selectedJobId) : null
      });

      const assistantMsg = {
        sender: "assistant",
        text: response.data.answer,
        citations: response.data.citations || [],
        spatial_reference: response.data.spatial_reference,
        job_context: response.data.job_context
      };

      setChatHistory((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      setChatHistory((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "⚠️ Sorry, an error occurred while connecting to the RAG Knowledge Engine. Please ensure the backend is running.",
          citations: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboardLayout">
      <Sidebar />

      <main className="dashboardContent" style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden", paddingBottom: "20px" }}>
        {/* Header */}
        <div className="dashboardHeader" style={{ paddingBottom: "15px", borderBottom: "1px solid #1e2d42" }}>
          <div>
            <div style={{ background: "linear-gradient(90deg, #168cff, #8b5cf6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", fontWeight: "bold", fontSize: "0.85rem", letterSpacing: "1px" }}>
              RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE
            </div>
            <h1 style={{ marginTop: "4px" }}>✦ Satellite RAG AI Assistant</h1>
            <p style={{ color: "#94a3b8", fontSize: "0.9rem" }}>
              Grounded AI reasoning powered by vector domain knowledge, ISRO sensor specs, and multi-temporal spatial index retrieval.
            </p>
          </div>

          {/* Job Context Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", background: "#0a1626", border: "1px solid #23354d", padding: "8px 14px", borderRadius: "8px" }}>
            <span style={{ fontSize: "0.85rem", color: "#64748b" }}>Target Job Context:</span>
            <select
              value={selectedJobId}
              onChange={(e) => setSelectedJobId(e.target.value)}
              style={{
                background: "#030a14",
                border: "1px solid #334155",
                color: "#e2e8f0",
                padding: "6px 10px",
                borderRadius: "6px",
                fontSize: "0.85rem",
                outline: "none"
              }}
            >
              <option value="">General Knowledge (No Job)</option>
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  Job #{j.id} ({j.filename}) - {j.status}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Chat Thread Container */}
        <div 
          style={{ 
            flex: 1, 
            overflowY: "auto", 
            padding: "20px 10px", 
            display: "flex", 
            flexDirection: "column", 
            gap: "20px" 
          }}
        >
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              style={{
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                maxWidth: msg.sender === "user" ? "75%" : "88%",
                background: msg.sender === "user" ? "#168cff" : "#091424",
                border: msg.sender === "user" ? "none" : "1px solid #1e2e45",
                borderRadius: msg.sender === "user" ? "16px 16px 2px 16px" : "16px 16px 16px 2px",
                padding: "18px 22px",
                color: "#f8fafc",
                boxShadow: "0 4px 20px rgba(0,0,0,0.25)"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                <span style={{ fontSize: "1rem" }}>{msg.sender === "user" ? "👤 You" : "✦ RAG Copilot"}</span>
                {msg.sender === "assistant" && (
                  <span style={{ fontSize: "0.75rem", background: "#0284c7", color: "white", padding: "2px 8px", borderRadius: "10px" }}>
                    Vector Grounded
                  </span>
                )}
              </div>

              {/* Message Content */}
              <div style={{ lineHeight: "1.65", fontSize: "0.95rem", whiteSpace: "pre-wrap" }}>
                {msg.text}
              </div>

              {/* RAG Spatial Reference Card */}
              {msg.spatial_reference && msg.spatial_reference.retrieved_patches && (
                <div style={{ marginTop: "16px", background: "#040c17", border: "1px solid #16365c", borderRadius: "10px", padding: "12px 15px" }}>
                  <div style={{ fontSize: "0.8rem", color: "#38bdf8", fontWeight: "bold", textTransform: "uppercase" }}>
                    🗺️ Retrieved Spatial Reference Context
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "#cbd5e1", marginTop: "4px" }}>
                    Primary match: <strong>{msg.spatial_reference.primary_patch_id}</strong> (Similarity: {(msg.spatial_reference.best_similarity_score * 100).toFixed(1)}%)
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "#64748b", marginTop: "2px" }}>
                    {msg.spatial_reference.rag_summary}
                  </div>
                </div>
              )}

              {/* Citations Drawer */}
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ marginTop: "14px", paddingTop: "10px", borderTop: "1px solid #162438", display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  <span style={{ fontSize: "0.75rem", color: "#64748b", width: "100%" }}>Source Knowledge Citations:</span>
                  {msg.citations.map((c, cIdx) => (
                    <span 
                      key={cIdx} 
                      style={{ 
                        fontSize: "0.75rem", 
                        background: "#0d223a", 
                        border: "1px solid #1e3a5f", 
                        color: "#38bdf8", 
                        padding: "4px 10px", 
                        borderRadius: "6px" 
                      }}
                    >
                      📖 [{c.category}] {c.title} (Score: {c.relevance_score})
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div style={{ alignSelf: "flex-start", background: "#091424", border: "1px solid #1e2e45", borderRadius: "14px", padding: "15px 20px", display: "flex", alignItems: "center", gap: "10px" }}>
              <div className="spinner" style={{ width: "18px", height: "18px" }}></div>
              <span style={{ color: "#38bdf8", fontSize: "0.9rem" }}>Searching Satellite Vector Index & Synthesizing Grounded Answer...</span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Quick Sample Prompts Pills */}
        <div style={{ display: "flex", gap: "10px", overflowX: "auto", padding: "10px 0", borderTop: "1px solid #1a273a" }}>
          {samplePrompts.map((sp, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(sp)}
              style={{
                whiteSpace: "nowrap",
                background: "#081526",
                border: "1px solid #20354e",
                color: "#94a3b8",
                padding: "6px 14px",
                borderRadius: "20px",
                fontSize: "0.8rem",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#168cff";
                e.currentTarget.style.color = "#38bdf8";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#20354e";
                e.currentTarget.style.color = "#94a3b8";
              }}
            >
              ✦ {sp}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div style={{ display: "flex", gap: "12px", marginTop: "8px" }}>
          <input
            type="text"
            placeholder="Ask RAG Assistant about LISS-IV bands, cloud removal, job performance, or NDVI..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            style={{
              flex: 1,
              padding: "14px 18px",
              background: "#030a14",
              border: "1px solid #23354d",
              borderRadius: "10px",
              color: "white",
              fontSize: "0.95rem",
              outline: "none"
            }}
          />
          <button
            className="primaryBtn"
            onClick={() => handleSend()}
            disabled={loading || !prompt.trim()}
            style={{ padding: "0 28px", borderRadius: "10px" }}
          >
            Send ✦
          </button>
        </div>
      </main>
    </div>
  );
}

export default RAGAssistant;
