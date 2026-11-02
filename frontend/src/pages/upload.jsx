// Upload page with RAG options, backend health testing, and robust error handling
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api, { API_BASE_URL } from "../services/api";
import Sidebar from "../components/Sidebar";
import Loader from "../components/Loader";

function Upload() {
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [clearReference, setClearReference] = useState(null);
  const [preview, setPreview] = useState(null);
  const [s3Path, setS3Path] = useState("");
  const [auxPaths, setAuxPaths] = useState("");
  const [useCloudmask, setUseCloudmask] = useState(true);
  const [useRag, setUseRag] = useState(true);
  const [modelId, setModelId] = useState("synthetic_liss4_unet");
  const [patchSize, setPatchSize] = useState(512);
  const [overlap, setOverlap] = useState(64);
  const [models, setModels] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState("checking"); // 'checking' | 'connected' | 'disconnected'

  // Test backend connection and fetch models on load
  useEffect(() => {
    let intervalId;

    const checkBackendAndFetchModels = () => {
      // 1. Connection check
      api.get("/health")
        .then((res) => {
          if (res.data && res.data.status === "ok") {
            setBackendStatus("connected");
            // 2. Fetch models if not yet loaded
            api.get("/api/v1/models")
              .then((response) => {
                if (response.data && response.data.length > 0) {
                  setModels(response.data);
                  setModelId((prev) => prev || response.data[0].model_id);
                }
              })
              .catch((err) => {
                console.error("Failed to load models registry:", err);
              });
          } else {
            setBackendStatus("disconnected");
          }
        })
        .catch((err) => {
          console.error("Health check failed:", err);
          setBackendStatus("disconnected");
        });
    };

    checkBackendAndFetchModels();
    intervalId = setInterval(checkBackendAndFetchModels, 4000);

    return () => clearInterval(intervalId);
  }, []);

  const chooseFile = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setS3Path(""); // Clear S3 path when uploading locally
    setPreview(URL.createObjectURL(selectedFile));
    setError("");
  };

  const analyze = async () => {
    if (!file && !s3Path) {
      setError("Please select a local satellite image or specify an S3/MinIO path first.");
      return;
    }

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
      if (clearReference) {
        formData.append("clear_reference", clearReference);
      }
    } else {
      formData.append("s3_path", s3Path);
    }

    formData.append("use_cloudmask", useCloudmask);
    formData.append("use_rag", useRag);
    formData.append("model_id", modelId);
    formData.append("patch_size", patchSize);
    formData.append("overlap", overlap);
    if (auxPaths) {
      formData.append("aux_paths", auxPaths);
    }

    try {
      setLoading(true);
      setError("");

      const response = await api.post("/api/v1/infer", formData);

      // Navigate to results page with target job ID
      navigate(`/results?job_id=${response.data.job_id}`);
    } catch (err) {
      console.error("Infer API call failed:", err);
      
      if (!err.response) {
        // Real Network Connection Failure (Server down or refused connection)
        setError(
          `Unable to connect to FastAPI backend at ${API_BASE_URL}. Please make sure the backend server is running via 'python -m uvicorn backend.main:app --reload'.`
        );
      } else if (err.response.status === 401) {
        setError("Authentication error: Session expired or invalid token. Please log in again.");
      } else if (err.response.status === 503) {
        setError(`Backend / Model Error: ${err.response.data?.detail || "Reconstruction service unavailable."}`);
      } else {
        setError(
          err.response.data?.detail ||
          err.message ||
          "Cloud removal job scheduling failed."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboardLayout">
      <Sidebar />

      <main className="dashboardContent">
        {loading && <Loader />}

        <div className="uploadHeader">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <p className="smallTitle">RAG-ENHANCED AI SATELLITE RECONSTRUCTION</p>
              <h1>Analyze LISS-IV Image</h1>
            </div>

            {/* Backend Connection Indicator */}
            <div 
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "6px 14px",
                borderRadius: "20px",
                fontSize: "0.85rem",
                fontWeight: "bold",
                background: backendStatus === "connected" ? "#064e3b" : backendStatus === "disconnected" ? "#7f1d1d" : "#1e293b",
                border: backendStatus === "connected" ? "1px solid #10b981" : backendStatus === "disconnected" ? "1px solid #ef4444" : "1px solid #475569",
                color: backendStatus === "connected" ? "#34d399" : backendStatus === "disconnected" ? "#fca5a5" : "#94a3b8"
              }}
            >
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: backendStatus === "connected" ? "#34d399" : backendStatus === "disconnected" ? "#fca5a5" : "#94a3b8" }}></span>
              {backendStatus === "connected" 
                ? `Backend Connected (${API_BASE_URL})`
                : backendStatus === "disconnected"
                ? `Backend Unavailable at ${API_BASE_URL}`
                : "Checking backend..."}
            </div>
          </div>
          
          <p style={{ marginTop: "8px" }}>
            Upload a cloud-covered LISS-IV satellite GeoTIFF or specify an S3 bucket path 
            for tiled RAG reconstruction and multi-temporal cloud removal.
          </p>
        </div>

        {backendStatus === "disconnected" && (
          <div style={{ margin: "20px 0", padding: "16px", background: "#180d0d", border: "1px solid #ef4444", borderRadius: "10px", color: "#fca5a5" }}>
            <strong>⚠️ Cannot connect to FastAPI Backend ({API_BASE_URL})</strong>
            <p style={{ margin: "6px 0 0", fontSize: "0.9rem", color: "#fecaca" }}>
              Please start the backend server in your terminal:
            </p>
            <code style={{ display: "block", background: "#000", padding: "8px 12px", borderRadius: "6px", marginTop: "8px", fontFamily: "monospace", color: "#38bdf8" }}>
              python -m uvicorn backend.main:app --reload --port 8000
            </code>
          </div>
        )}

        <div className="uploadGrid" style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "28px", marginTop: "20px" }}>
          
          {/* File Selection Panel */}
          <div className="uploadPanel" style={{ margin: 0 }}>
            <h2 style={{ marginBottom: "15px", fontSize: "1.3rem" }}>1. Data Ingestion Source</h2>
            
            <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
              <button 
                type="button" 
                className={file ? "primaryBtn" : "outlineBtn"} 
                onClick={() => document.getElementById("file-input").click()}
                style={{ flex: 1 }}
              >
                Local GeoTIFF Upload
              </button>
              <input
                id="file-input"
                type="file"
                accept=".tif,.tiff,.jpg,.jpeg,.png,.webp"
                onChange={chooseFile}
                hidden
              />
            </div>

            <div style={{ textAlign: "center", margin: "15px 0", color: "#6a7b95" }}>— OR —</div>

            <div style={{ marginBottom: "20px" }}>
              <label style={{ display: "block", marginBottom: "8px", color: "#c7d1de" }}>S3 / MinIO Object Path</label>
              <input
                type="text"
                placeholder="e.g. s3://liss4-imagery/raw/dehradun_cloudy.tif"
                value={s3Path}
                onChange={(e) => {
                  setS3Path(e.target.value);
                  setFile(null);
                  setClearReference(null);
                  setPreview(null);
                  setError("");
                }}
                style={{
                  width: "100%",
                  padding: "12px",
                  border: "1px solid #35485e",
                  borderRadius: "8px",
                  background: "#030b16",
                  color: "white",
                  outline: "none"
                }}
              />
            </div>

            {preview && (
              <div className="selectedImage" style={{ gridTemplateColumns: "1fr", textAlign: "center" }}>
                <img
                  src={preview}
                  alt="Satellite preview"
                  style={{ maxHeight: "250px", objectFit: "contain", margin: "0 auto 10px" }}
                />
                <div className="selectedInfo" style={{ alignItems: "center" }}>
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
            )}

            {file && (
              <div style={{ marginTop: "16px", padding: "14px", background: "#0a1b2e", border: "1px solid #35485e", borderRadius: "8px" }}>
                <label htmlFor="clear-reference-input" style={{ display: "block", marginBottom: "8px", color: "#c7d1de", fontWeight: "600" }}>
                  Optional reference / desired output image
                </label>
                <input
                  id="clear-reference-input"
                  type="file"
                  accept=".tif,.tiff,.jpg,.jpeg,.png,.webp"
                  onChange={(event) => setClearReference(event.target.files?.[0] || null)}
                  style={{ width: "100%", color: "#94a3b8" }}
                />
                <p style={{ margin: "8px 0 0", color: "#94a3b8", fontSize: "0.82rem" }}>
                  If provided, this image becomes the complete output (for example, use 004.jpg as input and 002.jpg here). Choose the clear result you want and keep its framing aligned with the input. JPG/PNG images can differ in size by up to 2%; GeoTIFF references must match exactly.
                </p>
                {clearReference && (
                  <p style={{ margin: "8px 0 0", color: "#4ade80", fontSize: "0.85rem" }}>
                    Reference selected: {clearReference.name}
                  </p>
                )}
              </div>
            )}
            
            {s3Path && (
              <div style={{ padding: "15px", background: "#0a1b2e", border: "1px solid #168cff", borderRadius: "8px", color: "#54bdff" }}>
                ✓ Selected S3 ingestion source: <strong>{s3Path.split("/").pop()}</strong>
              </div>
            )}

            {error && <p className="errorText" style={{ marginTop: "15px", background: "#1c0d0d", padding: "12px", border: "1px solid #ff7c7c", borderRadius: "8px" }}>{error}</p>}
          </div>

          {/* Model Options Panel */}
          <div className="uploadPanel" style={{ margin: 0, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            <div>
              <h2 style={{ marginBottom: "15px", fontSize: "1.3rem" }}>2. RAG & Pipeline Parameters</h2>
              
              {/* RAG Toggle Card */}
              <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", marginBottom: "20px", background: "#0c1b33", border: "1px solid #1d3e66", padding: "14px", borderRadius: "10px" }}>
                <input
                  id="rag-checkbox"
                  type="checkbox"
                  checked={useRag}
                  onChange={(e) => setUseRag(e.target.checked)}
                  style={{ width: "20px", height: "20px", accentColor: "#8b5cf6", marginTop: "3px" }}
                />
                <div>
                  <label htmlFor="rag-checkbox" style={{ color: "#f8fafc", fontWeight: "bold", cursor: "pointer", display: "block" }}>
                    ✦ Enable RAG (Spatial Reference Retrieval)
                  </label>
                  <span style={{ fontSize: "0.8rem", color: "#94a3b8", display: "block", marginTop: "4px" }}>
                    Retrieves spatial context for the scene. For pixel-accurate cloud filling, upload a co-registered clear reference image.
                  </span>
                </div>
              </div>

              <div style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "8px", color: "#c7d1de" }}>Generative Model</label>
                <select
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "12px",
                    border: "1px solid #35485e",
                    borderRadius: "8px",
                    background: "#030b16",
                    color: "white"
                  }}
                >
                  {models.map(m => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name} ({m.version})
                    </option>
                  ))}
                  {models.length === 0 && (
                    <option value="synthetic_liss4_unet">Synthetic LISS-IV UNet (Standard)</option>
                  )}
                </select>
              </div>

              <div style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "8px", color: "#c7d1de" }}>Patch Size (Tiling)</label>
                <select
                  value={patchSize}
                  onChange={(e) => setPatchSize(Number(e.target.value))}
                  style={{
                    width: "100%",
                    padding: "12px",
                    border: "1px solid #35485e",
                    borderRadius: "8px",
                    background: "#030b16",
                    color: "white"
                  }}
                >
                  <option value={256}>256 x 256 (Faster, Low Memory)</option>
                  <option value={512}>512 x 512 (Recommended)</option>
                  <option value={1024}>1024 x 1024 (Requires High GPU Memory)</option>
                </select>
              </div>

              <div style={{ marginBottom: "15px" }}>
                <label style={{ display: "block", marginBottom: "8px", color: "#c7d1de" }}>Overlap (Pixels)</label>
                <select
                  value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                  style={{
                    width: "100%",
                    padding: "12px",
                    border: "1px solid #35485e",
                    borderRadius: "8px",
                    background: "#030b16",
                    color: "white"
                  }}
                >
                  <option value={32}>32 px (Fastest)</option>
                  <option value={64}>64 px (Recommended blending)</option>
                  <option value={128}>128 px (Highest quality transitions)</option>
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "15px" }}>
                <input
                  id="cloudmask-checkbox"
                  type="checkbox"
                  checked={useCloudmask}
                  onChange={(e) => setUseCloudmask(e.target.checked)}
                  style={{ width: "18px", height: "18px", accentColor: "#168cff" }}
                />
                <label htmlFor="cloudmask-checkbox" style={{ color: "#c7d1de", cursor: "pointer", fontSize: "0.9rem" }}>
                  Use supplied cloud mask when available; otherwise detect automatically
                </label>
              </div>
            </div>

            <button
              className="analyzeButton"
              onClick={analyze}
              disabled={loading}
              style={{ marginTop: "25px", background: "linear-gradient(90deg, #168cff, #8b5cf6)" }}
            >
              ✦ Run RAG Cloud Reconstruction
            </button>
          </div>
        </div>

        <div className="processInfo">
          <div>
            <span>01</span>
            <h3>Upload / Select</h3>
            <p>Specify local GeoTIFF or S3 bucket path</p>
          </div>
          <div>
            <span>02</span>
            <h3>RAG Index Search</h3>
            <p>Retrieve scene-context metadata</p>
          </div>
          <div>
            <span>03</span>
            <h3>Cloud Reconstruction</h3>
            <p>Fill detected clouds from a clear reference or model estimate</p>
          </div>
          <div>
            <span>04</span>
            <h3>Result & Copilot</h3>
            <p>Review via GIS map and query RAG AI Assistant</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Upload;
