// Results Page with Interactive Leaflet Map, Overlay Slider, and RAG Spatial Context
import { useEffect, useState, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import L from "leaflet";
import api, { API_BASE_URL } from "../services/api";
import Sidebar from "../components/Sidebar";

function Results() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get("job_id");

  const [result, setResult] = useState(null);
  const [ragContext, setRagContext] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sliderVal, setSliderVal] = useState(50); // 0 (Cloudy) to 100 (Reconstructed)
  const [maskOpacity, setMaskOpacity] = useState(60); // 0 to 100
  const [showMask, setShowMask] = useState(false);
  const [activeTab, setActiveTab] = useState("map"); // 'map', 'comparison', 'rag', 'details'
  const [zoomFocus, setZoomFocus] = useState({ x: 0.5, y: 0.5 });

  const mapRef = useRef(null);
  const layersRef = useRef({ input: null, output: null, mask: null });
  const mapContainerId = `map-container-${jobId || "default"}`;
  const zoomSize = 0.24;

  // Poll job status and fetch RAG context
  useEffect(() => {
    if (!jobId) {
      setError("No Job ID provided in URL parameters.");
      setLoading(false);
      return;
    }

    let intervalId = null;

    const fetchStatus = async () => {
      try {
        const response = await api.get(`/api/v1/jobs/${jobId}`);
        const data = response.data;
        setResult(data);

        if (data.status === "Completed") {
          setLoading(false);
          clearInterval(intervalId);
          // Fetch RAG detailed context for this job
          api.get(`/api/v1/rag/context/${jobId}`)
            .then(ragRes => setRagContext(ragRes.data))
            .catch(err => console.error("Could not fetch RAG context:", err));
        } else if (data.status === "Failed") {
          setLoading(false);
          setError(data.error_message || "Reconstruction pipeline failed.");
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error(err);
        setError("Failed to fetch job details from server.");
        setLoading(false);
        clearInterval(intervalId);
      }
    };

    fetchStatus();
    intervalId = setInterval(fetchStatus, 2000);

    return () => clearInterval(intervalId);
  }, [jobId]);

  useEffect(() => {
    if (result?.status === "Completed" && !result.bounds) {
      setActiveTab("comparison");
    }
  }, [result?.status, result?.bounds]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (loading || error || !result || result.status !== "Completed") return;
    if (!result.bounds) return;

    // Parse bounds from min_lon,min_lat,max_lon,max_lat
    const coords = result.bounds.split(",").map(Number);
    if (coords.length < 4) return;

    const west = coords[0];
    const south = coords[1];
    const east = coords[2];
    const north = coords[3];

    const bounds = [
      [south, west],
      [north, east]
    ];
    const center = [(south + north) / 2, (west + east) / 2];

    const map = L.map(mapContainerId, {
      center: center,
      zoom: 14,
      zoomControl: true
    });
    mapRef.current = map;

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20
    }).addTo(map);

    const previewBase = `${API_BASE_URL}/api/v1/preview/${jobId}?bbox=${west},${south},${east},${north}`;
    const inputUrl = `${previewBase}&layer=input`;
    const outputUrl = `${previewBase}&layer=output`;
    const maskUrl = `${previewBase}&layer=mask`;

    const inputLayer = L.imageOverlay(inputUrl, bounds, {
      opacity: 0.5,
      interactive: true
    }).addTo(map);

    const outputLayer = L.imageOverlay(outputUrl, bounds, {
      opacity: 0.5,
      interactive: true
    }).addTo(map);

    const maskLayer = L.imageOverlay(maskUrl, bounds, {
      opacity: 0.0,
      interactive: true
    }).addTo(map);

    layersRef.current = {
      input: inputLayer,
      output: outputLayer,
      mask: maskLayer
    };

    map.fitBounds(bounds);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [loading, error, result]);

  // Sync sliders to Map Opacity
  useEffect(() => {
    const layers = layersRef.current;
    if (!layers.input || !layers.output || !layers.mask) return;

    const outOpacity = sliderVal / 100;
    const inOpacity = (100 - sliderVal) / 100;
    const maskOp = showMask ? maskOpacity / 100 : 0.0;

    layers.input.setOpacity(inOpacity);
    layers.output.setOpacity(outOpacity);
    layers.mask.setOpacity(maskOp);
  }, [sliderVal, maskOpacity, showMask]);

  const handleDownload = () => {
    if (!result || !result.generated_image) return;
    const targetUrl = result.generated_image.startsWith("http") ? result.generated_image : `${API_BASE_URL}${result.generated_image}`;
    window.open(targetUrl, "_blank");
  };

  const handleDownloadMask = () => {
    if (!result || !result.cloud_mask_image) return;
    const targetUrl = result.cloud_mask_image.startsWith("http") ? result.cloud_mask_image : `${API_BASE_URL}${result.cloud_mask_image}`;
    window.open(targetUrl, "_blank");
  };

  if (loading && result && (result.status === "Pending" || result.status === "Running")) {
    return (
      <div className="dashboardLayout">
        <Sidebar />
        <main className="dashboardContent" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", minHeight: "80vh" }}>
          <div className="spinner"></div>
          <h2 style={{ marginTop: "15px" }}>Processing Cloud Removal</h2>
          <p style={{ color: "#a1b2c9", marginTop: "8px" }}>
            Status: <span style={{ color: "#54bdff", fontWeight: "bold" }}>{result.status}</span>
          </p>
          
          <div style={{ width: "100%", maxWidth: "450px", height: "8px", background: "#112235", borderRadius: "10px", marginTop: "20px", overflow: "hidden" }}>
            <div style={{ width: `${result.progress || 0}%`, height: "100%", background: "linear-gradient(90deg, #168cff, #8b5cf6)", transition: "width 0.3s ease-in-out" }}></div>
          </div>
          <strong style={{ display: "block", marginTop: "10px", fontSize: "1.2rem", color: "#8b5cf6" }}>{result.progress || 0}% Completed</strong>
          
          <p style={{ color: "#74889f", fontSize: "0.9rem", marginTop: "15px" }}>
            Executing windowed tiling ({result.model_name || "UNet"}) with RAG spatial reference retrieval on GeoTIFF tiles.
          </p>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboardLayout">
        <Sidebar />
        <main className="dashboardContent">
          <div className="emptyState" style={{ background: "#0c1524", border: "1px solid #ff7c7c", borderRadius: "12px", padding: "40px" }}>
            <span style={{ fontSize: "3rem" }}>⚠️</span>
            <h2 style={{ color: "#ff7c7c" }}>Processing Failed</h2>
            <p style={{ maxWidth: "500px", color: "#b9cadf", margin: "10px 0" }}>{error}</p>
            <button className="primaryBtn" onClick={() => navigate("/upload")} style={{ marginTop: "15px" }}>
              Try Again
            </button>
          </div>
        </main>
      </div>
    );
  }

  const getAssetUrl = (path) => (
    path ? (path.startsWith("http") ? path : `${API_BASE_URL}${path}`) : null
  );
  const inputImageUrl = getAssetUrl(result?.original_image);
  const outputImageUrl = getAssetUrl(result?.generated_image);
  const zoomPosition = {
    x: ((zoomFocus.x - zoomSize / 2) / (1 - zoomSize)) * 100,
    y: ((zoomFocus.y - zoomSize / 2) / (1 - zoomSize)) * 100,
  };
  const getZoomStyle = (imageUrl) => ({
    backgroundImage: `url("${imageUrl}")`,
    backgroundSize: `${100 / zoomSize}% ${100 / zoomSize}%`,
    backgroundPosition: `${zoomPosition.x}% ${zoomPosition.y}%`,
  });
  const handleZoomSelect = (event) => {
    const image = event.currentTarget;
    if (!image.naturalWidth || !image.naturalHeight) return;

    const bounds = image.getBoundingClientRect();
    const scale = Math.min(
      bounds.width / image.naturalWidth,
      bounds.height / image.naturalHeight
    );
    const renderedWidth = image.naturalWidth * scale;
    const renderedHeight = image.naturalHeight * scale;
    const offsetX = (bounds.width - renderedWidth) / 2;
    const offsetY = (bounds.height - renderedHeight) / 2;
    const x = (event.clientX - bounds.left - offsetX) / renderedWidth;
    const y = (event.clientY - bounds.top - offsetY) / renderedHeight;

    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    setZoomFocus({
      x: Math.max(zoomSize / 2, Math.min(1 - zoomSize / 2, x)),
      y: Math.max(zoomSize / 2, Math.min(1 - zoomSize / 2, y)),
    });
  };
  const evaluationMetrics = [
    { label: "PSNR (dB) ↑", value: result?.psnr, digits: 2 },
    { label: "SSIM ↑", value: result?.ssim, digits: 4 },
    { label: "RMSE ↓", value: result?.rmse, digits: 4 },
    { label: "SAM (°) ↓", value: result?.sam, digits: 2 },
    { label: "ERGAS ↓", value: result?.ergas, digits: 2 },
  ];

  return (
    <div className="dashboardLayout">
      <Sidebar />

      <main className="dashboardContent">
        <div className="resultHeader">
          <div>
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <div className="successBadge" style={{ background: "#0a2e1c", border: "1px solid #16a34a", color: "#4ade80", padding: "6px 12px", borderRadius: "20px", fontSize: "0.85rem" }}>
                ✓ Reconstruction Completed in {result?.processing_time?.toFixed(2)}s
              </div>
              <div style={{ background: "rgba(139, 92, 246, 0.2)", border: "1px solid #8b5cf6", color: "#c084fc", padding: "6px 12px", borderRadius: "20px", fontSize: "0.85rem", fontWeight: "bold" }}>
                ✦ RAG Enhanced
              </div>
            </div>
            <h1 style={{ marginTop: "10px" }}>Cloud Removal Result</h1>
            <p>Compare original cloud coverage with the AI reconstructed output.</p>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <button className="outlineBtn" onClick={() => navigate(`/rag-assistant?job_id=${jobId}`)}>
              ✦ Ask RAG Copilot
            </button>
            <button className="primaryBtn" onClick={handleDownload}>
              ↓ Download GeoTIFF
            </button>
          </div>
        </div>

        {/* Tab Selector */}
        <div style={{ display: "flex", borderBottom: "1px solid #233044", gap: "20px", marginTop: "25px", marginBottom: "20px" }}>
          <button 
            type="button"
            onClick={() => setActiveTab("map")}
            style={{ 
              background: "transparent", 
              border: "none", 
              borderBottom: activeTab === "map" ? "2px solid #168cff" : "none", 
              color: activeTab === "map" ? "#168cff" : "#8e9daf", 
              padding: "10px 15px", 
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            Interactive GIS Map
          </button>
          <button 
            type="button"
            onClick={() => setActiveTab("comparison")}
            style={{ 
              background: "transparent", 
              border: "none", 
              borderBottom: activeTab === "comparison" ? "2px solid #168cff" : "none", 
              color: activeTab === "comparison" ? "#168cff" : "#8e9daf", 
              padding: "10px 15px", 
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            📊 3-Panel Visual Comparison & Metrics
          </button>
          <button 
            type="button"
            onClick={() => setActiveTab("rag")}
            style={{ 
              background: "transparent", 
              border: "none", 
              borderBottom: activeTab === "rag" ? "2px solid #8b5cf6" : "none", 
              color: activeTab === "rag" ? "#c084fc" : "#8e9daf", 
              padding: "10px 15px", 
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            ✦ RAG Retrieval Context
          </button>
          <button 
            type="button"
            onClick={() => setActiveTab("details")}
            style={{ 
              background: "transparent", 
              border: "none", 
              borderBottom: activeTab === "details" ? "2px solid #168cff" : "none", 
              color: activeTab === "details" ? "#168cff" : "#8e9daf", 
              padding: "10px 15px", 
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            Job Metadata & QC
          </button>
        </div>

        {activeTab === "map" ? (
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 0.6fr", gap: "28px" }}>
            {/* Map Frame */}
            <div>
              {result?.bounds ? (
                <>
                  <div
                    id={mapContainerId}
                    style={{
                      height: "550px",
                      width: "100%",
                      borderRadius: "14px",
                      border: "1px solid #2b3a4f",
                      boxShadow: "0 10px 30px rgba(0,0,0,0.4)"
                    }}
                  ></div>
                  <p style={{ color: "#71859c", fontSize: "0.85rem", marginTop: "8px", fontStyle: "italic" }}>
                    * RGB uploads retain their displayed colors; multiband GeoTIFFs use an NIR/Red/Green false-color composite.
                  </p>
                </>
              ) : (
                <div style={{ minHeight: "250px", display: "grid", placeContent: "center", textAlign: "center", padding: "30px", borderRadius: "14px", border: "1px solid #2b3a4f", background: "#0c1524" }}>
                  <h3>Map preview unavailable</h3>
                  <p style={{ color: "#94a3b8", marginTop: "8px" }}>
                    This image has no georeferencing. The reconstructed GeoTIFF preserves the pixels but does not invent a location.
                  </p>
                </div>
              )}
            </div>

            {/* Slider Map Controls */}
            <div className="uploadPanel" style={{ margin: 0, padding: "24px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div>
                <h3 style={{ borderBottom: "1px solid #233044", paddingBottom: "10px", marginBottom: "20px" }}>Visual Controls</h3>
                
                {/* Before After Slider */}
                <div style={{ marginBottom: "25px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                    <span style={{ color: "#ff7c7c" }}>Before (Cloudy)</span>
                    <span style={{ color: "#4ade80" }}>After (Clean)</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sliderVal}
                    onChange={(e) => setSliderVal(Number(e.target.value))}
                    style={{ width: "100%", accentColor: "#168cff", cursor: "pointer" }}
                  />
                  <div style={{ textAlign: "center", fontSize: "0.85rem", color: "#8e9daf", marginTop: "5px" }}>
                    Slider: {sliderVal}% Reconstructed
                  </div>
                </div>

                {/* Cloud Mask Toggle */}
                <div style={{ borderTop: "1px solid #233044", paddingTop: "20px", marginBottom: "20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "15px" }}>
                    <input
                      id="mask-toggle"
                      type="checkbox"
                      checked={showMask}
                      onChange={(e) => setShowMask(e.target.checked)}
                      style={{ width: "18px", height: "18px", accentColor: "#168cff" }}
                    />
                    <label htmlFor="mask-toggle" style={{ fontWeight: "600", color: "#c7d1de", cursor: "pointer" }}>
                      Overlay Cloud Mask
                    </label>
                  </div>
                  
                  {showMask && (
                    <div>
                      <label style={{ display: "block", fontSize: "0.85rem", color: "#8e9daf", marginBottom: "5px" }}>Mask Opacity</label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={maskOpacity}
                        onChange={(e) => setMaskOpacity(Number(e.target.value))}
                        style={{ width: "100%", accentColor: "#ef4444" }}
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Quick Actions */}
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                <button className="primaryBtn" onClick={handleDownload} style={{ width: "100%" }}>
                  ↓ Reconstructed GeoTIFF
                </button>
                {result?.cloud_mask_image && (
                  <button className="outlineBtn" onClick={handleDownloadMask} style={{ width: "100%" }}>
                    ↓ Extracted Cloud Mask
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : activeTab === "comparison" ? (
          <section className="evaluation-board" aria-label="Cloud removal visual comparison">
            <div className="evaluation-gallery">
              <article className="evaluation-image-card input-image-card">
                <h2>Input: Cloudy LISS-IV Image</h2>
                <div
                  className="evaluation-image-frame"
                  style={result?.width && result?.height
                    ? { aspectRatio: `${result.width} / ${result.height}` }
                    : undefined}
                >
                  {inputImageUrl ? (
                    <img src={inputImageUrl} alt="Cloudy input satellite image" />
                  ) : (
                    <p>Input preview is unavailable.</p>
                  )}
                </div>
              </article>

              <article className="evaluation-image-card output-image-card">
                <h2>Reconstructed Cloud-Free Image</h2>
                <div
                  className="evaluation-image-frame output-image-frame"
                  style={result?.width && result?.height
                    ? { aspectRatio: `${result.width} / ${result.height}` }
                    : undefined}
                >
                  {outputImageUrl ? (
                    <>
                      <img
                        src={outputImageUrl}
                        alt="Reconstructed cloud-free satellite image; click to select a zoom region"
                        onClick={handleZoomSelect}
                      />
                      <span
                        className="zoom-selection"
                        style={{
                          left: `${(zoomFocus.x - zoomSize / 2) * 100}%`,
                          top: `${(zoomFocus.y - zoomSize / 2) * 100}%`,
                          width: `${zoomSize * 100}%`,
                          height: `${zoomSize * 100}%`,
                        }}
                        aria-hidden="true"
                      />
                    </>
                  ) : (
                    <p>Reconstructed preview is unavailable.</p>
                  )}
                </div>
                <p className="zoom-instruction">Click the image to inspect a region.</p>
              </article>

              <aside className="zoom-comparison-card">
                <h2>Zoomed Region (Detail Comparison)</h2>
                <div className="zoom-comparison-images">
                  <div className="zoom-comparison-item">
                    {inputImageUrl && (
                      <div
                        className="zoom-crop"
                        style={getZoomStyle(inputImageUrl)}
                        role="img"
                        aria-label="Zoomed cloudy input region"
                      />
                    )}
                    <strong>Cloudy (Input)</strong>
                  </div>
                  <div className="zoom-comparison-item">
                    {outputImageUrl && (
                      <div
                        className="zoom-crop"
                        style={getZoomStyle(outputImageUrl)}
                        role="img"
                        aria-label="Zoomed reconstructed region"
                      />
                    )}
                    <strong>Reconstructed (Cloud-Free)</strong>
                  </div>
                </div>
              </aside>
            </div>

            <section className="evaluation-metrics" aria-label="Evaluation metrics">
              <h2>Evaluation Metrics (↑ Higher is Better, ↓ Lower is Better)</h2>
              <div className="evaluation-metric-grid">
                {evaluationMetrics.map((metric) => (
                  <div className="evaluation-metric" key={metric.label}>
                    <strong>{metric.label}</strong>
                    <span>
                      {metric.value == null ? "N/A" : metric.value.toFixed(metric.digits)}
                    </span>
                  </div>
                ))}
              </div>
              {result?.psnr == null && (
                <p className="evaluation-metrics-note">
                  Quality metrics require an aligned clear reference image. They are not estimated from the cloudy input.
                </p>
              )}
            </section>
          </section>
        ) : activeTab === "rag" ? (
          /* RAG Retrieval Context View */
          <div className="uploadPanel" style={{ margin: 0, maxWidth: "100%", padding: "30px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px", borderBottom: "1px solid #233044", paddingBottom: "15px" }}>
              <div>
                <h2 style={{ fontSize: "1.4rem", color: "#c084fc", margin: 0 }}>✦ RAG Retrieval-Augmented Context Details</h2>
                <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginTop: "4px" }}>
                  Spatial catalog metadata is shown for context; it is not a pixel-aligned reconstruction reference.
                </p>
              </div>

              <button className="primaryBtn" onClick={() => navigate(`/rag-assistant?job_id=${jobId}`)} style={{ background: "linear-gradient(90deg, #168cff, #8b5cf6)" }}>
                ✦ Ask RAG Copilot About This Scene
              </button>
            </div>

            {ragContext?.spatial_reference ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "25px" }}>
                {/* Spatial Patch Match Card */}
                <div style={{ background: "#071220", border: "1px solid #1a3250", borderRadius: "12px", padding: "20px" }}>
                  <h3 style={{ color: "#38bdf8", marginBottom: "12px" }}>🗺️ Primary Spatial Reference Match</h3>
                  <div style={{ marginBottom: "10px", color: "#e2e8f0" }}>
                    <strong>Patch ID:</strong> `{ragContext.spatial_reference.primary_patch_id}`
                  </div>
                  <div style={{ marginBottom: "10px", color: "#e2e8f0" }}>
                    <strong>Region:</strong> {ragContext.spatial_reference.retrieved_patches[0]?.region_name}
                  </div>
                  <div style={{ marginBottom: "10px", color: "#e2e8f0" }}>
                    <strong>Spatial Similarity Score:</strong>{" "}
                    <span style={{ color: "#4ade80", fontWeight: "bold" }}>
                      {(ragContext.spatial_reference.best_similarity_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ marginBottom: "10px", color: "#e2e8f0" }}>
                    <strong>Acquisition Date:</strong> {ragContext.spatial_reference.retrieved_patches[0]?.acquisition_date}
                  </div>
                  <div style={{ marginBottom: "10px", color: "#e2e8f0" }}>
                    <strong>Satellite Sensor:</strong> {ragContext.spatial_reference.retrieved_patches[0]?.satellite}
                  </div>
                </div>

                {/* RAG Process Summary Card */}
                <div style={{ background: "#071220", border: "1px solid #1a3250", borderRadius: "12px", padding: "20px" }}>
                  <h3 style={{ color: "#c084fc", marginBottom: "12px" }}>⚡ RAG Augmentation Summary</h3>
                  <p style={{ color: "#cbd5e1", lineHeight: "1.6", fontSize: "0.95rem" }}>
                    {ragContext.rag_context}
                  </p>

                  <div style={{ marginTop: "15px", padding: "12px", background: "#0d223a", border: "1px solid #1e3a5f", borderRadius: "8px", fontSize: "0.85rem", color: "#38bdf8" }}>
                    Spatial retrieval provides scene context only. Pixel-level cloud filling uses the optional aligned clear reference when one was uploaded.
                  </div>
                </div>
              </div>
            ) : ragContext ? (
              <p>No geospatial bounds are available for this image, so spatial catalog lookup was skipped.</p>
            ) : (
              <p>Loading RAG spatial context...</p>
            )}
          </div>
        ) : (
          /* Details Tab */
          <div className="uploadPanel" style={{ margin: 0, maxWidth: "100%" }}>
            <h2 style={{ marginBottom: "20px", borderBottom: "1px solid #233044", paddingBottom: "10px" }}>Geospatial Execution Summary</h2>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "25px" }}>
              <div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Job Identifier</td>
                      <td style={{ padding: "12px 0", textAlign: "right", fontWeight: "bold" }}>#{result.id}</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Filename</td>
                      <td style={{ padding: "12px 0", textAlign: "right", wordBreak: "break-all" }}>{result.filename}</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Coordinate System (CRS)</td>
                      <td style={{ padding: "12px 0", textAlign: "right", color: "#54bdff" }}>{result.crs || "N/A"}</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Resolution Shape (W x H)</td>
                      <td style={{ padding: "12px 0", textAlign: "right" }}>{result.width && result.height ? `${result.width} x ${result.height} pixels` : "N/A"}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <tbody>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Cloud Cover percentage</td>
                      <td style={{ padding: "12px 0", textAlign: "right", color: "#ff7c7c", fontWeight: "bold" }}>
                        {result.cloud_cover_percentage !== null ? `${result.cloud_cover_percentage.toFixed(2)}%` : "N/A"}
                      </td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>Processing Time</td>
                      <td style={{ padding: "12px 0", textAlign: "right", color: "#4ade80" }}>
                        {result.processing_time !== null ? `${result.processing_time.toFixed(2)} seconds` : "N/A"}
                      </td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>AI Reconstruction Model</td>
                      <td style={{ padding: "12px 0", textAlign: "right" }}>{result.model_name}</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid #162438" }}>
                      <td style={{ padding: "12px 0", color: "#8e9daf" }}>RAG Mode</td>
                      <td style={{ padding: "12px 0", textAlign: "right", color: "#c084fc", fontWeight: "bold" }}>
                        {result.use_rag ? "Enabled (Spatial Reference)" : "Disabled"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default Results;