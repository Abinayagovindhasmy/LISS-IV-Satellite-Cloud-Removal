// History page.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import api, { API_BASE_URL } from "../services/api";

function History() {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await api.get("/history/");
      setHistory(response.data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCardClick = (id) => {
    navigate(`/results?job_id=${id}`);
  };

  const getFullImgUrl = (path) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  };

  return (
    <div className="dashboardLayout">
      <Sidebar />

      <main className="dashboardContent">
        <div className="historyHeader">
          <p className="smallTitle">ACTIVITY</p>
          <h1>Prediction History</h1>
          <p>
            View and explore previous LISS-IV satellite image reconstruction records. 
            Click any record to inspect on the interactive GIS map.
          </p>
        </div>

        <div className="historyPanel">
          {loading ? (
            <p>Loading history...</p>
          ) : history.length === 0 ? (
            <div className="emptyState">
              <h3>No prediction history</h3>
              <p>Your processed images will appear here.</p>
              <button className="primaryBtn" onClick={() => navigate("/upload")} style={{ marginTop: "15px" }}>
                Upload First Image
              </button>
            </div>
          ) : (
            <div className="historyGrid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginTop: "20px" }}>
              {history.map((item) => (
                <div
                  className="historyCard"
                  key={item.id}
                  onClick={() => handleCardClick(item.id)}
                  style={{ 
                    cursor: "pointer", 
                    transition: "transform 0.2s, box-shadow 0.2s",
                    border: "1px solid #2d4056",
                    borderRadius: "10px",
                    overflow: "hidden",
                    background: "#050c16"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-3px)";
                    e.currentTarget.style.boxShadow = "0 8px 24px rgba(0,0,0,0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "none";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  <div className="historyImages" style={{ display: "flex", height: "140px", position: "relative", background: "#020711" }}>
                    <img
                      src={getFullImgUrl(item.original_image)}
                      alt="Original"
                      style={{ width: "50%", height: "100%", objectFit: "cover" }}
                    />
                    {item.status === "Completed" && item.generated_image ? (
                      <img
                        src={getFullImgUrl(item.generated_image)}
                        alt="Generated"
                        style={{ width: "50%", height: "100%", objectFit: "cover", borderLeft: "1px solid #2d4056" }}
                      />
                    ) : (
                      <div style={{ width: "50%", height: "100%", display: "flex", justifyContent: "center", alignItems: "center", background: "#0b1625", color: "#6a7b95", fontSize: "0.85rem", borderLeft: "1px solid #2d4056" }}>
                        {item.status}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "15px" }}>
                    <h3 style={{ fontSize: "1.1rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", margin: 0 }}>
                      {item.filename}
                    </h3>
                    
                    <div className="historyMeta" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "10px" }}>
                      <span style={{ fontSize: "0.85rem", color: "#8e9daf" }}>{item.model_name || "UNet"}</span>
                      <span 
                        className="status" 
                        style={{ 
                          fontSize: "0.8rem", 
                          padding: "3px 8px", 
                          borderRadius: "12px", 
                          fontWeight: "bold",
                          background: item.status === "Completed" ? "#0a2e1c" : item.status === "Failed" ? "#3b1e1e" : "#2e2e1c",
                          color: item.status === "Completed" ? "#4ade80" : item.status === "Failed" ? "#ff7c7c" : "#eab308"
                        }}
                      >
                        {item.status}
                      </span>
                    </div>
                    
                    <div style={{ marginTop: "12px", fontSize: "0.75rem", color: "#6a7b95", textAlign: "right" }}>
                      {new Date(item.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default History;