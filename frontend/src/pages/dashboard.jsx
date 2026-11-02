// Dashboard page.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import api, { API_BASE_URL } from "../services/api";

function Dashboard() {
  const navigate = useNavigate();

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const user = JSON.parse(
    localStorage.getItem("user") || "{}"
  );

  useEffect(() => {
    api
      .get("/history/")
      .then((response) => {
        setHistory(response.data);
      })
      .catch((error) => {
        console.error(error);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const handleCardClick = (id) => {
    navigate(`/results?job_id=${id}`);
  };

  const getFullImgUrl = (path) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  };

  const completedJobs = history.filter(
    (item) => item.status === "Completed"
  );

  return (
    <div className="dashboardLayout">
      <Sidebar />

      <main className="dashboardContent">
        <div className="dashboardHeader">
          <div>
            <p className="smallTitle">CLOUDVISION AI</p>
            <h1>Welcome, {user.name || "Analyst"}</h1>
            <p>
              Analyze and manage your LISS-IV satellite image cloud-removal and 
              reconstruction results.
            </p>
          </div>

          <button
            className="primaryBtn"
            onClick={() => navigate("/upload")}
          >
            + New Analysis
          </button>
        </div>

        <div className="dashboardStats">
          <div className="statCard">
            <span>Total Analysis Jobs</span>
            <strong>{history.length}</strong>
          </div>

          <div className="statCard">
            <span>Reconstruction Completed</span>
            <strong>{completedJobs.length}</strong>
          </div>

          <div className="statCard">
            <span>Core Model Registry</span>
            <strong className="smallStat">Synthetic UNet (v2.0)</strong>
          </div>
        </div>

        <div className="dashboardPanel">
          <div className="panelHeader">
            <h2>Recent Activity</h2>
            <button
              className="textButton"
              onClick={() => navigate("/history")}
              style={{ fontWeight: "600" }}
            >
              View All Activity →
            </button>
          </div>

          {loading ? (
            <p style={{ marginTop: "20px" }}>Loading records...</p>
          ) : history.length === 0 ? (
            <div className="emptyState">
              <div>☁</div>
              <h3>No predictions yet</h3>
              <p>Upload your first LISS-IV satellite GeoTIFF to start.</p>
              <button
                className="primaryBtn"
                onClick={() => navigate("/upload")}
                style={{ marginTop: "10px" }}
              >
                Analyze Image
              </button>
            </div>
          ) : (
            <div className="recentGrid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginTop: "20px" }}>
              {history.slice(0, 3).map((item) => (
                <div
                  className="recentCard"
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
                  <img
                    src={getFullImgUrl(item.status === "Completed" ? item.generated_image : item.original_image)}
                    alt={item.filename}
                    style={{ width: "100%", height: "160px", objectFit: "cover" }}
                  />

                  <div style={{ padding: "14px" }}>
                    <strong style={{ fontSize: "1.05rem", display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {item.filename}
                    </strong>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px" }}>
                      <span style={{ fontSize: "0.8rem", color: "#8e9daf" }}>#{item.id}</span>
                      <span 
                        style={{ 
                          fontSize: "0.8rem", 
                          fontWeight: "bold",
                          color: item.status === "Completed" ? "#4ade80" : item.status === "Failed" ? "#ff7c7c" : "#eab308"
                        }}
                      >
                        {item.status}
                      </span>
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

export default Dashboard;