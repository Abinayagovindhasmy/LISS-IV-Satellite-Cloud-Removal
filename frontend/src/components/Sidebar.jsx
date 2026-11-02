// Sidebar component with RAG Navigation
import { useNavigate, useLocation } from "react-router-dom";

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  const isActive = (path) => location.pathname === path;

  return (
    <aside className="sidebar">
      <div className="sidebarLogo">
        ☁ CloudVision <span>AI</span>
      </div>

      <div className="sidebarMenu">
        <button 
          onClick={() => navigate("/dashboard")}
          style={{ background: isActive("/dashboard") ? "#168cff" : "transparent" }}
        >
          ◫ Dashboard
        </button>

        <button 
          onClick={() => navigate("/upload")}
          style={{ background: isActive("/upload") ? "#168cff" : "transparent" }}
        >
          ↑ Analyze Image
        </button>

        <button 
          onClick={() => navigate("/rag-assistant")}
          style={{ 
            background: isActive("/rag-assistant") ? "linear-gradient(90deg, #168cff, #8b5cf6)" : "rgba(139, 92, 246, 0.15)",
            border: "1px solid rgba(139, 92, 246, 0.4)",
            color: "#f8fafc",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <span>✦ AI RAG Assistant</span>
          <span style={{ fontSize: "0.65rem", background: "#8b5cf6", color: "white", padding: "2px 6px", borderRadius: "10px" }}>
            NEW
          </span>
        </button>

        <button 
          onClick={() => navigate("/history")}
          style={{ background: isActive("/history") ? "#168cff" : "transparent" }}
        >
          ◷ History
        </button>

        <button 
          onClick={() => navigate("/")}
          style={{ background: isActive("/") ? "#168cff" : "transparent" }}
        >
          ⌂ Home
        </button>
      </div>

      <button
        className="logoutButton"
        onClick={logout}
      >
        Logout
      </button>
    </aside>
  );
}

export default Sidebar;