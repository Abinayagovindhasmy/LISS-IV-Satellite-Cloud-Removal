import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function UserProfileCorner() {
  const navigate = useNavigate();
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("user") || "null");
    } catch (e) {
      return null;
    }
  });
  const [isOpen, setIsOpen] = useState(false);
  const [dbStatus, setDbStatus] = useState("Connected");

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      api.get("/auth/me")
        .then((res) => {
          setUser(res.data);
          localStorage.setItem("user", JSON.stringify(res.data));
          setDbStatus("Connected (SQLite DB)");
        })
        .catch(() => {
          setDbStatus("Demo Local Session");
        });
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
    setIsOpen(false);
    navigate("/");
  };

  const handleQuickDemo = () => {
    const demoUser = {
      id: 1,
      name: "Demo Scientist",
      email: "admin@cloudvision.ai",
      created_at: new Date().toISOString()
    };
    localStorage.setItem("token", "demo_guest_token_12345");
    localStorage.setItem("user", JSON.stringify(demoUser));
    setUser(demoUser);
    setIsOpen(false);
  };

  const initial = user?.name ? user.name.charAt(0).toUpperCase() : "G";

  return (
    <div className="userProfileCornerContainer">
      {/* Top-Left Corner Glass Badge */}
      <div 
        className={`profileGlassBadge ${user ? 'isLoggedIn' : 'isGuest'}`}
        onClick={() => setIsOpen(!isOpen)}
        title="Click to toggle profile menu"
      >
        <div className="profileAvatarWrapper">
          <div className="profileAvatarCircle">
            {initial}
          </div>
          <span className={`statusDot ${user ? 'active' : 'guest'}`}></span>
        </div>

        <div className="profileInfoText">
          <div className="profileHeaderTitle">
            <span className="profileName">{user ? user.name : "Guest Analyst"}</span>
            <span className="profileBadgeTag">{user ? "DB AUTH" : "GUEST"}</span>
          </div>
          <p className="profileSubtext">{user ? user.email : "Not signed in"}</p>
        </div>

        <div className="profileChevron">{isOpen ? "▲" : "▼"}</div>
      </div>

      {/* Expanded Profile Popup Card */}
      {isOpen && (
        <div className="profileDropdownCard">
          <div className="profileCardHeader">
            <div className="profileCardAvatar">
              {initial}
            </div>
            <div>
              <h3>{user ? user.name : "Guest Analyst"}</h3>
              <p>{user ? user.email : "Sign in to persist database history"}</p>
              <div className="profileRolePill">
                🛰️ {user ? "LISS-IV Database User" : "Anonymous Session"}
              </div>
            </div>
          </div>

          <div className="profileCardBody">
            <div className="dbStatusRow">
              <span>Database Status:</span>
              <span className="dbBadge">🟢 {dbStatus}</span>
            </div>
            <div className="dbStatusRow">
              <span>Model Target:</span>
              <span className="modelBadge">⚡ LISS-IV 4-Band UNet</span>
            </div>

            {user && (
              <div className="quickNavGrid">
                <button onClick={() => { setIsOpen(false); navigate("/dashboard"); }}>
                  📊 Dashboard
                </button>
                <button onClick={() => { setIsOpen(false); navigate("/upload"); }}>
                  ↑ Analyze Image
                </button>
                <button onClick={() => { setIsOpen(false); navigate("/history"); }}>
                  📜 My History
                </button>
              </div>
            )}
          </div>

          <div className="profileCardFooter">
            {user ? (
              <button className="logoutBtn" onClick={handleLogout}>
                🚪 Sign Out
              </button>
            ) : (
              <div className="guestAuthButtons">
                <button className="loginBtn" onClick={() => { setIsOpen(false); navigate("/login"); }}>
                  Sign In
                </button>
                <button className="registerBtn" onClick={() => { setIsOpen(false); navigate("/register"); }}>
                  Create Account
                </button>
                <button className="demoBtn" onClick={handleQuickDemo}>
                  ⚡ Demo Account
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default UserProfileCorner;
