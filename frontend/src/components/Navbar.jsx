import { useNavigate } from "react-router-dom";
import UserProfileCorner from "./UserProfileCorner";

function Navbar() {
  const navigate = useNavigate();

  const user = JSON.parse(
    localStorage.getItem("user") || "null"
  );

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/");
  };

  return (
    <nav className="navbar">
      <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
        <UserProfileCorner />

        <div
          className="brand"
          onClick={() => navigate("/")}
        >
          <div className="brandIcon">☁</div>

          <div>
            <h2>
              CloudVision <span>AI</span>
            </h2>
            <small>Earth Observation Redefined</small>
          </div>
        </div>
      </div>

      <div className="navLinks">
        <button onClick={() => navigate("/")}>
          Home
        </button>

        {user && (
          <>
            <button onClick={() => navigate("/dashboard")}>
              Dashboard
            </button>

            <button onClick={() => navigate("/upload")}>
              Analyze
            </button>

            <button onClick={() => navigate("/history")}>
              History
            </button>
          </>
        )}
      </div>

      <div className="navActions">
        {!user ? (
          <>
            <button
              className="outlineBtn"
              onClick={() => navigate("/login")}
            >
              Login
            </button>

            <button
              className="primaryBtn"
              onClick={() => navigate("/register")}
            >
              Get Started
            </button>
          </>
        ) : (
          <button
            className="outlineBtn"
            onClick={logout}
          >
            Logout
          </button>
        )}
      </div>
    </nav>
  );
}

export default Navbar;