import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("admin@cloudvision.ai");
  const [password, setPassword] = useState("password123");
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const login = async (e) => {
    if (e) e.preventDefault();

    try {
      setLoading(true);
      setMessage("");

      const response = await api.post("/auth/login", {
        email,
        password,
      });

      localStorage.setItem("token", response.data.access_token);
      localStorage.setItem("user", JSON.stringify(response.data.user));

      navigate("/dashboard");
    } catch (error) {
      console.error("Login error:", error);
      const errorDetail = error.response?.data?.detail;
      
      if (errorDetail) {
        setMessage(errorDetail);
      } else {
        // Fallback for demo / offline backend connection issue
        setMessage("Unable to connect to backend server. Entering guest demo mode...");
        setTimeout(() => {
          localStorage.setItem("token", "demo_guest_token_12345");
          localStorage.setItem("user", JSON.stringify({ name: "Demo Scientist", email: "admin@cloudvision.ai" }));
          navigate("/dashboard");
        }, 1200);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = () => {
    setEmail("admin@cloudvision.ai");
    setPassword("password123");
    // Trigger login
    setTimeout(() => {
      const demoEvent = { preventDefault: () => {} };
      login(demoEvent);
    }, 100);
  };

  const handleGuestAccess = () => {
    localStorage.setItem("token", "guest_demo_token");
    localStorage.setItem("user", JSON.stringify({ name: "Guest User", email: "guest@cloudvision.ai" }));
    navigate("/dashboard");
  };

  return (
    <div className="authPage">
      <form className="authCard" onSubmit={login}>
        <div className="authBrand" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
          ☁ CloudVision <span style={{ color: "#38bdf8" }}>AI</span>
        </div>

        <p className="authTag">Earth Observation & LISS-IV Cloud Removal</p>

        <h1>Welcome Back</h1>

        <p className="authDescription">
          Sign in to access your satellite image analysis dashboard and Generative AI cloud reconstruction pipeline.
        </p>

        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "6px", fontSize: "0.9rem", color: "#cbd5e1" }}>Email Address</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={{ width: "100%", padding: "12px", borderRadius: "8px", border: "1px solid #334155", background: "#0f172a", color: "#fff" }}
          />
        </div>

        <div style={{ marginBottom: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
            <label style={{ fontSize: "0.9rem", color: "#cbd5e1" }}>Password</label>
            <span
              onClick={() => setShowPassword(!showPassword)}
              style={{ fontSize: "0.8rem", color: "#38bdf8", cursor: "pointer" }}
            >
              {showPassword ? "Hide" : "Show"}
            </span>
          </div>
          <input
            type={showPassword ? "text" : "password"}
            placeholder="Enter password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: "100%", padding: "12px", borderRadius: "8px", border: "1px solid #334155", background: "#0f172a", color: "#fff" }}
          />
        </div>

        <button className="authSubmit" disabled={loading} style={{ width: "100%", padding: "14px", borderRadius: "8px", background: "linear-gradient(90deg, #0284c7, #2563eb)", color: "#fff", fontWeight: "bold", border: "none", cursor: "pointer" }}>
          {loading ? "Signing in..." : "Login to Dashboard"}
        </button>

        {/* Quick Demo Fill Button */}
        <div style={{ display: "flex", gap: "10px", marginTop: "15px" }}>
          <button
            type="button"
            onClick={handleDemoLogin}
            style={{ flex: 1, padding: "10px", borderRadius: "6px", background: "#1e293b", border: "1px solid #38bdf8", color: "#38bdf8", fontSize: "0.85rem", cursor: "pointer", fontWeight: "600" }}
          >
            ⚡ Quick Demo Login
          </button>
          <button
            type="button"
            onClick={handleGuestAccess}
            style={{ flex: 1, padding: "10px", borderRadius: "6px", background: "#1e293b", border: "1px solid #475569", color: "#94a3b8", fontSize: "0.85rem", cursor: "pointer" }}
          >
            👤 Guest Access
          </button>
        </div>

        {message && (
          <div className="formMessage" style={{ marginTop: "15px", padding: "12px", borderRadius: "6px", background: "#450a0a", border: "1px solid #dc2626", color: "#fca5a5", fontSize: "0.85rem", textAlign: "center" }}>
            {message}
          </div>
        )}

        <p className="switchAuth" style={{ marginTop: "20px", textAlign: "center", fontSize: "0.9rem", color: "#94a3b8" }}>
          Don't have an account?{" "}
          <span onClick={() => navigate("/register")} style={{ color: "#38bdf8", cursor: "pointer", textDecoration: "underline" }}>
            Create Account
          </span>
        </p>
      </form>
    </div>
  );
}

export default Login;