// Registration page.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function Register() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const updateForm = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const register = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      setMessage("");

      await api.post("/auth/register", form);

      setMessage("Account created successfully.");

      setTimeout(() => {
        navigate("/login");
      }, 800);
    } catch (error) {
      setMessage(
        error.response?.data?.detail ||
          "Registration failed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="authPage">
      <div className="authBackground"></div>

      <form
        className="authCard"
        onSubmit={register}
      >
        <div
          className="authBrand"
          onClick={() => navigate("/")}
        >
          ☁ CloudVision <span>AI</span>
        </div>

        <p className="authTag">
          LISS-IV Cloud Removal Platform
        </p>

        <h1>Create Account</h1>

        <p className="authDescription">
          Create your account to analyze satellite images
          and maintain reconstruction history.
        </p>

        <label>Full Name</label>

        <input
          name="name"
          type="text"
          placeholder="Enter your name"
          value={form.name}
          onChange={updateForm}
          required
        />

        <label>Email Address</label>

        <input
          name="email"
          type="email"
          placeholder="Enter your email"
          value={form.email}
          onChange={updateForm}
          required
        />

        <label>Password</label>

        <input
          name="password"
          type="password"
          placeholder="Create password"
          value={form.password}
          onChange={updateForm}
          required
        />

        <button
          className="authSubmit"
          disabled={loading}
        >
          {loading
            ? "Creating account..."
            : "Create Account"}
        </button>

        {message && (
          <div className="formMessage">
            {message}
          </div>
        )}

        <p className="switchAuth">
          Already have an account?{" "}
          <span onClick={() => navigate("/login")}>
            Login
          </span>
        </p>
      </form>
    </div>
  );
}

export default Register;