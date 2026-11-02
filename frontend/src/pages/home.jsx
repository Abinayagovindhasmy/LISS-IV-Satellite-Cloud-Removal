import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import UserProfileCorner from "../components/UserProfileCorner";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="appPage">
      <Navbar />

      <main className="hero">
        <section className="heroText">
          <div style={{ display: "flex", alignItems: "center", gap: "15px", marginBottom: "20px" }}>
            <UserProfileCorner />
            <div className="tag" style={{ margin: 0 }}>
              ✦ AI POWERED SATELLITE IMAGERY
            </div>
          </div>

          <h1>
            Generative AI for
            <span> Cloud Removal</span>
          </h1>

          <h2>
            for LISS-IV Satellite Imagery
          </h2>

          <p>
            Remove clouds. Reconstruct the Earth.
            Transform cloud-obscured LISS-IV satellite
            images into clear, analysis-ready visuals
            using Generative AI.
          </p>

          <div className="heroButtons">
            <button
              className="primaryBtn bigBtn"
              onClick={() => navigate("/upload")}
            >
              ↑ Upload LISS-IV Image
            </button>

            <button
              className="outlineBtn bigBtn"
              onClick={() =>
                document
                  .getElementById("workflow")
                  ?.scrollIntoView({ behavior: "smooth" })
              }
            >
              ▶ View How It Works
            </button>
          </div>

          <div className="featureLine">
            <span>◈ LISS-IV Support</span>
            <span>◉ Pix2Pix GAN</span>
            <span>✓ Reconstruction Pipeline</span>
          </div>
        </section>

        <section className="comparison">
          <div className="compareImage">
            <span>Cloudy Image</span>

            <img
              src="/cloud.jpg"
              alt="Cloudy satellite"
            />
          </div>

          <div className="compareImage">
            <span>Reconstructed Image</span>

            <img
              src="/reconstruct.jpg"
              alt="Reconstructed satellite"
            />
          </div>
        </section>
      </main>

      <section
        className="workflow"
        id="workflow"
      >
        <h2>How It Works</h2>

        <div className="workflowSteps">
          <div className="stepCard">
            <div className="stepIcon">↑</div>
            <h3>1. Upload Image</h3>
            <p>
              Upload your cloud-covered LISS-IV satellite
              image.
            </p>
          </div>

          <div className="stepArrow">→</div>

          <div className="stepCard">
            <div className="stepIcon">▧</div>
            <h3>2. Preprocess</h3>
            <p>
              The backend prepares the satellite image for
              inference.
            </p>
          </div>

          <div className="stepArrow">→</div>

          <div className="stepCard">
            <div className="stepIcon">✦</div>
            <h3>3. AI Reconstruction</h3>
            <p>
              The reconstruction model processes the cloudy
              image.
            </p>
          </div>

          <div className="stepArrow">→</div>

          <div className="stepCard">
            <div className="stepIcon">✓</div>
            <h3>4. Get Result</h3>
            <p>
              Compare and download the reconstructed image.
            </p>
          </div>
        </div>
      </section>

      <section className="infoGrid">
        <div className="infoCard">
          <div className="infoIcon">☁</div>

          <div>
            <h3>The Challenge</h3>

            <p>
              Clouds obscure agricultural fields, forests,
              water bodies, roads and urban areas in optical
              satellite imagery.
            </p>
          </div>
        </div>

        <div className="infoCard">
          <div className="infoIcon">✦</div>

          <div>
            <h3>Our Solution</h3>

            <p>
              Generative image reconstruction is used to
              estimate cloud-obscured surface information.
            </p>
          </div>
        </div>

        <div className="infoCard">
          <div className="infoIcon">▦</div>

          <div>
            <h3>Applications</h3>

            <p>
              Agriculture, disaster management,
              environmental studies, urban planning and
              land-use analysis.
            </p>
          </div>
        </div>
      </section>

      <footer className="footer">
        React • FastAPI • PyTorch • Pix2Pix GAN • SQLite
      </footer>
    </div>
  );
}

export default Home;