import { useEffect, useState } from "react";
import { getHealth, screenResumes } from "./api.js";
import JdInput from "./components/JdInput.jsx";
import JdSummary from "./components/JdSummary.jsx";
import ResumeUpload from "./components/ResumeUpload.jsx";
import Results from "./components/Results.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [jdMode, setJdMode] = useState("file");
  const [jdFile, setJdFile] = useState(null);
  const [jdText, setJdText] = useState("");
  const [resumes, setResumes] = useState([]);
  const [results, setResults] = useState([]);
  const [jdAnalysis, setJdAnalysis] = useState(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "down" }));
  }, []);

  const maxResumes = health?.max_resumes || 20;
  const hasJd = jdMode === "file" ? !!jdFile : jdText.trim().length >= 50;
  const canSubmit = hasJd && resumes.length > 0 && !running;

  async function handleScreen() {
    setError("");
    setResults([]);
    setJdAnalysis(null);
    setProgress({ done: 0, total: resumes.length });
    setRunning(true);
    try {
      await screenResumes(
        { jdFile: jdMode === "file" ? jdFile : null, jdText, resumes },
        (event) => {
          if (event.type === "start") setProgress({ done: 0, total: event.total });
          if (event.type === "jd") setJdAnalysis(event.data);
          if (event.type === "error") setError(event.message);
          if (event.type === "result") {
            setResults((prev) => [...prev, event.data]);
            setProgress((p) => ({ ...p, done: p.done + 1 }));
          }
        }
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Resume Shortlister</h1>
          <p className="muted">Upload a job description and up to {maxResumes} resumes. AI ranks every candidate against the JD.</p>
        </div>
        <StatusPill health={health} />
      </header>

      <main>
        <div className="inputs">
          <JdInput
            mode={jdMode}
            onModeChange={setJdMode}
            file={jdFile}
            onFileChange={setJdFile}
            text={jdText}
            onTextChange={setJdText}
            disabled={running}
          />
          <ResumeUpload files={resumes} onChange={setResumes} maxFiles={maxResumes} disabled={running} />
        </div>

        <div className="actions">
          <button type="button" className="btn primary" onClick={handleScreen} disabled={!canSubmit}>
            {running
              ? jdAnalysis
                ? `Screening… ${progress.done}/${progress.total}`
                : "Reading job description…"
              :`Shortlist ${resumes.length || ""} candidate${resumes.length === 1 ? "" : "s"}`}
          </button>
          {!hasJd && <span className="muted small">Add a job description to continue</span>}
          {hasJd && !resumes.length && <span className="muted small">Add at least one resume</span>}
        </div>

        {running && (
          <div className="progress" aria-label="Screening progress">
            <div className="progress-fill" style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
          </div>
        )}

        {error && <div className="alert">{error}</div>}

        {jdAnalysis && <JdSummary analysis={jdAnalysis} />}

        {jdAnalysis && <Results results={results} running={running} />}
      </main>
    </div>
  );
}

function StatusPill({ health }) {
  if (!health) return <span className="pill">Connecting…</span>;
  if (health.status !== "ok") return <span className="pill pill-bad">Backend offline</span>;
  if (!health.api_key_configured) return <span className="pill pill-warn">API key missing</span>;
  return (
    <span className="pill pill-good" title="AI model in use">
      {health.model}
    </span>
  );
}
