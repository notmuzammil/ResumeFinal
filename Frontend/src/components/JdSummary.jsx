import { CRITERIA } from "../categories.js";

// Keep in sync with MAX_EDITED_MUST_HAVES in Backend/app/llm.py
const MAX_MUST_HAVES = 10;

export default function JdSummary({ analysis, editable = false, onChange }) {
  const meta = [analysis.department, analysis.seniority].filter(Boolean).join(" · ");
  const mustHaves = [analysis.experience_requirement, ...analysis.must_haves].filter(Boolean);

  return (
    <section className={`card${editable ? " card-review" : ""}`}>
      <div className="card-header">
        <div>
          <h2>{analysis.job_title}</h2>
          {meta && <span className="muted small">{meta}</span>}
        </div>
        <span className="muted small">{editable ? "Check before screening" : "How the AI read the JD"}</span>
      </div>

      <div className="jd-grid">
        <div>
          <h4>Scoring weights</h4>
          <div className="weights">
            {CRITERIA.map(({ key, label }) => (
              <div key={key} className="breakdown-item">
                <div className="breakdown-label">
                  <span>{label}</span>
                  <span>{analysis.weights[key]}%</span>
                </div>
                <div className="score-bar">
                  <div className="score-fill neutral" style={{ width: `${analysis.weights[key]}%` }} />
                </div>
              </div>
            ))}
          </div>
          {analysis.weight_reason && <p className="muted small">{analysis.weight_reason}</p>}
        </div>

        <div>
          <h4>Must-haves (missing any = Not Suitable)</h4>
          {editable ? (
            <MustHaveEditor analysis={analysis} onChange={onChange} />
          ) : mustHaves.length ? (
            <ul className="bullets">
              {mustHaves.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          ) : (
            <p className="muted small">The JD has no strictly mandatory requirements. Candidates are ranked by score only.</p>
          )}
          {analysis.nice_to_haves.length > 0 && (
            <>
              <h4 className="spaced">Nice to have</h4>
              <div className="chips">
                {analysis.nice_to_haves.map((n) => (
                  <span key={n} className="chip chip-neutral">
                    {n}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function MustHaveEditor({ analysis, onChange }) {
  const items = analysis.must_haves;

  function setItems(next) {
    onChange({ ...analysis, must_haves: next });
  }

  function setYears(value) {
    const years = value === "" ? null : Number(value);
    const valid = years > 0 ? years : null;
    onChange({
      ...analysis,
      min_experience_years: valid,
      experience_requirement: valid ? `Minimum ${valid} years of relevant experience` : null,
    });
  }

  return (
    <>
      <p className="muted small">
        The AI drafted this list from the JD. Each item rejects every candidate who does not show it, so remove
        anything that is not a real deal-breaker (practices, soft skills, nice-to-haves), fix the wording, or add
        what is missing.
      </p>

      <label className="years-row">
        <span>Minimum years of experience</span>
        <input
          type="number"
          min="0"
          step="0.5"
          className="jd-text years-input"
          placeholder="None"
          value={analysis.min_experience_years ?? ""}
          onChange={(e) => setYears(e.target.value)}
        />
      </label>

      <ul className="edit-list">
        {items.map((item, i) => (
          <li key={i} className="edit-row">
            <textarea
              className="jd-text edit-input"
              rows={1}
              value={item}
              aria-label={`Must-have ${i + 1}`}
              onChange={(e) => setItems(items.map((m, j) => (j === i ? e.target.value : m)))}
            />
            <button
              type="button"
              className="link-btn"
              aria-label={`Remove ${item || `must-have ${i + 1}`}`}
              onClick={() => setItems(items.filter((_, j) => j !== i))}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
      {!items.length && <p className="muted small">No must-haves: candidates are ranked by score only.</p>}

      <button
        type="button"
        className="link-btn"
        onClick={() => setItems([...items, ""])}
        disabled={items.length >= MAX_MUST_HAVES}
      >
        + Add must-have
      </button>
    </>
  );
}
