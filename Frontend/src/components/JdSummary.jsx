import { CRITERIA } from "../categories.js";

export default function JdSummary({ analysis }) {
  const meta = [analysis.department, analysis.seniority].filter(Boolean).join(" · ");
  const mustHaves = [analysis.experience_requirement, ...analysis.must_haves].filter(Boolean);

  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2>{analysis.job_title}</h2>
          {meta && <span className="muted small">{meta}</span>}
        </div>
        <span className="muted small">How the AI read the JD</span>
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
          {mustHaves.length ? (
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
