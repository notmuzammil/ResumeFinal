const API_URL = import.meta.env.VITE_API_URL || "";

export async function getHealth() {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error("Backend is not reachable");
  return res.json();
}

/** Step 1: the AI reads the JD; HR reviews and edits the result before screening. */
export async function analyzeJd({ jdFile, jdText }) {
  const res = await post("/api/analyze-jd", jdForm({ jdFile, jdText }));
  return res.json();
}

/**
 * Step 2: sends the JD, the reviewed JD analysis and the resumes, and calls onEvent for every streamed event:
 * { type: "start", total } | { type: "jd", data } | { type: "result", data } | { type: "done" }
 */
export async function screenResumes({ jdFile, jdText, resumes, jdAnalysis, ids }, onEvent) {
  const form = jdForm({ jdFile, jdText });
  resumes.forEach((file) => form.append("resumes", file));
  if (jdAnalysis) form.append("jd_analysis", JSON.stringify(jdAnalysis));
  // Retry: keep the original result ids so rows are replaced in place
  if (ids) form.append("ids", JSON.stringify(ids));

  const res = await post("/api/screen", form);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function jdForm({ jdFile, jdText }) {
  const form = new FormData();
  if (jdFile) form.append("jd_file", jdFile);
  else form.append("jd_text", jdText);
  return form;
}

async function post(path, form) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  } catch {
    throw new Error("Cannot reach the backend. Make sure it is running on port 8000.");
  }

  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // keep the generic message
    }
    throw new Error(message);
  }
  return res;
}
