const API_URL = import.meta.env.VITE_API_URL || "";

export async function getHealth() {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error("Backend is not reachable");
  return res.json();
}

/**
 * Sends the JD and resumes to the backend and calls onEvent for every streamed event:
 * { type: "start", total } | { type: "result", data } | { type: "done" }
 */
export async function screenResumes({ jdFile, jdText, resumes }, onEvent) {
  const form = new FormData();
  if (jdFile) form.append("jd_file", jdFile);
  else form.append("jd_text", jdText);
  resumes.forEach((file) => form.append("resumes", file));

  let res;
  try {
    res = await fetch(`${API_URL}/api/screen`, { method: "POST", body: form });
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
