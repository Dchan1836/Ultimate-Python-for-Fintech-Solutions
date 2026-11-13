export async function fetchDemos() {
  const response = await fetch("/api/demos");
  if (!response.ok) {
    throw new Error("The Python API is not responding. Start it on port 8000, then reload this page.");
  }
  return response.json();
}

export async function runDemo(id, params) {
  const response = await fetch(`/api/run/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === "string" ? detail : "The model run failed.";
    throw new Error(message);
  }
  return body;
}
