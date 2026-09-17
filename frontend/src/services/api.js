const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return await res.json();
  } catch (err) {
    console.error('API Health Check Error:', err);
    return { status: 'offline', error: err.message };
  }
}

export async function predictSonarImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
  }

  return await res.json();
}

export async function getPredictionsHistory() {
  const res = await fetch(`${API_BASE_URL}/api/predictions`);
  if (!res.ok) {
    throw new Error(`Failed to fetch history (${res.status})`);
  }
  return await res.json();
}

export async function getAnalyticsData() {
  const res = await fetch(`${API_BASE_URL}/api/analytics`);
  if (!res.ok) {
    throw new Error(`Failed to fetch analytics data (${res.status})`);
  }
  return await res.json();
}

