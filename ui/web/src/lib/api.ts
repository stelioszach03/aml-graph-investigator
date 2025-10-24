export type FetchError = { status: number; message: string; body?: string };

const envBase = (import.meta as any)?.env?.VITE_API_BASE as string | undefined;
const defaultToken = (import.meta as any)?.env?.VITE_DEFAULT_TOKEN as string | undefined;

(function seedDefaultToken(){
  try {
    if (defaultToken && !localStorage.getItem('API_TOKEN')) {
      localStorage.setItem('API_TOKEN', defaultToken);
    }
  } catch {}
})();

function detectBase(): string {
  if (envBase && envBase.trim() !== '') return envBase.trim();
  try {
    const loc = window.location;
    if (loc.port === '5173') return 'http://localhost:8000';
    return `${loc.protocol}//${loc.host}`;
  } catch { return 'http://localhost:8000'; }
}

export const API_BASE = detectBase();

export function getToken(): string | null { try { return localStorage.getItem('API_TOKEN'); } catch { return null; } }

async function fetchJson(path: string, init: RequestInit = {}) {
  const token = getToken();
  const headers: Record<string,string> = { Accept: 'application/json', ...(init.headers as any) };
  if (!('Content-Type' in headers) && init.body) headers['Content-Type'] = 'application/json';
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  const ctype = res.headers.get('content-type') || '';
  if (!res.ok) {
    let bodyTxt = '';
    try { bodyTxt = ctype.includes('application/json') ? JSON.stringify(await res.json()) : await res.text(); } catch {}
    const err: FetchError = { status: res.status, message: res.statusText || 'Request failed', body: bodyTxt };
    throw err;
  }
  if (!ctype.includes('application/json')) {
    const bodyTxt = await res.text();
    const err: FetchError = { status: res.status, message: 'Non-JSON response from server (check API base/url)', body: bodyTxt };
    throw err;
  }
  return res.json();
}

export const api = {
  health: () => fetchJson('/api/v1/health'),
  authStatus: async () => {
    try {
      return await fetchJson('/api/v1/auth/status');
    } catch (e: any) {
      if (e?.status === 404) return { protected: false };
      throw e;
    }
  },
  getMetricsLast: () => fetchJson('/api/v1/metrics/last'),
  score: (topk = 100) => fetchJson('/api/v1/score', { method: 'POST', body: JSON.stringify({ topk }) }),
  caseData: (id: string) => fetchJson(`/api/v1/case/${encodeURIComponent(id)}`),
  explain: (id: string) => fetchJson(`/api/v1/explain/${encodeURIComponent(id)}`),
  ingest: (path: string, pushNeo4j = false) => fetchJson('/api/v1/ingest', { method: 'POST', body: JSON.stringify({ path, push_neo4j: pushNeo4j }) }),
  train: (labelsPath: string) => fetchJson('/api/v1/train', { method: 'POST', body: JSON.stringify({ labels_path: labelsPath }) }),
  whatIf: (payload: any) => fetchJson('/api/v1/what-if', { method: 'POST', body: JSON.stringify(payload) }),
};
