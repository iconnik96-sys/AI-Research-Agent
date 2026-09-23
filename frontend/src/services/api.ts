import { getApiBaseUrl } from './config';
import {
  ApiError,
  HealthResponse,
  ReadyResponse,
  ResearchRequest,
  ResearchResponse,
} from '../types/api';

/**
 * Custom error class wrapping API failures with HTTP status and Request ID
 */
export class ResearchApiError extends Error implements ApiError {
  status?: number;
  requestId?: string;
  retryAfter?: number;

  constructor(message: string, status?: number, requestId?: string, retryAfter?: number) {
    super(message);
    this.name = 'ResearchApiError';
    this.status = status;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
  }
}

/**
 * Extracts human-readable message from FastAPI JSON response
 */
async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') {
      return data.detail;
    }
    if (Array.isArray(data.detail)) {
      // Pydantic validation error list
      return data.detail
        .map((err: { msg?: string; loc?: string[] }) => err.msg || JSON.stringify(err))
        .join('; ');
    }
    if (data.message && typeof data.message === 'string') {
      return data.message;
    }
  } catch {
    // Non-JSON response body
  }

  if (response.status === 429) {
    return 'Rate limit exceeded. Please wait a moment before submitting another research request.';
  }
  if (response.status === 502) {
    return 'Upstream service error (search or AI model provider). Please try again shortly.';
  }
  if (response.status === 503) {
    return 'Research service or database is currently unavailable.';
  }
  if (response.status === 504) {
    return 'The research operation timed out. Try narrowing your research question.';
  }

  return `Request failed with status code ${response.status} (${response.statusText || 'Error'}).`;
}

/**
 * Submits a research question to POST /api/research
 */
export async function submitResearch(
  question: string,
  signal?: AbortSignal
): Promise<ResearchResponse> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/api/research`;

  const payload: ResearchRequest = { question: question.trim() };

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ResearchApiError('Research request was cancelled.');
    }
    const message = err instanceof Error ? err.message : 'Network connection failed.';
    throw new ResearchApiError(
      `Unable to reach backend server: ${message}. Ensure the backend is running.`,
      0
    );
  }

  const requestId = response.headers.get('x-request-id') || response.headers.get('X-Request-ID') || undefined;

  if (!response.ok) {
    const errorMsg = await parseErrorResponse(response);
    const retryAfterHeader = response.headers.get('Retry-After');
    const retryAfter = retryAfterHeader ? parseInt(retryAfterHeader, 10) : undefined;
    throw new ResearchApiError(errorMsg, response.status, requestId, retryAfter);
  }

  try {
    const data: ResearchResponse = await response.json();
    return data;
  } catch {
    throw new ResearchApiError(
      'Invalid JSON response received from backend.',
      response.status,
      requestId
    );
  }
}

/**
 * Checks liveness probe via GET /health
 */
export async function checkHealth(): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/health`;

  const response = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Checks readiness probe via GET /ready
 */
export async function checkReadiness(): Promise<ReadyResponse> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/ready`;

  const response = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Readiness check failed with status ${response.status}`);
  }

  return response.json();
}
