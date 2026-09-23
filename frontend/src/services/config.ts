/**
 * Configuration service for API URL resolution
 */

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (!envUrl || typeof envUrl !== 'string' || !envUrl.trim()) {
    // Relative URL (uses Vite proxy in development, or same origin in production)
    return '';
  }
  // Trim trailing slashes
  return envUrl.trim().replace(/\/+$/, '');
}
