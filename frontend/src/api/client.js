// Base URL for the RiftRace FastAPI backend. Override with a .env file
// (VITE_API_BASE_URL=...) if the backend runs somewhere other than
// http://localhost:8000.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Lets components branch on `error.kind` instead of re-parsing status codes,
// and guarantees a user-friendly message is always available.
export class ApiError extends Error {
  constructor(kind, message) {
    super(message)
    this.kind = kind // 'network' | 'not_found' | 'server' | 'client'
  }
}

// Wraps fetch with: network-failure detection (backend down/unreachable),
// HTTP status handling, and normalizing FastAPI's different error body
// shapes -- HTTPException -> {detail: "..."}, our global handler ->
// {error, message}, pydantic validation -> {detail: [{msg: "..."}]} --
// into a single readable message.
export default async function apiFetch(path, options) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new ApiError(
      'network',
      `Could not reach the RiftRace backend at ${API_BASE_URL}. Is it running?`,
    )
  }

  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`
    try {
      const body = await response.json()
      if (Array.isArray(body.detail)) {
        message = body.detail.map((d) => d.msg).join(', ')
      } else if (typeof body.detail === 'string') {
        message = body.detail
      } else if (body.message) {
        message = body.message
      }
    } catch {
      // Response body wasn't JSON (e.g. a raw error page) -- keep the generic message.
    }

    if (response.status === 404) throw new ApiError('not_found', message)
    if (response.status >= 500) throw new ApiError('server', message)
    throw new ApiError('client', message)
  }

  return response.json()
}
