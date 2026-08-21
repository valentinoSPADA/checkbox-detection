/**
 * Typed client for the detection API.
 *
 * This is the only module that knows the wire format. Everything else in the UI works with
 * the exported types, so a change to the backend contract lands here and nowhere else.
 */

/** Which detection strategy the backend should run. */
export type EngineName = 'local' | 'vlm' | 'assisted'

/** One detected checkbox, in source-image pixel coordinates. */
export interface DetectedBox {
  /** [x1, y1, x2, y2] — top-left and bottom-right corners. */
  bbox: [number, number, number, number]
  is_checked: boolean
  /** Probability of the winning class. Present only on verbose responses. */
  confidence: number
  /** Which engine produced this box; `assisted` mixes two sources in one response. */
  source: EngineName
}

/** Diagnostics describing how a response was produced. */
export interface DetectMeta {
  engine: EngineName
  width: number
  height: number
  elapsed_ms: number
  stats: {
    raw_proposals?: number
    scored_proposals?: number
    candidates: number
    escalated?: number
    returned: number
  }
}

export interface DetectResponse {
  boxes: DetectedBox[]
  meta: DetectMeta
}

export interface EnginesResponse {
  engines: EngineName[]
  default: EngineName
}

/**
 * An API failure carrying the HTTP status, so callers can distinguish a bad upload (4xx,
 * the user's problem, worth showing verbatim) from an outage (5xx, worth advising a retry).
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Base URL of the API.
 *
 * Read from the build-time environment with a localhost default, so `npm run dev` works with
 * no configuration while a deployed build points at the real host. An empty string is
 * treated as unset rather than as "same origin", because an accidentally-empty variable in
 * CI would otherwise produce a build that silently posts to itself.
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || 'http://localhost:8080'

/** Parse an error body into a message, tolerating a non-JSON response from a proxy. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: string; details?: string }
    return body.details ? `${body.error}: ${body.details}` : (body.error ?? response.statusText)
  } catch {
    // A gateway or load balancer can answer with HTML; surfacing the status is more useful
    // than surfacing a JSON parse error the user can do nothing about.
    return response.statusText || `request failed with status ${response.status}`
  }
}

/**
 * Upload one image and return its detections.
 *
 * Always requests the verbose response: the UI colours boxes by confidence and shows the
 * pipeline counters, neither of which exists in the strict schema.
 *
 * @param file the image to analyse
 * @param options.engine which strategy to run; omit to use the server default
 * @param options.minConfidence override the server's confidence floor for this request
 * @param options.signal abort signal, so switching engines cancels the in-flight request
 *   rather than racing it — without this, a slow `vlm` response can land after a fast
 *   `local` one and overwrite it.
 * @throws ApiError on any non-2xx response
 */
export async function detect(
  file: File,
  options: { engine?: EngineName; minConfidence?: number; signal?: AbortSignal } = {},
): Promise<DetectResponse> {
  const form = new FormData()
  form.append('file', file)

  const params = new URLSearchParams({ verbose: 'true' })
  if (options.engine) params.set('engine', options.engine)
  if (options.minConfidence !== undefined) params.set('min_confidence', String(options.minConfidence))

  const response = await fetch(`${API_BASE}/detect?${params}`, {
    method: 'POST',
    body: form,
    signal: options.signal,
  })
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status)
  }
  return (await response.json()) as DetectResponse
}

/**
 * List the engines this backend instance can actually run.
 *
 * Engine availability is configuration-dependent — without an API key the vision engines are
 * not registered — so the UI asks rather than assuming, and never offers a control that is
 * guaranteed to fail.
 */
export async function listEngines(signal?: AbortSignal): Promise<EnginesResponse> {
  const response = await fetch(`${API_BASE}/engines`, { signal })
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status)
  }
  return (await response.json()) as EnginesResponse
}

/** Human-readable label and description for each engine, used by the picker. */
export const ENGINE_INFO: Record<EngineName, { label: string; hint: string }> = {
  local: {
    label: 'Local',
    hint: 'Geometric proposals plus the trained CNN. Fast, free, runs offline.',
  },
  vlm: {
    label: 'Claude vision',
    hint: 'The page is tiled and read directly by Claude. Slow and paid, but needs no training.',
  },
  assisted: {
    label: 'Assisted',
    hint: 'Local pipeline, with only its uncertain candidates escalated to Claude.',
  },
}
