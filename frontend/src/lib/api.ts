/**
 * Typed client for the detection API.
 *
 * This is the only module that knows the wire format. Everything else in the UI works with
 * the exported types, so a change to the backend contract lands here and nowhere else.
 */

/**
 * Which detection strategy the backend should run.
 *
 * One value today. The type, the `source` field and the `/engines` lookup all survive a
 * single-engine build on purpose: the backend enumerates its registry rather than a literal,
 * so registering a second engine there starts advertising it without a frontend release.
 */
export type EngineName = 'local'

/** One detected checkbox, in source-image pixel coordinates. */
export interface DetectedBox {
  /** [x1, y1, x2, y2] — top-left and bottom-right corners. */
  bbox: [number, number, number, number]
  is_checked: boolean
  /** Probability of the winning class. Present only on verbose responses. */
  confidence: number
  /** Which engine produced this box. Per-box, so a mixed-source response stays attributable. */
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
 * Sentinel for a build whose API lives on the same origin as the page.
 *
 * A word rather than the empty string, because an empty variable is what a misconfigured CI
 * job produces and "posts to itself" must never be something a build falls into by accident.
 * Same-origin is a real deployment shape here -- the production image serves the bundle from
 * the Go binary that also answers /detect -- so it needs a way to be *chosen*.
 */
const SAME_ORIGIN = 'same-origin'

/**
 * Base URL of the API.
 *
 * Read from the build-time environment with a localhost default, so `npm run dev` works with
 * no configuration while a deployed build points at the real host.
 */
const configuredBase = import.meta.env.VITE_API_BASE_URL as string | undefined
const API_BASE =
  configuredBase === SAME_ORIGIN ? '' : configuredBase || 'http://localhost:8080'

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
 * Asked rather than assumed. It reports one engine today, and the call is kept because the
 * alternative — hard-coding the list in the UI — is what makes a backend change require a
 * frontend release to become visible.
 */
export async function listEngines(signal?: AbortSignal): Promise<EnginesResponse> {
  const response = await fetch(`${API_BASE}/engines`, { signal })
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status)
  }
  return (await response.json()) as EnginesResponse
}

/** Human-readable label for each engine, used wherever one is named in the UI. */
export const ENGINE_INFO: Record<EngineName, { label: string; hint: string }> = {
  local: {
    label: 'Local',
    hint: 'Geometric proposals plus a trained CNN. Runs offline, costs nothing per page.',
  },
}
