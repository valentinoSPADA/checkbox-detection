import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, detect, listEngines } from './api'

/** Build a minimal Response stand-in for the fetch mock. */
function jsonResponse(body: unknown, init: { ok?: boolean; status?: number; statusText?: string } = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: async () => body,
  } as Response
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('detect', () => {
  it('always requests the verbose response', async () => {
    // The strict schema carries no confidence, and the UI colours and filters by it; if this
    // flag were dropped the overlay would silently render every box as equally certain.
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ boxes: [], meta: {} }))
    vi.stubGlobal('fetch', fetchMock)

    await detect(new File(['x'], 'page.png', { type: 'image/png' }))

    const url = new URL(fetchMock.mock.calls[0][0] as string)
    expect(url.searchParams.get('verbose')).toBe('true')
  })

  it('forwards engine and min_confidence when provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ boxes: [], meta: {} }))
    vi.stubGlobal('fetch', fetchMock)

    await detect(new File(['x'], 'page.png'), { engine: 'assisted', minConfidence: 0.42 })

    const url = new URL(fetchMock.mock.calls[0][0] as string)
    expect(url.searchParams.get('engine')).toBe('assisted')
    expect(url.searchParams.get('min_confidence')).toBe('0.42')
  })

  it('omits engine when not specified so the server default applies', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ boxes: [], meta: {} }))
    vi.stubGlobal('fetch', fetchMock)

    await detect(new File(['x'], 'page.png'))

    const url = new URL(fetchMock.mock.calls[0][0] as string)
    expect(url.searchParams.has('engine')).toBe(false)
  })

  it('sends the file as multipart form data under the "file" field', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ boxes: [], meta: {} }))
    vi.stubGlobal('fetch', fetchMock)

    const file = new File(['x'], 'page.png', { type: 'image/png' })
    await detect(file)

    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.method).toBe('POST')
    expect(init.body).toBeInstanceOf(FormData)
    expect((init.body as FormData).get('file')).toBe(file)
  })

  it('raises ApiError carrying the status so callers can tell 4xx from 5xx', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({ error: 'invalid upload', details: 'not a decodable image' }, { ok: false, status: 400 }),
      ),
    )

    await expect(detect(new File(['x'], 'page.png'))).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      message: 'invalid upload: not a decodable image',
    })
  })

  it('survives a non-JSON error body from a proxy', async () => {
    // A gateway can answer with HTML; a JSON parse failure here would mask the real status.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: 'Bad Gateway',
        json: async () => {
          throw new SyntaxError('Unexpected token <')
        },
      } as unknown as Response),
    )

    const err = await detect(new File(['x'], 'page.png')).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(502)
    expect((err as ApiError).message).toBe('Bad Gateway')
  })

  it('passes the abort signal through so a superseded request can be cancelled', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ boxes: [], meta: {} }))
    vi.stubGlobal('fetch', fetchMock)

    const ctl = new AbortController()
    await detect(new File(['x'], 'page.png'), { signal: ctl.signal })

    expect((fetchMock.mock.calls[0][1] as RequestInit).signal).toBe(ctl.signal)
  })
})

describe('listEngines', () => {
  it('returns the engines the instance actually registered', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ engines: ['local'], default: 'local' })))
    await expect(listEngines()).resolves.toEqual({ engines: ['local'], default: 'local' })
  })

  it('raises ApiError on failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ error: 'nope' }, { ok: false, status: 500 })))
    await expect(listEngines()).rejects.toBeInstanceOf(ApiError)
  })
})
