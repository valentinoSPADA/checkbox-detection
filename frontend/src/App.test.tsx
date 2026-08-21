import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

/**
 * These tests exist for one reason: detection costs money on two of the three engines, so
 * "nothing is sent until the user asks" is a behavioural guarantee, not a UI preference.
 * An earlier version fired a request on file selection *and* on every engine change, which
 * meant trying two engines on one page spent three calls, two of them unrequested.
 */

function stubFetch(detectBody: unknown = { boxes: [], meta: emptyMeta() }) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const body = url.includes('/engines')
      ? { engines: ['local', 'vlm', 'assisted'], default: 'local' }
      : detectBody
    return { ok: true, status: 200, statusText: 'OK', json: async () => body } as Response
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function emptyMeta() {
  return { engine: 'local', width: 100, height: 100, elapsed_ms: 5, stats: { candidates: 0, returned: 0 } }
}

function pickFile(name = 'page.png') {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['x'], name, { type: 'image/png' })
  fireEvent.change(input, { target: { files: [file] } })
  return file
}

/** Requests to /detect only — the engines lookup on mount is not a detection. */
function detectCalls(fetchMock: ReturnType<typeof stubFetch>) {
  return fetchMock.mock.calls.filter((c) => String(c[0]).includes('/detect'))
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('detection is explicit', () => {
  it('sends nothing on mount', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(detectCalls(fetchMock)).toHaveLength(0)
  })

  it('sends nothing when a file is chosen', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickFile()

    expect(detectCalls(fetchMock)).toHaveLength(0)
    expect(screen.getByText(/nothing has been sent to the API yet/i)).toBeDefined()
  })

  it('sends nothing when the engine changes', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickFile()
    fireEvent.change(screen.getByLabelText('Engine'), { target: { value: 'vlm' } })

    expect(detectCalls(fetchMock)).toHaveLength(0)
  })

  it('sends exactly one request when Detect is pressed', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickFile()
    fireEvent.click(screen.getByRole('button', { name: /detect/i }))

    await waitFor(() => expect(detectCalls(fetchMock)).toHaveLength(1))
  })

  it('keeps the Detect button disabled until an image is chosen', async () => {
    stubFetch()
    render(<App />)
    const button = screen.getByRole('button', { name: /detect/i }) as HTMLButtonElement
    expect(button.disabled).toBe(true)

    pickFile()
    expect(button.disabled).toBe(false)
  })

  it('warns that a shown result came from a different engine', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickFile()
    fireEvent.click(screen.getByRole('button', { name: /detect/i }))
    await waitFor(() => expect(detectCalls(fetchMock)).toHaveLength(1))

    fireEvent.change(screen.getByLabelText('Engine'), { target: { value: 'assisted' } })

    // Silently relabelling the on-screen result as the newly-selected engine would be a
    // small lie with real consequences: it is the comparison the whole UI exists to support.
    await waitFor(() =>
      expect(screen.getByText(/Showing the previous result/i)).toBeDefined(),
    )
    expect(detectCalls(fetchMock)).toHaveLength(1)
  })

  it('warns that the selected engine costs money', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    expect(screen.queryByText(/costs money per run/i)).toBeNull()
    fireEvent.change(screen.getByLabelText('Engine'), { target: { value: 'vlm' } })
    expect(screen.getByText(/costs money per run/i)).toBeDefined()
  })
})
