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

/** The engine picker is a segmented radiogroup, so selection is a click, not a change. */
function pickEngine(label: string) {
  fireEvent.click(screen.getByRole('radio', { name: label }))
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
  // The theme is persisted, so without this a render in one test decides the starting
  // theme of the next one — the kind of order dependence that makes a suite flaky when
  // tests are reordered or run in isolation.
  localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
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
    pickEngine('Claude vision')

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

    pickEngine('Assisted')

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
    pickEngine('Claude vision')
    expect(screen.getByText(/costs money per run/i)).toBeDefined()
  })
})

describe('theme', () => {
  it('starts from the OS preference and flips on demand', async () => {
    // A first visit must not override a deliberate system-wide dark setting.
    vi.stubGlobal('matchMedia', (q: string) => ({
      matches: q.includes('dark'),
      media: q,
      addEventListener() {},
      removeEventListener() {},
    }))
    stubFetch()
    render(<App />)

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')

    fireEvent.click(screen.getByRole('button', { name: /switch to light theme/i }))
    await waitFor(() =>
      expect(document.documentElement.getAttribute('data-theme')).toBe('light'),
    )
  })

  it('remembers the choice across mounts', async () => {
    vi.stubGlobal('matchMedia', () => ({ matches: false, addEventListener() {}, removeEventListener() {} }))
    stubFetch()
    const first = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /switch to dark theme/i }))
    await waitFor(() => expect(document.documentElement.getAttribute('data-theme')).toBe('dark'))
    first.unmount()

    render(<App />)
    // The stored preference wins over the OS default on the next visit.
    await waitFor(() => expect(document.documentElement.getAttribute('data-theme')).toBe('dark'))
  })
})

describe('lightbox', () => {
  async function renderWithResult() {
    const fetchMock = stubFetch({
      boxes: [
        { bbox: [10, 20, 32, 42], is_checked: true, confidence: 0.98, source: 'local' },
        { bbox: [60, 20, 82, 42], is_checked: false, confidence: 0.97, source: 'local' },
      ],
      meta: { ...emptyMeta(), width: 400, height: 300, stats: { candidates: 2, returned: 2 } },
    })
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    pickFile()
    fireEvent.click(screen.getByRole('button', { name: /^detect$/i }))
    await waitFor(() => expect(detectCalls(fetchMock)).toHaveLength(1))
  }

  it('opens from Expand and closes again', async () => {
    await renderWithResult()
    expect(screen.queryByRole('dialog')).toBeNull()

    fireEvent.click(await screen.findByRole('button', { name: /expand/i }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toBeDefined()

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('closes on Escape', async () => {
    await renderWithResult()
    fireEvent.click(await screen.findByRole('button', { name: /expand/i }))
    await screen.findByRole('dialog')

    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('zooms within bounds', async () => {
    await renderWithResult()
    fireEvent.click(await screen.findByRole('button', { name: /expand/i }))
    await screen.findByRole('dialog')

    expect(screen.getByText('100%')).toBeDefined()
    fireEvent.click(screen.getByRole('button', { name: /zoom in/i }))
    expect(screen.getByText('125%')).toBeDefined()

    // The floor must hold: a zoom that can reach 0 renders an invisible page.
    const out = screen.getByRole('button', { name: /zoom out/i })
    for (let i = 0; i < 12; i++) fireEvent.click(out)
    expect(screen.getByText('50%')).toBeDefined()
    expect((out as HTMLButtonElement).disabled).toBe(true)
  })

  it('actually scales the image, not just the label', async () => {
    // The zoom label and the rendered size are separate things, and they came apart once:
    // the .overlay rule caps width at 100% so the fitted preview cannot overflow its card,
    // and that cap silently swallowed every level above 100%. jsdom computes no layout, so
    // the assertion is on the inline style that drives it.
    await renderWithResult()
    fireEvent.click(await screen.findByRole('button', { name: /expand/i }))
    const dialog = await screen.findByRole('dialog')

    const overlay = dialog.querySelector('.overlay') as HTMLElement
    expect(overlay.style.width).toBe('100%')
    expect(overlay.style.maxWidth).toBe('none')

    fireEvent.click(screen.getByRole('button', { name: /zoom in/i }))
    expect((dialog.querySelector('.overlay') as HTMLElement).style.width).toBe('125%')
  })

  it('restores page scrolling when it closes', async () => {
    // A dialog that leaks overflow:hidden leaves the page unscrollable after dismissal.
    await renderWithResult()
    fireEvent.click(await screen.findByRole('button', { name: /expand/i }))
    await screen.findByRole('dialog')
    expect(document.body.style.overflow).toBe('hidden')

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    await waitFor(() => expect(document.body.style.overflow).not.toBe('hidden'))
  })
})
