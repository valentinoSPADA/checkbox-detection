import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from './App'

/**
 * "Nothing is sent until the user asks" is a behavioural guarantee, not a UI preference, and
 * these tests are what hold it. It was written when two of the three engines billed per page;
 * those engines are gone and the rule outlived them, because an auto-run still means a request
 * the user did not ask for and several seconds of CPU they did not want to spend.
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

describe('loading state', () => {
  /** A detect call that never settles, so the pending UI can be inspected. */
  function stubHangingDetect() {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).includes('/engines')) {
        return { ok: true, status: 200, json: async () => ({ engines: ['local'], default: 'local' }) } as Response
      }
      return new Promise<Response>(() => {})
    })
    vi.stubGlobal('fetch', fetchMock)
    return fetchMock
  }

  it('shows a skeleton of the result rather than a bare spinner', async () => {
    // The point of the skeleton is that it reserves the result's geometry, so the viewer does
    // not reflow when the response lands. Assert the shape, not the animation.
    const fetchMock = stubHangingDetect()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    pickFile()
    fireEvent.click(screen.getByRole('button', { name: /^detect$/i }))

    const status = await screen.findByRole('status')
    expect(status.querySelectorAll('.skeleton').length).toBeGreaterThan(5)
    expect(status.querySelector('.skeleton--page')).not.toBeNull()
    expect(status.getAttribute('aria-busy')).toBe('true')
  })

  it('advances the progress bar while waiting', async () => {
    vi.useFakeTimers()
    try {
      stubHangingDetect()
      render(<App />)
      pickFile()
      fireEvent.click(screen.getByRole('button', { name: /^detect$/i }))

      const width = () =>
        parseFloat((document.querySelector('.progress__fill') as HTMLElement).style.width)
      await act(async () => {
        vi.advanceTimersByTime(500)
      })
      const early = width()
      await act(async () => {
        vi.advanceTimersByTime(4000)
      })
      expect(width()).toBeGreaterThan(early)
    } finally {
      vi.useRealTimers()
    }
  })

  it('never lets the estimate claim completion', async () => {
    // A bar that reaches 100% and sits there reports a finish that has not happened. The
    // last movement must be caused by the response actually arriving.
    vi.useFakeTimers()
    try {
      stubHangingDetect()
      render(<App />)
      pickFile()
      fireEvent.click(screen.getByRole('button', { name: /^detect$/i }))
      await act(async () => {
        vi.advanceTimersByTime(10 * 60 * 1000)
      })
      const width = parseFloat(
        (document.querySelector('.progress__fill') as HTMLElement).style.width,
      )
      expect(width).toBeLessThan(100)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('upload guards', () => {
  /** Attach a file of a given size and MIME type without allocating its bytes. */
  function pickSized(name: string, size: number, type: string) {
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(['x'], name, { type })
    Object.defineProperty(file, 'size', { value: size })
    fireEvent.change(input, { target: { files: [file] } })
  }

  it('refuses an oversized file without contacting the API', async () => {
    // The server enforces this too; rejecting here saves the user a long upload that can only
    // end in a 413.
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickSized('huge.png', 40 * 1024 * 1024, 'image/png')

    expect(await screen.findByText(/the limit is 25 MB/i)).toBeDefined()
    expect(detectCalls(fetchMock)).toHaveLength(0)
    expect((screen.getByRole('button', { name: /detect/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('refuses a file whose type the sidecar cannot decode', async () => {
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickSized('report.pdf', 1024, 'application/pdf')

    expect(await screen.findByText(/expected a PNG, JPEG, WEBP or TIFF/i)).toBeDefined()
    expect(detectCalls(fetchMock)).toHaveLength(0)
  })

  it('accepts a file the browser reports no type for', async () => {
    // Some browsers report an empty type for a dragged file. Guessing here would block a
    // valid upload to enforce a check the server does properly, on the bytes.
    const fetchMock = stubFetch()
    render(<App />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    pickSized('scan', 2048, '')

    expect(screen.queryByText(/Expected a PNG/i)).toBeNull()
    expect((screen.getByRole('button', { name: /detect/i }) as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('attribution', () => {
  /**
   * The HomeVision mark is third-party branding on a page that is not theirs. These tests
   * exist so a refactor cannot quietly turn attribution into what looks like ownership —
   * dropping the label and leaving the logo would do exactly that, and would still render
   * fine, which is why it needs a test rather than a comment.
   */
  it('never shows the logo without the sentence that frames it', async () => {
    stubFetch()
    render(<App />)

    const logo = screen.getByRole('img', { name: 'HomeVision' })
    const line = logo.closest('.attrib')
    expect(line).not.toBeNull()
    expect(line!.textContent).toMatch(/take-home challenge for/i)
  })

  it('keeps the app’s own identity primary', async () => {
    // The header must still say what this thing is. If the only name in it were HomeVision's,
    // the page would read as their product regardless of the label.
    stubFetch()
    render(<App />)
    expect(screen.getByText('Checkbox Detection')).toBeDefined()
  })

  it('announces the whole phrase to a screen reader, not just the mark', async () => {
    stubFetch()
    render(<App />)
    const line = screen.getByRole('img', { name: 'HomeVision' }).closest('.attrib')
    expect(line!.textContent!.replace(/\s+/g, ' ').trim()).toBe('Take-home challenge for')
  })
})
