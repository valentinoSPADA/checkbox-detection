import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  ENGINE_INFO,
  detect,
  listEngines,
  type DetectResponse,
  type EngineName,
} from './lib/api'
import { applyTheme, initialTheme, type Theme } from './lib/theme'
import { Overlay } from './components/Overlay'
import { Lightbox } from './components/Lightbox'
import { Icon } from './components/Icon'
import { HomeVisionLogo } from './components/HomeVisionLogo'
import { ResultSkeleton } from './components/ResultSkeleton'
import { useDetectProgress } from './lib/useDetectProgress'

/** Height of the inline preview frame, in CSS px. See `.preview` in styles.css for why. */
const PREVIEW_HEIGHT = 420

/**
 * Upload guards, mirroring the ones the API enforces.
 *
 * Duplicated on the client deliberately. The server's limits are the real ones — a browser
 * check is trivially bypassed and is not a security boundary — but rejecting a 200 MB file
 * here saves the user a minute of uploading to earn a 413, which is the worse experience for
 * exactly the person who did nothing malicious.
 */
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024
const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/tiff']

/**
 * The whole UI.
 *
 * One screen, no router and no state library: the app has a single job — make the detector's
 * output inspectable — and a reviewer judging detection quality needs to see boxes on a page,
 * not an application framework. Anything more would be scope the challenge did not ask for.
 */
export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string>('')
  const [result, setResult] = useState<DetectResponse | null>(null)
  const [error, setError] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [engine, setEngine] = useState<EngineName>('local')
  // Matches the server's calibrated default (domain.DefaultPolicy). Starting lower would
  // greet the user with the noise floor and make a working detector look broken before they
  // have touched anything.
  const [threshold, setThreshold] = useState(0.9)
  const [dragging, setDragging] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const progress = useDetectProgress(busy)

  // Held in a ref so a new run cancels the previous one, and so a stale response can never
  // land after a fresh one and overwrite it.
  const inflight = useRef<AbortController | null>(null)

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    const ctl = new AbortController()
    listEngines(ctl.signal)
      .then((r) => setEngine(r.default))
      .catch(() => {
        // Non-fatal: `local` is what every instance registers.
      })
    return () => ctl.abort()
  }, [])

  // Object URLs are revoked when the file changes, otherwise every upload in a session
  // leaks the full decoded bitmap — noticeable quickly with 2550x4200 scans.
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const run = useCallback(async (target: File, which: EngineName) => {
    inflight.current?.abort()
    const ctl = new AbortController()
    inflight.current = ctl
    setBusy(true)
    setError('')
    try {
      // A low floor is requested and the threshold applied client-side, so moving the
      // slider is instant and shows what was rejected instead of re-running detection.
      const res = await detect(target, { engine: which, minConfidence: 0.05, signal: ctl.signal })
      setResult(res)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setError(err instanceof ApiError ? `${err.status}: ${err.message}` : String(err))
      setResult(null)
    } finally {
      if (inflight.current === ctl) setBusy(false)
    }
  }, [])

  /**
   * Accept a file, or say why not.
   *
   * Selecting a file loads it for display but does NOT detect. Choosing what to run and
   * choosing to run it are different decisions, and an earlier auto-run conflated them:
   * picking a file fired a request the user never asked for.
   */
  const onFile = useCallback((f: File) => {
    setResult(null)
    if (f.size > MAX_UPLOAD_BYTES) {
      setFile(null)
      setError(
        `“${f.name}” is ${(f.size / 1024 / 1024).toFixed(1)} MB; the limit is ${
          MAX_UPLOAD_BYTES / 1024 / 1024
        } MB.`,
      )
      return
    }
    // An empty type is allowed through rather than rejected: some browsers report nothing for
    // a dragged file, and the server sniffs the actual bytes anyway. Guessing here would block
    // a valid upload in order to enforce a check that is not this layer's job.
    if (f.type && !ACCEPTED_TYPES.includes(f.type)) {
      setFile(null)
      setError(`“${f.name}” is ${f.type}. Expected a PNG, JPEG, WEBP or TIFF image.`)
      return
    }
    setError('')
    setFile(f)
  }, [])

  const onDetect = useCallback(() => {
    if (!file || busy) return
    void run(file, engine)
  }, [busy, engine, file, run])

  const visible = useMemo(
    () => (result?.boxes ?? []).filter((b) => b.confidence >= threshold),
    [result, threshold],
  )
  const checkedCount = visible.filter((b) => b.is_checked).length
  const caption = result ? `${result.boxes.length} candidates · ${visible.length} above threshold` : ''

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark">
            <Icon name="checkbox" size={18} strokeWidth={2.2} />
          </span>
          <span>
            <span className="brand__name" style={{ display: 'block' }}>
              Checkbox Detection
            </span>
            <span className="brand__sub">Appraisal document extraction</span>
          </span>
        </div>

        {/*
          Attribution, not branding. The logo is never shown on its own — the sentence around
          it is what makes clear this is a submission *for* HomeVision rather than a product
          *of* HomeVision, so the wording and the mark travel together as one element.
        */}
        <p className="attrib">
          Take-home challenge for
          <HomeVisionLogo />
        </p>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="pill">
            <span className={`pill__dot${error ? ' pill__dot--down' : ''}`} />
            {error ? 'Engine error' : 'Engine online'}
          </span>
          <button
            className="btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="layout">
        <div className="rail">
          <div className="card card--pad">
            <label
              className={`drop ${dragging ? 'drop--active' : ''}`}
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                const f = e.dataTransfer.files?.[0]
                if (f) onFile(f)
              }}
            >
              <input
                type="file"
                accept={ACCEPTED_TYPES.join(',')}
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) onFile(f)
                }}
              />
              <Icon name="upload" size={22} />
              {file ? (
                <>
                  <span className="drop__name">{file.name}</span>
                  <span className="hint">Drop another page, or click to replace</span>
                </>
              ) : (
                <>
                  <span className="hint">Drop a document image here, or click to choose</span>
                  <span className="hint">PNG, JPEG, WEBP or TIFF · up to 25 MB</span>
                </>
              )}
            </label>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <button className="btn btn--primary" onClick={onDetect} disabled={!file || busy}>
                <Icon name="scan" size={17} />
                {busy ? 'Detecting…' : result ? 'Detect again' : 'Detect'}
              </button>

              {!file && !error && <span className="hint">Choose an image first.</span>}
              {file && !result && !busy && !error && (
                <span className="hint">Ready — nothing has been sent to the API yet.</span>
              )}
              {error && (
                <span className="notice notice--error">
                  <Icon name="alert" size={15} />
                  <span>{error}</span>
                </span>
              )}
            </div>
          </div>

          <div className="card card--pad" style={{ gap: 9 }}>
            <div
              style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}
            >
              <span className="label">Confidence</span>
              <span
                style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                ≥ {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={threshold}
              aria-label="Confidence threshold"
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className="hint">
              Calibrated default. Lower it to see what the model rejected.
            </span>
          </div>
        </div>

        <div className="viewer">
          {busy && <ResultSkeleton percent={progress} height={PREVIEW_HEIGHT} />}

          {!busy && result && (
            <>
              <div className="stats">
                <Stat value={visible.length} label="Shown" />
                <Stat value={checkedCount} label="Checked" tone="checked" />
                <Stat value={visible.length - checkedCount} label="Unchecked" tone="unchecked" />
                <Stat value={ENGINE_INFO[result.meta.engine].label} label="Engine" />
                <Stat value={formatLatency(result.meta.elapsed_ms)} label="Latency" />
              </div>

              <div className="card viewer__card">
                <div className="viewer__head">
                  <div className="legend">
                    <span className="legend__item">
                      <span className="legend__swatch legend__swatch--checked" />
                      checked
                    </span>
                    <span className="legend__item">
                      <span className="legend__swatch" />
                      unchecked
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    <span className="viewer__meta">{caption}</span>
                    <button className="btn" onClick={() => setExpanded(true)}>
                      <Icon name="expand" size={14} />
                      Expand
                    </button>
                  </div>
                </div>

                <div className="preview" title="Click to open at full size">
                  <Overlay
                    imageUrl={imageUrl}
                    width={result.meta.width}
                    height={result.meta.height}
                    boxes={visible}
                    sizing={{ mode: 'fit', maxHeight: PREVIEW_HEIGHT }}
                    onClick={() => setExpanded(true)}
                  />
                </div>
              </div>

              <p className="viewer__foot">
                <Icon name="info" size={14} />
                Boxes below the confidence threshold are hidden, not dimmed — a rejected box
                drawn in the colour of a decision the system did not make reads as a false
                positive.
              </p>
            </>
          )}
        </div>
      </main>

      {expanded && result && (
        <Lightbox
          imageUrl={imageUrl}
          width={result.meta.width}
          height={result.meta.height}
          boxes={visible}
          filename={file?.name ?? 'document'}
          caption={caption}
          onClose={() => setExpanded(false)}
        />
      )}
    </>
  )
}

/**
 * Latency in the unit that carries information at that magnitude.
 *
 * Always formatting as seconds printed "0.0s" for a fast local run, which reads as broken
 * rather than fast; below a second the milliseconds are the interesting number.
 */
function formatLatency(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

function Stat({
  value,
  label,
  tone,
}: {
  value: string | number
  label: string
  tone?: 'checked' | 'unchecked'
}) {
  return (
    <div className="card stat">
      <span className={`stat__value${tone ? ` stat__value--${tone}` : ''}`}>{value}</span>
      <span className="stat__label">{label}</span>
    </div>
  )
}
