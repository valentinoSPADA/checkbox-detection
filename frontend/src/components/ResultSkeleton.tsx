interface ResultSkeletonProps {
  /** Estimated progress, 0-100. See lib/useDetectProgress for why it is an estimate. */
  percent: number
  /** Height of the page placeholder, matched to the real preview frame. */
  height: number
}

/**
 * The placeholder shown while a page is being detected.
 *
 * It is the shape of the result, not a spinner. The result is a row of stat cards above a tall
 * page image, and a small spinner in the middle of an empty column meant the whole viewer
 * jumped when the response landed. Reserving the same geometry removes that reflow, and tells
 * the reader what is coming rather than only that something is.
 *
 * Announced through one `role="status"` on the wrapper with `aria-busy`, so a screen reader
 * hears "detecting" once instead of narrating a dozen decorative boxes. The individual
 * skeletons are `aria-hidden` for the same reason.
 */
export function ResultSkeleton({ percent, height }: ResultSkeletonProps) {
  return (
    <div role="status" aria-busy="true" aria-live="polite" style={{ display: 'grid', gap: 14 }}>
      <div className="progress__label">
        <span>Detecting…</span>
        <span
          style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
          aria-hidden="true"
        >
          {Math.round(percent)}%
        </span>
      </div>
      {/*
        aria-valuenow is deliberately omitted: the value is a time-based guess, and exposing it
        as a measured percentage to assistive tech would assert a precision that does not exist.
        The visible number is decorative; the status message above is the real announcement.
      */}
      <div className="progress" aria-hidden="true">
        <div className="progress__fill" style={{ width: `${percent}%` }} />
      </div>

      {/* Five cards, matching the five stats a finished result shows. */}
      <div className="stats" aria-hidden="true">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className="card stat">
            <span className="skeleton skeleton--line" style={{ width: '55%', height: 20 }} />
            <span className="skeleton skeleton--line" style={{ width: '75%', marginTop: 8 }} />
          </div>
        ))}
      </div>

      <div className="card viewer__card" aria-hidden="true">
        <div className="viewer__head">
          <span className="skeleton skeleton--line" style={{ width: 150 }} />
          <span className="skeleton skeleton--line" style={{ width: 190 }} />
        </div>
        <div className="preview">
          <span className="skeleton skeleton--page" style={{ height }} />
        </div>
      </div>
    </div>
  )
}
