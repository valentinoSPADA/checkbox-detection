interface ResultSkeletonProps {
  /** Height of the page placeholder, matched to the real preview frame. */
  height: number
}

/**
 * The placeholder shown while a page is being detected.
 *
 * It is the shape of the result, not a spinner, and that has to be true to the pixel: the
 * status strip above it is rendered by the viewer in BOTH states, so nothing here sits at a
 * height it will not occupy once the response lands. An earlier version owned the strip
 * itself, and the stat cards jumped ~40px upward on arrival — which made the skeletons read
 * as unrelated boxes rather than as the things that were about to become the result.
 *
 * Announced through one `role="status"` on the wrapper with `aria-busy`, so a screen reader
 * hears "detecting" once instead of narrating a dozen decorative boxes. The individual
 * skeletons are `aria-hidden` for the same reason.
 */
export function ResultSkeleton({ height }: ResultSkeletonProps) {
  return (
    <div role="status" aria-busy="true" aria-live="polite" style={{ display: 'contents' }}>
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
          <span className="skeleton skeleton--line" style={{ width: 90 }} />
        </div>
        <div className="preview">
          <span className="skeleton skeleton--page" style={{ height }} />
        </div>
      </div>
    </div>
  )
}
