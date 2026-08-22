import type { DetectedBox } from '../lib/api'

/** How the image is sized. `fit` contains it in a frame; `zoom` scales it to a percentage. */
export type Sizing = { mode: 'fit'; maxHeight: number } | { mode: 'zoom'; percent: number }

interface OverlayProps {
  imageUrl: string
  /** Natural size of the source image — box coordinates are in these pixels. */
  width: number
  height: number
  /** Already filtered to the current threshold by the caller. */
  boxes: DetectedBox[]
  sizing: Sizing
  onClick?: () => void
}

/**
 * The page image with its detection boxes drawn over it.
 *
 * Boxes are positioned as PERCENTAGES of a wrapper that shrink-wraps the rendered image
 * (`display: inline-block`), which is what lets the same component serve both the fixed
 * preview and the zoomable lightbox: whatever size the browser renders the image at, the
 * wrapper matches it exactly and the percentages stay aligned. Sizing the wrapper instead —
 * to the card, say — would misplace every box on any page whose aspect ratio differs from
 * the frame's, and appraisal pages differ wildly (1586x846 against 2550x4200).
 *
 * Border width is a constant 2 CSS px rather than scaled with zoom, so an outline stays
 * readable at 50% and does not swallow a 22 px checkbox at 400%.
 *
 * Sub-threshold boxes are not drawn at all — the caller filters before passing them in. An
 * earlier version dimmed them, on the theory that showing what the threshold rejected was
 * informative; it read as the opposite, since a rejected box is still drawn in the colour of
 * a decision the system did not make. Moving the threshold slider already makes the trade
 * visible, by changing what appears.
 */
export function Overlay({ imageUrl, width, height, boxes, sizing, onClick }: OverlayProps) {
  // `maxWidth: none` is load-bearing in zoom mode: the `.overlay` rule caps width at 100% so
  // the fitted preview can never overflow its card, and that same cap silently swallowed
  // every zoom level above 100% — the label moved, the image did not.
  const wrapperStyle =
    sizing.mode === 'zoom'
      ? { width: `${sizing.percent}%`, maxWidth: 'none', flexShrink: 0 }
      : undefined
  const imgStyle =
    sizing.mode === 'zoom'
      ? { width: '100%', height: 'auto' as const }
      : { maxHeight: sizing.maxHeight, maxWidth: '100%', width: 'auto' as const, height: 'auto' as const }

  // When the caller makes the preview clickable it becomes a control, and a control has to be
  // reachable without a mouse. Rather than a div with a click handler -- which no keyboard user
  // can operate -- it takes button semantics, focus, and Enter/Space. Without `onClick` it stays
  // a plain image and is given no interactive role it does not have.
  const interactive = onClick
    ? {
        role: 'button' as const,
        tabIndex: 0,
        'aria-label': 'Open the page at full size',
        onClick,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onClick()
          }
        },
      }
    : {}

  return (
    <div className="overlay" style={wrapperStyle} {...interactive}>
      <img src={imageUrl} alt={`Document page with ${boxes.length} detected checkboxes`} style={imgStyle} />
      {boxes.map((box, index) => {
        const [x1, y1, x2, y2] = box.bbox
        return (
          <div
            key={`${x1}-${y1}-${x2}-${y2}-${index}`}
            className={`overlay__box${box.is_checked ? ' overlay__box--checked' : ''}`}
            style={{
              left: `${(x1 / width) * 100}%`,
              top: `${(y1 / height) * 100}%`,
              width: `${((x2 - x1) / width) * 100}%`,
              height: `${((y2 - y1) / height) * 100}%`,
            }}
            title={`${box.is_checked ? 'checked' : 'unchecked'} · ${(box.confidence * 100).toFixed(0)}% · ${box.source}`}
          />
        )
      })}
    </div>
  )
}
