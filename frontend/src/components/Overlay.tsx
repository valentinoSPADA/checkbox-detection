import { useMemo, useState } from 'react'
import type { DetectedBox } from '../lib/api'

interface OverlayProps {
  imageUrl: string
  width: number
  height: number
  /** Already filtered to the current threshold by the caller — see the note below. */
  boxes: DetectedBox[]
  zoom: number
}

/**
 * Draws detection boxes over the page.
 *
 * An SVG overlay with a `viewBox` in the image's own pixel space is used rather than a
 * canvas, because it makes the coordinate mapping exact and free: box coordinates are
 * written straight from the API response with no scaling arithmetic in the UI, which is the
 * single place a subtle off-by-a-ratio bug would otherwise hide. It also keeps strokes crisp
 * at any zoom and each box hoverable as a real DOM node.
 *
 * Stroke width is divided by zoom so outlines stay one screen pixel wide as the user zooms
 * in; without that, a 22 px checkbox at 4x disappears under its own outline.
 *
 * Sub-threshold boxes are not drawn at all — the caller filters before passing them in. An
 * earlier version dimmed them instead, on the theory that showing what the threshold rejected
 * was informative. In practice it read as the opposite: a rejected box is still drawn in the
 * colour of a decision the system did not make, so a page carrying faint red outlines looked
 * like a detector reporting false positives. Moving the threshold slider already makes the
 * trade visible, by changing what appears.
 */
export function Overlay({ imageUrl, width, height, boxes, zoom }: OverlayProps) {
  const [hovered, setHovered] = useState<number | null>(null)

  // Sorted so lower-confidence boxes paint first and the confident ones sit on top: where two
  // detections overlap, the one the model is surer of should be the one you can hover.
  const ordered = useMemo(
    () => boxes.map((b, i) => ({ box: b, index: i })).sort((a, b) => a.box.confidence - b.box.confidence),
    [boxes],
  )

  const strokeWidth = Math.max(1, Math.round(2 / zoom))

  return (
    <div className="overlay-wrap" style={{ width: width * zoom }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width * zoom}
        height={height * zoom}
        className="overlay-svg"
        role="img"
        aria-label={`Document with ${boxes.length} detected checkboxes`}
      >
        <image href={imageUrl} x={0} y={0} width={width} height={height} />
        {ordered.map(({ box, index }) => {
          const [x1, y1, x2, y2] = box.bbox
          const colour = box.is_checked ? '#e5484d' : '#30a46c'
          return (
            <g key={index}>
              <rect
                x={x1}
                y={y1}
                width={Math.max(1, x2 - x1)}
                height={Math.max(1, y2 - y1)}
                fill={hovered === index ? colour : 'none'}
                fillOpacity={hovered === index ? 0.25 : 0}
                stroke={colour}
                strokeWidth={strokeWidth}
                onMouseEnter={() => setHovered(index)}
                onMouseLeave={() => setHovered(null)}
              >
                <title>
                  {`${box.is_checked ? 'checked' : 'unchecked'} · ${(box.confidence * 100).toFixed(0)}% · ${box.source}`}
                </title>
              </rect>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
