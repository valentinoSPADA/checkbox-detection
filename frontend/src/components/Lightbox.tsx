import { useCallback, useEffect, useState } from 'react'
import type { DetectedBox } from '../lib/api'
import { Overlay } from './Overlay'
import { Icon } from './Icon'

interface LightboxProps {
  imageUrl: string
  width: number
  height: number
  boxes: DetectedBox[]
  filename: string
  caption: string
  onClose: () => void
}

const MIN_ZOOM = 50
const MAX_ZOOM = 400
const STEP = 25

/**
 * Full-window inspection of one page.
 *
 * Free zoom lives here rather than in the main view on purpose: pages arrive at very
 * different aspect ratios, and letting each one drive the size of the inline preview made
 * the whole layout jump between documents. The preview stays a predictable frame; this is
 * where a 2550x4200 scan gets looked at closely.
 */
export function Lightbox({
  imageUrl,
  width,
  height,
  boxes,
  filename,
  caption,
  onClose,
}: LightboxProps) {
  const [zoom, setZoom] = useState(100)

  // Escape closes. Registered on keydown rather than keyup so it fires before anything else
  // reacts to the key, and removed on unmount so it cannot outlive the dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The page behind must not scroll while a full-window dialog is open, or dismissing it
  // leaves the reader somewhere they never navigated to.
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [])

  const zoomIn = useCallback(() => setZoom((z) => Math.min(MAX_ZOOM, z + STEP)), [])
  const zoomOut = useCallback(() => setZoom((z) => Math.max(MIN_ZOOM, z - STEP)), [])

  return (
    <div className="lightbox" role="dialog" aria-modal="true" aria-label={`${filename}, full size`}>
      <div className="lightbox__bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, minWidth: 0 }}>
          <span className="lightbox__title">{filename}</span>
          <span className="lightbox__meta">{caption}</span>
        </div>

        <div className="lightbox__tools">
          <div className="zoomgroup">
            <button
              className="zoomgroup__btn"
              onClick={zoomOut}
              disabled={zoom <= MIN_ZOOM}
              aria-label="Zoom out"
              title="Zoom out"
            >
              <Icon name="minus" size={15} />
            </button>
            <span className="zoomgroup__value">{zoom}%</span>
            <button
              className="zoomgroup__btn"
              onClick={zoomIn}
              disabled={zoom >= MAX_ZOOM}
              aria-label="Zoom in"
              title="Zoom in"
            >
              <Icon name="plus" size={15} />
            </button>
          </div>
          <button className="lightbox__ghost" onClick={() => setZoom(100)}>
            Fit
          </button>
          <button
            className="lightbox__ghost lightbox__close"
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            <Icon name="close" size={16} />
          </button>
        </div>
      </div>

      {/* Clicking the backdrop closes; the click is stopped on the image itself so that
          inspecting a box never dismisses the thing being inspected. */}
      <div className="lightbox__stage" onClick={onClose}>
        <div onClick={(e) => e.stopPropagation()} style={{ display: 'contents' }}>
          <Overlay
            imageUrl={imageUrl}
            width={width}
            height={height}
            boxes={boxes}
            sizing={{ mode: 'zoom', percent: zoom }}
          />
        </div>
      </div>

      <div className="lightbox__foot">
        <span className="legend__item">
          <span className="legend__swatch legend__swatch--checked" />
          checked
        </span>
        <span className="legend__item">
          <span className="legend__swatch" />
          unchecked
        </span>
        <span className="lightbox__meta">Click the backdrop or press Esc to close</span>
      </div>
    </div>
  )
}
