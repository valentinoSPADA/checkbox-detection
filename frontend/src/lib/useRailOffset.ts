import { useEffect, type RefObject } from 'react'

/** Width of the rail column, matching `grid-template-columns` in styles.css. */
const RAIL_WIDTH = 340

/** Below this the layout is a single column, so there is nowhere to slide to. */
const SINGLE_COLUMN_BELOW = 1000

/**
 * Publishes, as a CSS custom property in pixels, how far the rail must move to sit centred.
 *
 * This exists because the obvious pure-CSS form does not animate. Writing the offset as
 * `translateX(calc((min(100vw, 1600px) - 64px - 340px) / 2))` produces the correct position,
 * but no transition ever starts: measured in the browser, `element.getAnimations()` came back
 * empty and the rail jumped between the two positions in one frame. Substituting a plain
 * `translateX(454px)` under otherwise identical conditions started a transition immediately.
 * The engine will not interpolate a transform whose endpoint still carries unresolved
 * viewport math, so the value has to arrive already resolved.
 *
 * A ResizeObserver rather than a resize listener: the layout is capped at 1600px and can stop
 * growing while the window keeps growing, and it can also change width without the window
 * resizing at all — a scrollbar appearing when the first result renders does exactly that.
 * Watching the element measures the thing the offset is actually derived from.
 */
export function useRailOffset(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return

    const measure = () => {
      const style = getComputedStyle(el)
      const inner =
        el.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
      // Clamped at zero so a narrow window can never push the rail off its own left edge.
      const offset =
        el.clientWidth < SINGLE_COLUMN_BELOW ? 0 : Math.max(0, (inner - RAIL_WIDTH) / 2)
      el.style.setProperty('--rail-offset', `${offset}px`)
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
}
