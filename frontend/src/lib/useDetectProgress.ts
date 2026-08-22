import { useEffect, useState } from 'react'

/**
 * How long a page typically takes, in ms. Measured on the four samples through the running
 * stack: 2550x4200 scans land around 8-9 seconds, the smaller crop under three.
 */
const EXPECTED_MS = 8500

/**
 * The bar stops here while waiting. It never reaches 100 on its own.
 *
 * This is the whole honesty of the component. The API returns one response with no
 * intermediate events, so any progress shown before it arrives is an estimate — and an
 * estimate that reaches 100% and then sits there is worse than no bar at all, because it
 * reports a completion that has not happened. Stopping short leaves the last movement to be
 * caused by the thing actually finishing.
 */
const CEILING = 92

const TICK_MS = 120

/**
 * A time-based estimate of detection progress, in percent.
 *
 * Returns 0 when idle, climbs while `busy`, and is not a measurement — the backend reports no
 * progress, and pretending otherwise would be a lie told with a widget. What it does buy is
 * the difference between "something is happening and roughly this far along" and a spinner
 * that conveys only "wait": on an 8-second request those are very different experiences.
 *
 * The curve eases out rather than running linearly, so a page that takes longer than expected
 * slows down instead of stalling at a hard stop — which reads as progress continuing, which is
 * true, rather than as a hang, which is not.
 */
export function useDetectProgress(busy: boolean): number {
  const [percent, setPercent] = useState(0)

  useEffect(() => {
    if (!busy) {
      // Reset immediately rather than animating back: the next run should start from empty,
      // and a bar that visibly drains after a result is noise about work already finished.
      setPercent(0)
      return
    }
    const started = Date.now()
    const id = setInterval(() => {
      const elapsed = Date.now() - started
      // 1 - e^(-t/T) approaches the ceiling asymptotically: fast at first, never arriving.
      setPercent(Math.min(CEILING, CEILING * (1 - Math.exp(-elapsed / EXPECTED_MS))))
    }, TICK_MS)
    return () => clearInterval(id)
  }, [busy])

  return percent
}
