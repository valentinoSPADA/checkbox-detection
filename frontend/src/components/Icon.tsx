/**
 * The icon set, as inline stroke SVG.
 *
 * Drawn rather than pulled from a font or an emoji: these scale and recolour with the
 * surrounding text, they carry no network dependency (the artifact CSP allows no CDN), and
 * a stroke set on one grid reads as one family where mixed sources never do. All paths are
 * on a 24-unit grid at stroke-width 2 so weights match across sizes.
 */

export type IconName =
  | 'checkbox'
  | 'sun'
  | 'moon'
  | 'upload'
  | 'scan'
  | 'expand'
  | 'close'
  | 'plus'
  | 'minus'
  | 'info'
  | 'alert'

const PATHS: Record<IconName, React.ReactNode> = {
  checkbox: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      <path d="M8 12.5l2.5 2.5L16 9" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />,
  upload: (
    <>
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <path d="M7 9l5-5 5 5" />
      <path d="M12 4v12" />
    </>
  ),
  scan: (
    <>
      <path d="M3 7V5a2 2 0 012-2h2M17 3h2a2 2 0 012 2v2M21 17v2a2 2 0 01-2 2h-2M7 21H5a2 2 0 01-2-2v-2" />
      <path d="M3 12h18" />
    </>
  ),
  expand: <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />,
  close: <path d="M18 6L6 18M6 6l12 12" />,
  plus: <path d="M12 5v14M5 12h14" />,
  minus: <path d="M5 12h14" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16v-4M12 8.5v.01" />
    </>
  ),
  alert: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v5M12 16.5v.01" />
    </>
  ),
}

interface IconProps {
  name: IconName
  size?: number
  /** Stroke weight. Raised slightly for the small brand mark, where 2 looks thin on colour. */
  strokeWidth?: number
}

/**
 * Renders one icon at `size` px, inheriting `currentColor`.
 *
 * `aria-hidden` on every instance: each icon in this UI sits next to a text label or inside
 * a button that carries its own `aria-label`, so announcing the glyph as well would only
 * duplicate what a screen reader already says.
 */
export function Icon({ name, size = 16, strokeWidth = 2 }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name]}
    </svg>
  )
}
